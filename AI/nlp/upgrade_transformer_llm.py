#!/usr/bin/env python3
"""
Tiny Transformer LLM — Advanced Edition (PyTorch)
- Byte-level tokenizer (works with any UTF‑8 text: Korean/English/emoji)
- GPT-style causal Transformer (multi-head attention)
- Train/val split, mixed precision, gradient accumulation, cosine LR w/ warmup
- Best/last checkpointing + resume, deterministic seed
- Flexible sampling: temperature, top_k, top_p (nucleus)
- Auto device: MPS (Apple Silicon) > CUDA > CPU

Usage (train):
  python tiny_transformer_llm_plus.py --data input_long.txt --steps 5000 --batch_size 64 --device auto

Usage (sample from checkpoint):
  python tiny_transformer_llm_plus.py --sample --checkpoint tinyllm_best.pt --prompt "인공지능은 " --max_new_tokens 300

Tested with Python 3.9+ and PyTorch >= 2.1.
Note: MPS works on Apple Silicon with macOS 12.3+ and recent PyTorch.
"""
import argparse
import os
import math
import time
import json
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- Utilities ----------------------
def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auto_device(pref: str):
    if pref == 'auto':
        if torch.backends.mps.is_available():
            print('Using device: mps (Apple Silicon GPU)')
            return torch.device('mps')
        if torch.cuda.is_available():
            print('Using device: cuda (NVIDIA GPU)')
            return torch.device('cuda')
        print('Using device: cpu')
        return torch.device('cpu')
    print(f'Using device (forced): {pref}')
    return torch.device(pref)

# ---------------------- Tokenizer ----------------------
class ByteTokenizer:
    """Byte-level tokenizer (GPT-2 style conceptually, but minimal):
    - Each UTF-8 byte (0..255) is a token.
    - Works for any language out of the box.
    """
    def __init__(self):
        self.vocab_size = 256
    def encode(self, s: str) -> torch.Tensor:
        b = s.encode('utf-8', errors='ignore')
        return torch.tensor(list(b), dtype=torch.long)
    def decode(self, ids: torch.Tensor) -> str:
        ba = bytes(int(i) for i in ids)
        return ba.decode('utf-8', errors='ignore')

# ---------------------- Model ----------------------
@dataclass
class GPTConfig:
    vocab_size: int
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    block_size: int = 256
    dropout: float = 0.1
    ln_eps: float = 1e-5

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.drop(y)
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.fc(x)

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.ln_eps)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.ln_eps)
        self.mlp = MLP(config.n_embd, config.dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.ln_eps)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, 'Sequence exceeds block_size'
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    @torch.autocast('cuda', enabled=False)
    def dummy(self):
        pass

@torch.no_grad()
def generate(model: TinyGPT,
             idx: torch.Tensor,
             max_new_tokens: int = 200,
             temperature: float = 0.8,
             top_k: Optional[int] = None,
             top_p: Optional[float] = 0.9,
             repetition_penalty: float = 1.1,
             freq_penalty: float = 0.0,
             pres_penalty: float = 0.0,
             penalty_recent_window: int = 256,
             allow_bytes: Optional[torch.Tensor] = None):
    """Advanced sampler with repetition/frequency/presence penalties and nucleus sampling.
    - repetition_penalty > 1.0 reduces probability of tokens seen recently
    - freq_penalty penalizes tokens proportional to count in recent window
    - pres_penalty penalizes tokens that appeared at least once in recent window
    - allow_bytes: optional mask over vocab (size 256) with True for allowed bytes
      (default allows all bytes except most control chars; keeps Korean UTF-8 intact)
    """
    model.eval()
    cfg = model.config
    device = next(model.parameters()).device

    # Default allow mask: allow TAB (9), LF (10), CR (13), and bytes 32..255
    # (so Korean/UTF-8 multi-byte outputs are not suppressed)
    if allow_bytes is None:
        mask = torch.zeros(cfg.vocab_size, dtype=torch.bool, device=device)
        allowed = [9, 10, 13] + list(range(32, 256))
        for b in allowed:
            mask[b] = True
        allow_bytes = mask

    recent = []  # track recent tokens for penalties

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # temperature
        logits = logits / max(1e-6, temperature)

        # ban disallowed bytes
        logits[:, ~allow_bytes] = -float('inf')

        # repetition / frequency / presence penalties
        if recent and (repetition_penalty > 1.0 or freq_penalty > 0.0 or pres_penalty > 0.0):
            window = recent[-penalty_recent_window:]
            counts = torch.bincount(torch.tensor(window, device=device), minlength=cfg.vocab_size).float()
            seen = counts > 0
            # repetition: downscale logits for seen tokens
            logits[:, seen] = logits[:, seen] / repetition_penalty
            # frequency: subtract proportional to frequency
            if freq_penalty > 0.0:
                logits = logits - freq_penalty * counts.unsqueeze(0)
            # presence: subtract a flat penalty if appeared at least once
            if pres_penalty > 0.0:
                logits[:, seen] = logits[:, seen] - pres_penalty

        probs = F.softmax(logits, dim=-1)

        # top-k
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(probs, top_k)
            cutoff = v[:, [-1]]
            probs = torch.where(probs < cutoff, torch.zeros_like(probs), probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # top-p (nucleus)
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_local = torch.multinomial(sorted_probs, num_samples=1)
            next_token = torch.gather(sorted_idx, -1, next_local)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = int(next_token.item())
        recent.append(token_id)
        idx = torch.cat([idx, next_token], dim=1)

    return idx

# ---------------------- Data ----------------------
def load_text(path: str) -> str:
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    # fallback sample
    return (
        '인공지능 언어 모델을 직접 만들어 보자. ' \
        'Transformer 구조는 자기회귀 방식으로 다음 문자를 예측한다.\n' \
        'Let us build a tiny GPT-like model from scratch.\n'
    )

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, ids: torch.Tensor, block_size: int):
        self.ids = ids
        self.block = block_size
    def __len__(self):
        return max(0, len(self.ids) - self.block - 1)
    def __getitem__(self, i):
        x = self.ids[i: i + self.block]
        y = self.ids[i + 1: i + 1 + self.block]
        return x, y

# ---------------------- Training ----------------------
def cosine_warmup(step, warmup, total_steps):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def evaluate(model, loader, device, amp_dtype):
    model.eval()
    total, count = 0.0, 0
    scaler_enabled = (amp_dtype is not None)
    autocast_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=scaler_enabled)
    with torch.no_grad(), autocast_ctx:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            total += loss.item()
            count += 1
    return total / max(1, count)


def train_loop(model, optimizer, train_loader, val_loader, device, steps, grad_accum, warmup, amp_dtype, log_interval, ckpt_best, ckpt_last, scheduler_total):
    scaler_enabled = (amp_dtype is not None)
    autocast_ctx = lambda: torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=scaler_enabled)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val = float('inf')
    step = 0
    ema = None
    start = time.time()

    while step < steps:
        for x, y in train_loader:
            model.train()
            x = x.to(device)
            y = y.to(device)
            with autocast_ctx():
                _, loss = model(x, y)
                loss = loss / grad_accum
            if device.type == 'cuda':
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % grad_accum == 0:
                if device.type == 'cuda':
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # LR schedule
            for pg in optimizer.param_groups:
                pg['lr'] = pg.get('base_lr', 3e-4) * cosine_warmup(step, warmup, scheduler_total)

            step += 1
            ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
            if step % log_interval == 0:
                elapsed = time.time() - start
                print(f"step {step:6d} | train_loss {ema:.4f} | lr {optimizer.param_groups[0]['lr']:.6f} | {elapsed:.1f}s")
                start = time.time()

            if step % max(1, log_interval * 2) == 0:
                val = evaluate(model, val_loader, device, amp_dtype)
                print(f"          > val_loss {val:.4f}")
                if val < best_val:
                    best_val = val
                    save_checkpoint(model, optimizer, ckpt_best)
                    print(f"          > saved BEST to {ckpt_best}")
                save_checkpoint(model, optimizer, ckpt_last)

            if step >= steps:
                break

    # final save
    save_checkpoint(model, optimizer, ckpt_last)


def save_checkpoint(model, optimizer, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': model.config.__dict__}, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    return ckpt

# ---------------------- Main ----------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--data', type=str, default='data/input_long.txt')
    pa.add_argument('--device', type=str, default='auto')
    pa.add_argument('--batch_size', type=int, default=64)
    pa.add_argument('--block', type=int, default=256)
    pa.add_argument('--n_embd', type=int, default=384)
    pa.add_argument('--n_head', type=int, default=6)
    pa.add_argument('--n_layer', type=int, default=6)
    pa.add_argument('--dropout', type=float, default=0.1)
    pa.add_argument('--lr', type=float, default=3e-4)
    pa.add_argument('--steps', type=int, default=3000)
    pa.add_argument('--grad_accum', type=int, default=1, help='virtual batch via gradient accumulation')
    pa.add_argument('--warmup', type=int, default=200)
    pa.add_argument('--log_interval', type=int, default=100)
    pa.add_argument('--seed', type=int, default=42)
    pa.add_argument('--amp', type=str, default='auto', choices=['auto','off','fp16','bf16'], help='mixed precision dtype')
    pa.add_argument('--checkpoint', type=str, default='tinyllm_last.pt')
    pa.add_argument('--best_checkpoint', type=str, default='tinyllm_best.pt')
    pa.add_argument('--resume', action='store_true')
    # sampling
    pa.add_argument('--sample', action='store_true')
    pa.add_argument('--prompt', type=str, default='인공지능은 ')
    pa.add_argument('--max_new_tokens', type=int, default=200)
    pa.add_argument('--temperature', type=float, default=1.0)
    pa.add_argument('--top_k', type=int, default=50)
    pa.add_argument('--top_p', type=float, default=0.0, help='0 disables nucleus sampling')

    args = pa.parse_args()

    set_seed(args.seed)
    device = auto_device(args.device)

    # AMP dtype selection
    if args.amp == 'off':
        amp_dtype = None
    elif args.amp == 'auto':
        if device.type == 'cuda':
            amp_dtype = torch.float16
        elif device.type == 'mps':
            # MPS autocast supports float16 in recent PyTorch versions
            amp_dtype = torch.float16
        else:
            amp_dtype = None
    else:
        amp_dtype = torch.float16 if args.amp == 'fp16' else torch.bfloat16

    if args.sample and os.path.exists(args.checkpoint):
        # load and sample
        ckpt = load_checkpoint(args.checkpoint, device)
        config = GPTConfig(**ckpt['config'])
        model = TinyGPT(config).to(device)
        model.load_state_dict(ckpt['model'])
        tokenizer = ByteTokenizer()
        model.eval()
        idx = tokenizer.encode(args.prompt).to(device).unsqueeze(0)
        out = generate(model, idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=(args.top_p if args.top_p>0 else None))
        print(tokenizer.decode(out[0].cpu()))
        return

    # Train
    text = load_text(args.data)
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode(text)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block,
        dropout=args.dropout,
    )

    # split train/val
    n = len(ids)
    split = int(n * 0.9)
    train_ids = ids[:split]
    val_ids = ids[split:]

    train_ds = TextDataset(train_ids, block_size=config.block_size)
    val_ds = TextDataset(val_ids, block_size=config.block_size)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = TinyGPT(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    for pg in opt.param_groups:
        pg['base_lr'] = args.lr

    # resume optional
    if args.resume and os.path.exists(args.checkpoint):
        ckpt = load_checkpoint(args.checkpoint, device)
        model.load_state_dict(ckpt['model'])
        try:
            opt.load_state_dict(ckpt['optimizer'])
            print(f"Resumed optimizer from {args.checkpoint}")
        except Exception as e:
            print(f"Warning: could not load optimizer state ({e})")

    total_steps = args.steps
    train_loop(model, opt, train_loader, val_loader, device, steps=total_steps, grad_accum=args.grad_accum, warmup=args.warmup, amp_dtype=amp_dtype, log_interval=args.log_interval, ckpt_best=args.best_checkpoint, ckpt_last=args.checkpoint, scheduler_total=total_steps)

    # quick sample at the end
    idx = tokenizer.encode(args.prompt).to(device).unsqueeze(0)
    out = generate(model, idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=(args.top_p if args.top_p>0 else None))
    print('\nSample:\n' + tokenizer.decode(out[0].cpu()))

if __name__ == '__main__':
    main()
