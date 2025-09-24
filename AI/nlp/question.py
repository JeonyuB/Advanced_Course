import torch
from TinyChatGPT import TinyGPT, GPTConfig




if __name__ == "__main__":
    # 체크포인트 불러오기
    ckpt = torch.load("tinyllm.pt", map_location="cpu")

    # 모델/토크나이저 복원
    config = GPTConfig(**ckpt['config'])
    model = TinyGPT(config)
    model.load_state_dict(ckpt['model'])
    model.eval()

    tokenizer = ckpt['tokenizer']

    # 입력 문장
    context = "창조성"
    idx = torch.tensor([tokenizer['stoi'].get(c, 0) for c in context], dtype=torch.long).unsqueeze(0)

    # 텍스트 생성
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=100, temperature=1.0, top_k=50)

    # 디코딩
    decoded = ''.join([tokenizer['itos'][int(i)] for i in out[0]])
    print(decoded)
