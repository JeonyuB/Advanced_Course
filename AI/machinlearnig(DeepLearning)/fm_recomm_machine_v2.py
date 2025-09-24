import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from recommend_module_v2 import extract_high_rating_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from tqdm import tqdm


# --- 1. PyTorch Dataset 클래스 ---
class RatingDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = users
        self.items = items
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


# --- 2. PyTorch Matrix Factorization 모델 ---
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.user_embedding.weight.data.uniform_(0, 0.05)
        self.item_embedding.weight.data.uniform_(0, 0.05)

    def forward(self, user_indices, item_indices):
        user_vector = self.user_embedding(user_indices)
        item_vector = self.item_embedding(item_indices)
        rating = (user_vector * item_vector).sum(1)
        return rating


# ------------------------
# 메인 함수
# ------------------------
def pytorch_mf_model(users_df, factors=32, minimum_num_ratings=4, epochs=20, lr=0.01):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("gpu available")
    else:
        device = torch.device("cpu")
        print("cpu available")

    # 사용자/아이템 필터링
    user_counts = users_df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= minimum_num_ratings].index
    movie_counts = users_df["movie_id"].value_counts()
    valid_movies = movie_counts[movie_counts >= minimum_num_ratings].index
    filtered_data = users_df[
        (users_df["user_id"].isin(valid_users)) & (users_df["movie_id"].isin(valid_movies))
        ].copy()

    # 인덱스 매핑
    user_ids = filtered_data['user_id'].unique()
    movie_ids = filtered_data['movie_id'].unique()
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    movie_to_idx = {m: i for i, m in enumerate(movie_ids)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    idx_to_movie = {i: m for m, i in movie_to_idx.items()}

    filtered_data['user_idx'] = filtered_data['user_id'].map(user_to_idx)
    filtered_data['movie_idx'] = filtered_data['movie_id'].map(movie_to_idx)

    # 훈련/테스트 분리
    train_data, test_data = train_test_split(filtered_data, test_size=0.2, random_state=42)

    # Dataset, DataLoader
    train_dataset = RatingDataset(
        torch.LongTensor(train_data['user_idx'].values),
        torch.LongTensor(train_data['movie_idx'].values),
        torch.FloatTensor(train_data['rating'].values)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 모델
    num_users, num_items = len(user_to_idx), len(movie_to_idx)
    model = MatrixFactorization(num_users, num_items, embedding_dim=factors).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 학습
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            optimizer.zero_grad()
            pred = model(u, i)
            loss = loss_fn(pred, r)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # ------------------------
    # 테스트 평가
    # ------------------------
    model.eval()
    with torch.no_grad():
        test_users = torch.LongTensor(test_data['user_idx'].values).to(device)
        test_items = torch.LongTensor(test_data['movie_idx'].values).to(device)
        preds = model(test_users, test_items).cpu().numpy()

    actuals = test_data['rating'].values
    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, preds)
    print(f"\n📊 Test Performance → RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # ------------------------
    # 결과 DataFrame 생성 (실제 vs 예측)
    # ------------------------
    result_df = test_data[['user_id', 'movie_id', 'rating']].copy()
    result_df['predicted_rating'] = preds

    return result_df


if __name__ == '__main__':
    users = extract_high_rating_data()
    user_df = users[['user_id', 'movie_id', 'rating']].copy()
    predictions = pytorch_mf_model(users, factors=8, minimum_num_ratings=10)

    print("\n--- 실제 vs 예측 결과 ---")
    print(predictions.head(10))