import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import k_means, KMeans
from sklearn.manifold import TSNE

if __name__ == '__main__':
    # 1) 데이터 준비
    digits = load_digits()
    X = digits.data
    y = digits.target

    # 2) 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3) KMeans 라벨링 (k=10)
    kmeans = KMeans(n_clusters=10,
                    init="k-means++",
                    n_init=10,
                    random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # 4) (선택) t-SNE 전에 PCA로 50차원 축소 (속도/안정성 개선용)
    # pca = PCA(n_components=50, random_state=42)
    # X_for_tsne = pca.fit_transform(X_scaled)
    # t-SNE 입력
    X_for_tsne = X_scaled

    # 5) t-SNE 2D 임베딩

    tsne = TSNE(n_components=2,
                perplexity=30,
                learning_rate="auto",
                init="pca",
                random_state=77
                )
    X_tsne = tsne.fit_transform(X_scaled)
    # 6) 시각화: KMeans 클러스터 기준
    plt.figure(figsize=(8, 6))
    for c in np.unique(clusters):
        mask = clusters == c
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=20, label=f"Cluster {c}", alpha=0.7)
    plt.title("Digits - KMeans Clusters (t-SNE 2D)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.show()

    # 7) 시각화: 실제 정답 라벨 기준
    plt.figure(figsize=(8, 6))
    for c in np.unique(y):
        mask = y == c
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=20, label=f"Digit {c}", alpha=0.7)
    plt.title("Digits - True Labels (t-SNE 2D)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.show()