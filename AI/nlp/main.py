import ast
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from collections import Counter


def preprocessing_and_to_pickle():
    df = pd.read_excel("data/NAVER-Webtoon_OSMU.xlsx")
    columns = ["synopsis", "genre"]
    df = df[columns]
    df['genre'] = df['genre'].apply(lambda row: ast.literal_eval(row))

    embedder = SentenceTransformer("sentence-transformers/xlm-r-base-en-ko-nli-ststb")
    X_embedding = embedder.encode(df['synopsis'].tolist(),
                                  convert_to_tensor=False,
                                  show_progress_bar=True)
    X_embedding = normalize(X_embedding)
    df['X_embedding'] = list(X_embedding)

    df.to_pickle("data/NAVER-Webtoon_OSMU.pkl")

def pca_and_kmeans():
    # ----- 4. 한글 폰트 (macOS) -----
    # font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
    # fm.FontProperties(fname=font_path)
    # plt.rc('font', family='AppleGothic')
    # plt.rcParams['axes.unicode_minus'] = False

    # ----- 4. 한글 폰트 (Windows) -----
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    df = pd.read_pickle("data/NAVER-Webtoon_OSMU.pkl")

    embedding_matrix = np.vstack(df['X_embedding'].values)
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embedding_matrix)

    # 라벨링
    df['pca_x'] = pca_results[:, 0]
    df['pca_y'] = pca_results[:, 1]

    X = df[['pca_x', 'pca_y']]

    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')  # 좌표축(n_clusters) 4개로 분리,
    kmeans.fit(X)
    df['label'] = kmeans.labels_

    return df

    # x테스트용 결과
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # plt.figure(figsize=[8,8])
    # plt.scatter(pca_results[:,0], pca_results[:,1], alpha=0.5,s=30)
    # plt.title("2차원 시각화", fontsize=20)
    #
    # plt.savefig("data/NAVER-Webtoon_OSMU.png")
    # plt.show()

def lda_visualization():
    df = pca_and_kmeans()
    X = np.vstack(df["X_embedding"].values)
    Y = df['label'].values
    # print(df.keys())

    lda = LDA(n_components=2)
    lda_results = lda.fit_transform(X, Y)
    df['lda_x'] = lda_results[:, 0]
    df['lda_y'] = lda_results[:, 1]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x="lda_x", y="lda_y", hue="label", data=df, palette="viridis", s=30, alpha=0.5)  # s= 점의 크기
    plt.title("LDA 축 시각화", fontsize=20)
    plt.savefig("data/NAVER-Webtoon_OSMU_LDA.png")
    plt.show()

# def

if __name__=="__main__":
    # preprocessing_and_to_pickle() # 피클화
    df=pca_and_kmeans()

    genre_series = df.groupby('label')['genre'].sum().apply(Counter)

    sorted_genre = genre_series.apply(lambda x: dict(x.most_common()))

    df['genre_kinds'] = df['label'].map(sorted_genre)
    print(df[['genre_kinds','label']])










