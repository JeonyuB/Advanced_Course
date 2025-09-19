import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target  # 라벨 추가
    df = pd.DataFrame(X, columns=iris.feature_names)

    pca = PCA(n_components=2)
    new_X = pca.fit_transform(X)
    new_df = pd.DataFrame(new_X, columns=['X', 'Y'])
    new_df['label'] = y  # 라벨 정보도 데이터프레임에 추가

    # 라벨별 색상과 이름
    target_names = iris.target_names
    colors = ['red', 'green', 'blue']

    plt.figure(figsize=(8, 6))
    for i, color, label in zip([0, 1, 2], colors, target_names):
        plt.scatter(new_df[new_df['label'] == i]["X"],
                    new_df[new_df['label'] == i]["Y"],
                    color=color, label=label, s=40)

    plt.title("Iris Dataset (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()

#라벨링 전
# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#
# if __name__ == "__main__":
#     iris = load_iris()
#     X=iris.data
#     df=pd.DataFrame(X, columns=iris.feature_names)
#
#     pca=PCA(n_components=2)
#     new_X=pca.fit_transform(X)
#     new_df=pd.DataFrame(new_X, columns=['X', 'Y'])
#     plt.figure(figsize=(8,6))
#     plt.scatter(new_df["X"], new_df["Y"], s=40)
#     plt.title("Iris Dataset")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#
#
#     plt.show()