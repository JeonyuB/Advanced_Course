import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#3차원 경우

if __name__ == "main":
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    cov_matrix = df.cov().values
    eig_val, eig_vec = np.linalg.eig(cov_matrix)

    # print(eig_vec)
    trans_matrix = eig_vec[:,:3]
    # print(trans_matrix)

    new_matrix =  df.values @ trans_matrix
    # print(new_matrix)
    new_df = pd.DataFrame(new_matrix, columns=["X", "Y", "Z"])

    # 3D 산점도
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 색상: Iris 클래스 (0,1,2)
    ax.scatter(new_df["X"], new_df["Y"], new_df["Z"])

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.title("Iris Dataset PCA (3D Projection)")
    plt.show()

    # plt.scatter(new_df["X"], new_df["Y"], c='blue', marker='o')
    # plt.show()