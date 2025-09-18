import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#4차원 아이리스 예제

if __name__ == "__main__":
    iris = load_iris()
    df=pd.DataFrame(iris.data, columns=iris.feature_names)
    cov_matrix=df.cov().values
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    print(eigen_values)
    print(eigen_vectors)

    trans_matrix=eigen_vectors[:,:2]
    print(trans_matrix)

    new_matrix = df.values @ trans_matrix

    print(new_matrix)


    new_df=pd.DataFrame(new_matrix, columns=["X","Y"])
    plt.scatter(new_df["X"], new_df["Y"], c='blue', marker='o')
    plt.show()


