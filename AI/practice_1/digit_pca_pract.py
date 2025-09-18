import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

if __name__ == '__main__':
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
#    print(df.values)#데이터 개많음
#    print(df.cov().values)#이거도 개많음

    cov_matrix=df.cov().values
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    # print(eigen_values)
    # print("===========================")
    # print(eigen_vectors)



    trans_matrix=eigen_vectors[:,:2]
    print("============2차원===============")
    print(trans_matrix)

    new_matrix = df.values @ trans_matrix

    print(new_matrix)

    new_df=pd.DataFrame(new_matrix, columns=["X","Y"])
    plt.scatter(new_df["X"], new_df["Y"], c='red', marker='o')
    plt.show()
