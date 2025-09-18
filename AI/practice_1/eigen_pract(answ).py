import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#

if __name__ == '__main__':
    df =pd.read_csv("./data/SOCR_HeightWeight.csv")
    columns = ["Height(Inches)", "Weight(Pounds)"]
    df=df[columns]
    # print(df.head())

    print(df.cov().values) #선형대수
    print(df.shape)
    mean = df.mean()
    print(mean) #아무 명령 없을 경우, axis가 2개(출력격ㄹ과가 2개 나옴)= axis가 0일 경우랑 같음
    tmp = df-mean
    #exit()
    #

    #고유값 및 고유벡터 구하기
    cov_matrix= tmp.cov().values #공분산
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    print(eigen_values)
    print(eigen_vectors)
    exit()
    #

    print(tmp.cov().values)
    # mean = df.mean(axis=1)#axis=1일 경우,
    # print(mean)
    # exit()

    plt.scatter(df["Height(Inches)"], df["Weight(Pounds)"], c='blue', marker='o') #그래프 보여줌
    plt.show()
