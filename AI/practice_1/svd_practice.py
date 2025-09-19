import numpy as np
import pandas as pd
from scipy.linalg import svd


if __name__=="__main__":
    # columns = ['cust_id', 'movie_id', 'rating', 'timestamp']
    # df=pd.read_csv('data/ratings.dat',
    #                sep='::', names=columns, engine='python')
    # # print(df.head())
    # df.to_pickle('data/ratings.pkl')
    #
    # # exit()
    # df=pd.read_pickle('data/ratings.pkl')

    # print(df.head())

    # users = df['cust_id'].value_counts().reset_index().loc[:19]
    # movies = df['movie_id'].value_counts().reset_index().loc[:]
    # print(users.shape)


    # df_custId_20= df[df['cust_id'].isin(users['cust_id'])] #이게 팬시 인덱스라네.
    # print(df_custId_20.shape)
    #
    # pivoted_table=df_custId_20.pivot_table(index='cust_id', columns='movie_id',
    #                                          values='rating',
    #                                         # fill_value=0
    #                                        )

    # means=pivoted_table.mean(axis=0)
    # pivoted_table.fillna(means, inplace=True)
    # print(pivoted_table)

    # means=pivoted_table.mean(axis=0)
    # pivoted_table.fillna(means, inplace=True)
    # df=pivoted_table.iloc[:10, :5]
    # df.to_pickle('data/example_pk.pkl')

    df = pd.read_pickle("data/example_pk.pkl")

    X=df.values
    # U,S,VT=svd(X)
    # S=S[:-1]
    # #S: 고윳값, U: 열에 대한 고유벡터, VT: 행에 대한 고유벡터

    U, S, VT = svd(X, full_matrices=False)
    k=3
    A_k=U[:,:k]@np.diag(S[:k])@VT[:k,:]


    # D=np.zeros((10,5))
    # np.fill_diagonal(D,S)
    # print(D)
    # D=np.diag(S)
    # A=U@D@VT #세개 다 곱하면 원래 값으로 돌아감.(=A)
    #10 by 10, 10by 5, 5 by 5 화 시켜야함.


    print(X)
    # print(A[0])
    # print(A_k[0])
    # print(D)
    # print(S)
    # print(U.shape)
    # print(VT.shape)

    # U=X.T@X
    # V=X@X.T

    # U_evalues, U_evectors = np.linalg.eigh(U)
    # print(U_evalues)
    # V_evalues, V_evectors = np.linalg.eigh(V)
    # print(V_evalues)
    # # print(U_evectors)



