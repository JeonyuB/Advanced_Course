import pandas as pd

if __name__=="__main__":
    # columns = ['cust_id', 'movie_id', 'rating', 'timestamp']
    # df=pd.read_csv('data/ratings.dat',
    #                sep='::', names=columns, engine='python')
    # # print(df.head())
    # df.to_pickle('data/ratings.pkl')

    # exit()
    df=pd.read_pickle('data/ratings.pkl')

    # print(df.head())

    users = df['cust_id'].value_counts().reset_index().loc[:19]
    movies = df['movie_id'].value_counts().reset_index().loc[:]
    # print(users.shape)


    df_custId_20= df[df['cust_id'].isin(users['cust_id'])] #이게 팬시 인덱스라네.
    print(df_custId_20.shape)

    pivoted_table=df_custId_20.pivot_table(index='cust_id', columns='movie_id',
                                             values='rating',
                                            # fill_value=0
                                           )

    means=pivoted_table.mean(axis=0)
    pivoted_table.fillna(means, inplace=True)
    print(pivoted_table)

