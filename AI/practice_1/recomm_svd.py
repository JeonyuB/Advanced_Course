import pandas as pd
from sklearn.decomposition import TruncatedSVD

from pivot_table_pract import ratings_table

if __name__=="__main__":
    pivot_df, nullcols_71420 = ratings_table()
    # df=pd.read_pickle("data/ratings.pkl")

    # print(pivot_df)
    _df=pivot_df.stack().reset_index()
    _df.columns=['cust_id','movie_id','rating']
    print(_df[ _df["cust_id"]==71420]['rating'].to_list()[:20]) #원본데이터(71420가 직접넣은 평점)


    columns = pivot_df.columns
    index = pivot_df.index

    # exit()
    svd=TruncatedSVD(n_components=10, random_state=42)#100차원임

    user_svd=svd.fit_transform(pivot_df)
    rating_predict=user_svd@svd.components_

    df=pd.DataFrame(rating_predict,columns=columns, index=index)

    unpivot_df=df.stack().reset_index()
    unpivot_df.columns=['cust_id','movie_id','rating']
    pred_71420=unpivot_df[ (unpivot_df['cust_id']==71420) & (unpivot_df['movie_id'].isin(nullcols_71420))
                                                            &(unpivot_df['rating']>=4.0)]
    print(unpivot_df[unpivot_df["cust_id"]==71420]['rating'].to_list()[:20])#추측 데이터(71420가 넣을 것같은 평점)

    # print(unpivot_df)
    print(pred_71420)
