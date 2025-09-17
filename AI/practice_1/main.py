import pandas as pd
from pandas import DataFrame
from sklearn.datasets import load_iris

# pandas 테이블:DataFrame










#__name__ 뜻: 코드를 짜고 있는 스크립트 파일 이름
#"__main__" 뜻: 실행하는 스크립트 파일 이름
if __name__=="__main__":
    iris = load_iris() #데이터가 날라옴.
    print(iris.keys())
    # print(iris.target_names)
    # print(iris.target)
    # exit()
    df = pd.DataFrame(iris['data'])
    df.columns = iris.feature_names
    # print(df.head())
    print(df.corr().values)