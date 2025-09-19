import pandas as pd
from pyexpat import features
from sklearn.datasets import load_iris
from scipy.stats import f_oneway

if __name__ == "__main__":
    data=load_iris()
    X = data.data
    y = data.target
    df=pd.DataFrame(X,columns=data.feature_names)
    features=data.feature_names
    df['target']=y
    kinds=df['target'].unique()
    sepal_length_groups=[
        df[df['target']==kind][features[0]]
        for kind in kinds
    ]
    _, p_value = f_oneway(*sepal_length_groups)
    print(p_value)
