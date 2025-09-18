import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

if __name__ == '__main__':
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(df)

    df_pca = pd.DataFrame(data=x_pca, columns=["X1", "X2"])

    plt.scatter(df_pca["X1"], df_pca["X2"])
    plt.show()

