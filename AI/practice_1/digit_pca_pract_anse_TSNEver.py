import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

if __name__ == '__main__':
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    # pca = PCA(n_components=2)
    # x_pca = pca.fit_transform(df)
    #
    # df_pca = pd.DataFrame(data=x_pca, columns=["X1", "X2"])


    tsne = TSNE(n_components=2, random_state=17, perplexity=30, init='pca', learning_rate="auto")
    x_tsne = tsne.fit_transform(df.values)
    new_df = pd.DataFrame(x_tsne, columns=["X", "Y"])
    plt.scatter(new_df["X"], new_df["Y"])
    plt.show()

