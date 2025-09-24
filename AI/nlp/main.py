import pandas as pd
import numpy as np
from numpy.ma.extras import column_stack

if __name__=="__main__":
    df = pd.read_excel("data/NAVER-Webtoon_OSMU.xlsx")
    columns = ["synopsis", "genre"]
    df = df[columns]