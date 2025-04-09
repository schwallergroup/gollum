from ast import literal_eval
import pandas as pd
import numpy as np

def precalculated(features):
    if isinstance(features.iloc[0], str):
        features = features.apply(literal_eval)
    return features.apply(pd.Series).values

def all_continuous(df):
    return df.to_numpy(dtype=np.float64)

