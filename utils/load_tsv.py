import numpy as np
import pandas as pd

def load_from_tsvfile(file_path):
    df = pd.read_csv(file_path, sep="\t", index_col=False, header=None)
    y = df.iloc[:, 0].to_numpy()
    X = np.expand_dims(df.iloc[:, 1:].to_numpy(), axis=1)
    df = df.to_numpy()
    return X, y