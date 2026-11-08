import numpy as np
import pandas as pd

NAN_FILL = -1e9 

def to_matrix(df, feat_cols):
    X = df[feat_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=NAN_FILL, posinf=1e12, neginf=-1e12)
    return X



def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def confusion(y_true, y_pred, classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.asarray(classes)
    return pd.crosstab(
        pd.Series(classes[y_true], name="true"),
        pd.Series(classes[y_pred], name="pred"),
    )