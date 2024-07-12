# https://github.com/fuyuanlyu/AutoFS-in-CTR/tree/main/LPFS
import numpy as np
import pandas as pd
from pymfe.mfe import MFE
import scipy
from scipy.stats import shapiro
from sklearn.ensemble import IsolationForest


def get_xxx_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    mfe = MFE()
    mfe.fit(np.array(train_x), np.array(train_y))
    train_metafeatures = mfe.extract()
    train_trm = fit_scaler_transformations(train_x, train_metafeatures)
    print(train_trm)
    mfe.fit(np.array(test_x))
    test_metafeatures = mfe.extract()
    test_trm = fit_scaler_transformations(test_x, test_metafeatures)
    print(test_trm)
    return train_x, test_x


def fit_scaler_transformations(X, metafeatures):
    if _test_outliers_for_robust_scaler(X):
        scaler_index = 0

    elif _test_normal_distribution(X):
        scaler_index = 1
    else:
        scaler_index = 2

    TRM_scalers = {
        'encoding': metafeatures,
        'top_t_index': scaler_index
    }

    return TRM_scalers


def _test_outliers_for_robust_scaler(X, threshold = 0.11):
    iso = IsolationForest(random_state = 42, n_jobs = 4, contamination = 0.1)
    outliers = iso.fit_predict(X)
    count_outliers = np.count_nonzero(outliers == -1)
    if (count_outliers / len(X)) >= threshold:
        return True
    else:
        return False


def _test_normal_distribution(X, threshold=0.5):
    count = 0
    cols = X.columns
    for i in range(len(cols)):
        np.array(X)
        if shapiro(X[:, i])[1] > 0.05:
            count += 1

    if count / cols > threshold:
        return True
    else:
        return False
