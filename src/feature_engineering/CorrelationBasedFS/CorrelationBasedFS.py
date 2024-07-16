# https://github.com/Doctorado-ML/mufs

import pandas as pd
from scipy.io import arff

from src.feature_engineering.CorrelationBasedFS.MUFS import MUFS


def get_correlationbased_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    float_array_of_features = train_x.to_numpy().astype("float64")
    float_array_of_targets = train_y.to_numpy().astype("float64")

    mufsc = MUFS(discrete=False)
    cfs_f = mufsc.cfs(float_array_of_features, float_array_of_targets).get_results()

    print(cfs_f)
    print(mufsc.get_scores())

    train_x = train_x.iloc[:, cfs_f]
    test_x = test_x.iloc[:, cfs_f]

    return train_x, test_x
