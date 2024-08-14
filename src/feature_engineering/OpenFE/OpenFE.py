# https://github.com/IIIS-Li-Group/OpenFE

import pandas as pd
from src.feature_engineering.OpenFE.method import OpenFE, transform


def get_openFE_features(train_x, train_y, test_x, n_jobs, name) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    openFE = OpenFE()
    features = openFE.fit(name=name, data=train_x, label=train_y, n_jobs=n_jobs)  # generate new features
    train_x, test_x = transform(train_x, test_x, features, n_jobs=n_jobs)
    return train_x, test_x
