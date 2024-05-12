import pandas as pd
from openfe import OpenFE, transform


def get_openFE_features(train_x, train_y, test_x, n_jobs) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    openFE = OpenFE()
    features = openFE.fit(train_x, label=train_y, n_jobs=n_jobs, verbose=1, )  # generate new features
    train_x, test_x = transform(train_x, test_x, features, n_jobs=n_jobs)
    return train_x, test_x
