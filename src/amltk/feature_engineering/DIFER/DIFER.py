# https://github.com/PasaLab/DIFER

import pandas as pd


def get_difer_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    return train_x, test_x
