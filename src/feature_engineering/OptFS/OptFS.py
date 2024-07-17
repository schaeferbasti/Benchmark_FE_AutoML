# https://github.com/fuyuanlyu/OptFS

import pandas as pd

from src.feature_engineering.OptFS.method.trainer import main


def get_xxx_features(train_x, train_y, test_x, test_y) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    df_train = pd.concat([train_x, train_y], axis=1)
    df_test = pd.concat([test_x, test_y], axis=1)
    df = pd.concat([df_train, df_test], axis=0)

    main(df)

    return train_x, test_x
