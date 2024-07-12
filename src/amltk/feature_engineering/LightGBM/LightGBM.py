# https://github.com/Microsoft/LightGBM

import pandas as pd
import lightgbm as lgb


def get_xxx_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    categorical_features = []
    for column in train_x.columns:
        if train_x[column].dtype == object:
            categorical_features.append(column)

    data = lgb.Dataset(train_x, label=train_y, feature_name=train_x.columns, categorical_feature=train_x.columns)
    # No feature engineering
    return train_x, test_x
