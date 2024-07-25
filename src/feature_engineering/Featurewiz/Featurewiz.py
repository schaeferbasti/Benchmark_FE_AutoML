# https://github.com/AutoViML/featurewiz

import pandas as pd
from src.feature_engineering.Featurewiz.method.featurewiz import FeatureWiz


def get_featurewiz_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    fwiz = FeatureWiz(feature_engg='', nrows=None, transform_target=True, scalers="std",
                      category_encoders="auto", add_missing=False, verbose=0, imbalanced=False,
                      ae_options={})
    train_x, train_y = fwiz.fit_transform(train_x, train_y)
    test_x = fwiz.transform(test_x)
    return train_x, test_x
