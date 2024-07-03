# https://github.com/numb3r33/fgcnn
from boruta import BorutaPy

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_xxx_features(train_x, train_y, test_x, ) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

    feat_selector = BorutaPy(rf, n_estimators="auto", verbose=2)
    feat_selector.fit(train_x, train_y)
    train_x = feat_selector.transform(train_x)
    test_x = feat_selector.transform(test_x)
    return train_x, test_x
