# https://github.com/thieu1995/mafese

import numpy as np
import pandas as pd
from mafese import Data
from mafese import UnsupervisedSelector
from mafese import get_dataset

from src.datasets.Datasets import preprocess_data


def get_mafese_features(train_x, train_y, test_x, test_y, name, num_features) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    """
    train_x = pd.DataFrame(train_x)
    for column in train_x.select_dtypes(include=['object', 'category']).columns:
        train_x[column], uniques = pd.factorize(train_x[column])
    test_x = pd.DataFrame(test_x)
    for column in test_x.select_dtypes(include=['object', 'category']).columns:
        test_x[column], uniques = pd.factorize(test_x[column])
    train_y = pd.DataFrame(train_y)
    for column in train_y.select_dtypes(include=['object', 'category']).columns:
        train_y[column], uniques = pd.factorize(train_y[column])
    test_y = pd.DataFrame(test_y)
    for column in test_y.select_dtypes(include=['object', 'category']).columns:
        test_y[column], uniques = pd.factorize(test_y[column])
    train_y = pd.Series(train_y.iloc[0])
    test_y = pd.Series(test_y.iloc[0])
    """

    train_x, test_x = preprocess_data(train_x, test_x)

    X = pd.concat([train_x, test_x], axis=0).values
    y = pd.concat([train_y, test_y], axis=0).values
    data = Data(X, y, name)

    data.split_train_test(test_size=0.1, inplace=True)

    # Transform and Encode Data
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
    data.X_test = scaler_X.transform(data.X_test)
    data.y_train, scaler_y = data.encode_label(data.y_train)  # This is for classification problem only
    data.y_test = scaler_y.transform(data.y_test)

    # Feature Selector
    feat_selector = UnsupervisedSelector(problem='classification', method='DR', n_features=None)
    feat_selector.fit(data.X_train, data.y_train)

    X_train_selected = feat_selector.transform(data.X_train)
    X_test_selected = feat_selector.transform(data.X_test)

    train_x = pd.DataFrame(X_train_selected)
    test_x = pd.DataFrame(X_test_selected)
    return train_x, test_x
