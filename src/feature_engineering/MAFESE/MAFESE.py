# https://github.com/thieu1995/mafese
import numpy as np
import pandas as pd
from mafese import Data
from mafese import UnsupervisedSelector

def get_xxx_features(train_x, train_y, test_x, test_y, task_hint, num_features) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    column_names = train_x.columns
    X = np.array(pd.concat([train_x, test_x], axis=0))
    y = np.array(pd.concat([train_y, test_y], axis=0))
    data = Data(X, y)

    data.split_train_test(test_size=0.1, inplace=True)

    # Transform and Encode Data
    train_x, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
    data.X_test = scaler_X.transform(data.X_test)
    data.y_train, scaler_y = data.encode_label(data.y_train)  # This is for classification problem only
    data.y_test = scaler_y.transform(data.y_test)

    # Feature Selector
    feat_selector = UnsupervisedSelector(problem=task_hint, method='DR', n_features=num_features)
    feat_selector.fit(train_x, train_y)

    index = ['Row' + str(i) for i in range(1, len(train_x) + 1)]

    train_x = feat_selector.transform(train_x)
    # test_x = feat_selector.transform(test_x)
    train_x = pd.DataFrame(train_x)
    test_x = pd.DataFrame(test_x)
    return train_x, test_x
