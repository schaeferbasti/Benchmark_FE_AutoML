from src.amltk.datasets.Datasets import preprocess_data, preprocess_target

import pandas as pd

from h2o.assembly import *
from h2o.transforms.preprocessing import *

def get_h2o_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    h2o.init()

    X_y_train_h = h2o.H2OFrame(pd.concat([train_x, train_y], axis='columns'))
    X_test_h = h2o.H2OFrame(test_x)

    train_cols = X_y_train_h.columns
    """
    train_x = preprocess_data(train_x)
    train_y = preprocess_target(train_y)
    test_x = preprocess_data(test_x)
    """
    assembly = H2OAssembly(steps=[
        ("col_select", H2OColSelect(train_cols))
    ])
    result = assembly.fit(X_y_train_h)  # fit the assembly and perform the munging operations


    X_train_h = result.drop("Class", axis=1)
    with open('results/h2o.txt', 'w') as f:
        f.write(str(X_train_h))
    X_train_h = X_train_h.as_data_frame(use_pandas=True)
    X_test_h = X_test_h.as_data_frame(use_pandas=True)
    return X_train_h, X_test_h
