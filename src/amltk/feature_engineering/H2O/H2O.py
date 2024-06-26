# https://github.com/h2oai/h2o-3

import pandas as pd

from h2o.assembly import *
from h2o.transforms.preprocessing import *


def get_h2o_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    h2o.init()

    X_train_h2o = h2o.H2OFrame(train_x)
    X_test_h2o = h2o.H2OFrame(test_x)

    train_cols = X_train_h2o.columns

    # Create H2OAssembly
    assembly = H2OAssembly(
        steps=[
            ("col_op_1", H2OColOp(op=H2OFrame.cos, col=train_cols[0], inplace=True)),
            ("col_op_2", H2OColOp(op=H2OFrame.log, col=train_cols[1], inplace=True)),
            ("col_op_3", H2OColOp(op=H2OFrame.sin, col=train_cols[2], inplace=True)),
            ("col_op_4", H2OColOp(op=H2OFrame.tan, col=train_cols[3], inplace=True)),
            # ("group_by", GroupBy(by=train_cols[1], fr=X_train_h2o)),
            # ("col_select", H2OColSelect(train_cols)),
        ]
    )


    X_train_h2o = assembly.fit(X_train_h2o)

    train_x = X_train_h2o.as_data_frame(use_pandas=True)
    test_x = X_test_h2o.as_data_frame(use_pandas=True)
    return train_x, test_x
