# https://github.com/alteryx/featuretools
import numpy as np
import pandas as pd
import featuretools as ft


def get_xxx_features(train_x, train_y, test_x, test_y, name) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    # train_y = pd.DataFrame(train_y, columns=["target"])
    # train_df = pd.concat([train_x, train_y], axis=1)
    train_x.insert(loc=0, column="id", value=train_x.reset_index().index)
    train_y = pd.DataFrame(train_y)
    train_y.insert(loc=0, column="id", value=train_y.reset_index().index)
    es = ft.EntitySet(id=name)
    es = es.add_dataframe(
        dataframe_name=name,
        dataframe=train_x,
    )
    es = es.add_dataframe(
        dataframe_name="target",
        dataframe=train_y,
    )
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="target"
    )
    print("feature_matrix")
    print(feature_matrix)
    print("feature_defs")
    print(feature_defs)

    return train_x, test_x
