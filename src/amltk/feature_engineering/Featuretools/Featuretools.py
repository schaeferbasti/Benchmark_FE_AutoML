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
    print(train_x)
    train_y = pd.DataFrame(train_y)
    train_y.insert(loc=0, column="id", value=train_y.reset_index().index)
    test_x.insert(loc=0, column="id", value=test_x.reset_index().index)
    test_y = pd.DataFrame(test_y)
    test_y.insert(loc=0, column="id", value=test_y.reset_index().index)

    train_es = ft.EntitySet(id=name)
    train_es = train_es.add_dataframe(
        dataframe_name=name + "_train_x",
        dataframe=train_x,
    )
    train_es = train_es.add_dataframe(
        dataframe_name=name + "_test_x",
        dataframe=test_x,
    )
    train_es = train_es.add_dataframe(
        dataframe_name="train_y",
        dataframe=train_y,
    )

    train_feature_matrix, train_features = ft.dfs(
        entityset=train_es,
        target_dataframe_name=name + "_train_x",
    )


    test_es = ft.EntitySet(id=name)
    test_es = test_es.add_dataframe(
        dataframe_name=name + "_test_x",
        dataframe=test_x,
    )
    # Shouldn't input test_y to avoid leaking
    test_es = test_es.add_dataframe(
        dataframe_name="test_y",
        dataframe=test_y,
    )

    test_feature_matrix, test_features = ft.dfs(
        entityset=test_es,
        target_dataframe_name="test_y",
        agg_primitives=["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"],  # default transformations
        max_depth=10
    )

    train_x = train_feature_matrix
    test_x = test_feature_matrix

    print(train_x)
    print(test_x)

    return train_x, test_x
