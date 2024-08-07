# https://github.com/alteryx/featuretools
import pandas as pd
import featuretools as ft


def get_featuretools_features(train_x, train_y, test_x, test_y, name) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:

    for column in train_x.select_dtypes(include=['object', 'category']).columns:
        train_x[column], uniques = pd.factorize(train_x[column])
    for column in test_x.select_dtypes(include=['object', 'category']).columns:
        test_x[column], uniques = pd.factorize(test_x[column])
    train_y = pd.DataFrame(train_y)
    for column in train_y.select_dtypes(include=['object', 'category']).columns:
        train_y[column], uniques = pd.factorize(train_y[column])
    test_y = pd.DataFrame(test_y)
    for column in test_y.select_dtypes(include=['object', 'category']).columns:
        test_y[column], uniques = pd.factorize(test_y[column])

    train_x.insert(loc=0, column="id", value=train_x.reset_index().index)
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
        dataframe_name="train_y",
        dataframe=train_y,
    )
    # for i in range(len(train_x.columns) - 1):
    train_es = train_es.add_relationship(name + "_train_x", train_x.columns[0], name + "_train_x", train_x.columns[1])

    train_feature_matrix, train_features = ft.dfs(
        entityset=train_es,
        target_dataframe_name=name + "_train_x",
        agg_primitives=["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"],
        max_depth=1
    )


    test_es = ft.EntitySet(id=name)
    test_es = test_es.add_dataframe(
        dataframe_name=name + "_test_x",
        dataframe=test_x,
    )
    test_es = test_es.add_dataframe(
        dataframe_name="test_y",
        dataframe=test_y,
    )
    # for i in range(len(test_x.columns) - 1):
    test_es = test_es.add_relationship(name + "_test_x",  test_x.columns[0], name + "_test_x", test_x.columns[1])

    test_feature_matrix, test_features = ft.dfs(
        entityset=test_es,
        target_dataframe_name=name + "_test_x",
        agg_primitives=["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"],
        max_depth=1
    )

    train_x = train_feature_matrix
    test_x = test_feature_matrix

    print(train_x.columns)
    print(test_x.columns)

    return train_x, test_x
