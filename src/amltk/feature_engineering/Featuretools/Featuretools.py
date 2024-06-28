# https://github.com/alteryx/featuretools

import pandas as pd
import featuretools as ft

def get_xxx_features(train_x, train_y, test_x, test_y, name) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    es = ft.demo.load_mock_customer(return_entityset=True)
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name=name)
    print(feature_matrix)
    print(feature_defs)
    return train_x, test_x
