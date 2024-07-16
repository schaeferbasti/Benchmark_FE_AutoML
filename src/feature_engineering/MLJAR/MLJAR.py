# https://github.com/mljar/mljar-supervised

import pandas as pd
import numpy as np
import random

from supervised.preprocessing.goldenfeatures_transformer import GoldenFeaturesTransformer
from src.datasets.Datasets import preprocess_data, preprocess_target


def create_feature_list(cols, num_features):
    feature_list = []
    name_list = []
    operator_list = ["multiply", "ratio", "sum"]
    for i in range(num_features):
        feature_1 = random.choice(cols)
        feature_2 = random.choice(cols)
        operation = random.choice(operator_list)
        score = np.random.rand()
        entry = {
            "feature1": feature_1,
            "feature2": feature_2,
            "operation": operation,
            "score": score
        }
        name_list.append(str(feature_1) + "_" + str(operation) + "_" + str(feature_2))
        feature_list.append(entry)
    return feature_list, name_list


def get_mljar_features(train_x, train_y, test_x, num_features) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    print(train_x.shape, test_x.shape)
    print("Generate Golden Features")

    preprocessor = GoldenFeaturesTransformer()

    # train_x, test_x = preprocess_data(train_x, test_x)
    # train_y = preprocess_target(train_y)

    cols = train_x.columns
    feature_list, name_list = create_feature_list(cols, num_features)
    preprocessor._new_features = feature_list
    preprocessor._new_columns = name_list

    preprocessor.fit(train_x, train_y)
    train_x = preprocessor.transform(train_x)
    test_x = preprocessor.transform(test_x)
    train_x = pd.DataFrame(train_x)
    test_x = pd.DataFrame(test_x)

    print(train_x.shape, test_x.shape)

    """
    # MLJAR with AutoML with generation of Golden Features
    mljar = AutoML(golden_features=True, kmeans_features=True, features_selection=True)
    mljar.fit(train_x, train_y)
    """

    return train_x, test_x
