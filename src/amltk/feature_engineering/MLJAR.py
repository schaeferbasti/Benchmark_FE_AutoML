import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
from supervised.automl import AutoML
from supervised.preprocessing.kmeans_transformer import KMeansTransformer
from supervised.preprocessing.preprocessing import Preprocessing
from supervised.preprocessing.goldenfeatures_transformer import GoldenFeaturesTransformer

from src.amltk.datasets.Datasets import preprocess_data, preprocess_target


def get_mljar_features(train_x, train_y, test_x, test_y) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:

    print(train_x.shape, test_x.shape)
    print("Generate Golden Features")
    # Golden Features
    preprocessor = GoldenFeaturesTransformer()
    preprocessor._new_features = [
        {
            "feature1": "V1",
            "feature2": "V3",
            "operation": "ratio",
            "score": 0.5143360052
        },
        {
            "feature1": "V1",
            "feature2": "V2",
            "operation": "ratio",
            "score": 0.5292468849
        },
        {
            "feature1": "V3",
            "feature2": "V2",
            "operation": "ratio",
            "score": 0.5488743845
        },
        {
            "feature1": "V2",
            "feature2": "V3",
            "operation": "ratio",
            "score": 0.5488743845
        },
        {
            "feature1": "V4",
            "feature2": "V1",
            "operation": "sum",
            "score": 0.6422473144
        },
        {
            "feature1": "V3",
            "feature2": "V1",
            "operation": "sum",
            "score": 0.6929436099
        },
        {
            "feature1": "V2",
            "feature2": "V3",
            "operation": "diff",
            "score": 0.7018018361
        },
        {
            "feature1": "V3",
            "feature2": "V2",
            "operation": "multiply",
            "score": 0.7080949868
        },
        {
            "feature1": "V3",
            "feature2": "V2",
            "operation": "sum",
            "score": 0.7080949868
        },
        {
            "feature1": "V4",
            "feature2": "V1",
            "operation": "ratio",
            "score": 0.7932164679
        }
    ]
    preprocessor._new_columns = [
        "V1_ratio_V3",
        "V1_ratio_V2",
        "V3_ratio_V2",
        "V2_ratio_V3",
        "V4_sum_V1",
        "V3_sum_V1",
        "V2_multiply_V3",
        "V3_multiply_V2",
        "V3_sum_V2",
        "V4_ratio_V1"
    ]
    train_x, test_x = preprocess_with_preprocessor(preprocessor, train_x, train_y, test_x, test_y)

    train_x = pd.DataFrame(train_x)
    test_x = pd.DataFrame(test_x)

    print(train_x.shape, test_x.shape)

    """
    # MLJAR with AutoML with generation of Golden Features
    mljar = AutoML(golden_features=True, kmeans_features=True, features_selection=True)
    mljar.fit(train_x, train_y)
    """

    return train_x, test_x


def preprocess_with_preprocessor(preprocessor, train_x, train_y, test_x, test_y):
    # train_x = preprocessor.fit_transform(train_x, train_y)
    preprocessor.fit(train_x, train_y)
    train_x = preprocessor.transform(train_x)
    test_x = preprocessor.transform(test_x)
    train_x = pd.DataFrame(train_x)
    test_x = pd.DataFrame(test_x)
    return train_x, test_x
