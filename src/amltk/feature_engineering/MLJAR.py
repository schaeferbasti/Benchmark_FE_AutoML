import pandas as pd
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

    preprocessor = Preprocessing()# preprocessing_params, name, k_fold, repeat)
    train_x, train_y, sample_weight = preprocessor.fit_and_transform(train_x, train_y)
    test_x, test_y, sample_weight = preprocessor.transform(test_x, test_y)

    """
    # Golden Features
    preprocessor = GoldenFeaturesTransformer()
    train_x, test_x = preprocess_with_preprocessor(preprocessor, train_x, train_y, test_x)
    # KMeans Features
    preprocessor = KMeansTransformer()
    train_x, test_x = preprocess_with_preprocessor(preprocessor, train_x, train_y, test_x)
    """

    train_x = pd.DataFrame(train_x)
    test_x = pd.DataFrame(test_x)

    print(train_x.shape, test_x.shape)

    return train_x, test_x


def preprocess_with_preprocessor(preprocessor, train_x, train_y, test_x):
    preprocessor = preprocessor.fit(train_x, train_y)
    preprocessor = preprocessor.fit(train_x)
    train_x = preprocessor.transform(train_x)
    test_x = preprocessor.transform(test_x)
    train_x = pd.DataFrame(train_x)
    test_x = pd.DataFrame(test_x)
    return train_x, test_x
