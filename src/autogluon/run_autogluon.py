import warnings

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from src.datasets.Datasets import preprocess_data, preprocess_target
from src.datasets.Splits import get_splits

warnings.simplefilter(action='ignore', category=FutureWarning)


datasets = ["abalone"]
methods = ["original", "autofeat"]  # , "autogluon", "bioautoml", "boruta", "correlationbased", "featuretools", "featurewiz", "h2o", "macfe", "mafese", "mljar", "openfe"]

for name in datasets:
    for method in methods:
        data = pd.read_csv(f'../datasets/feature_engineered_datasets/{name}_{method}.csv')
        label = data.columns[-1]

        X = data.drop(label, axis=1)
        y = data[label]

        train_x, train_y, test_x, test_y = get_splits(X, y)

        """
        train_x, test_x = preprocess_data(train_x, test_x)
        train_y = preprocess_target(train_y)
        test_y = preprocess_target(test_y)
        """

        train_data = pd.concat([train_x, train_y], axis=1)
        test_data = pd.concat([test_x, test_y], axis=1)
        train_data = TabularDataset(train_data)
        test_data = TabularDataset(test_data)

        predictor = TabularPredictor(label=label, verbosity=0).fit(train_data)

        y_pred = predictor.predict(test_data.drop(columns=[label]))
        print(y_pred.head())
        eval_dict = predictor.evaluate(test_data, silent=True)
        print(eval_dict)
