import warnings

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.exceptions import UndefinedMetricWarning

from src.datasets.Datasets import preprocess_data, preprocess_target
from src.datasets.Splits import get_splits

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)

task_hint = 'regression'
datasets = ["abalone"]
methods = ["original", "autofeat", "autogluon"]  # , "bioautoml", "boruta", "correlationbased", "featuretools", "featurewiz", "h2o", "macfe", "mafese", "mljar", "openfe"]

for name in datasets:
    for method in methods:
        print(f"\n****************************************\n{name} - {method}\n****************************************")
        data = pd.read_csv(f'../datasets/feature_engineered_datasets/{name}_{method}.csv')
        label = data.columns[-1]

        X = data.drop(label, axis=1)
        y = data[label]

        train_x, train_y, test_x, test_y = get_splits(X, y)

        train_x, test_x = preprocess_data(train_x, test_x)
        train_y = preprocess_target(train_y)
        test_y = preprocess_target(test_y)

        train_data = pd.concat([train_x, train_y], axis=1)
        test_data = pd.concat([test_x, test_y], axis=1)
        train_data = TabularDataset(train_data)
        test_data = TabularDataset(test_data)
        if task_hint == 'regression':
            predictor = TabularPredictor(label=label, verbosity=0, eval_metric="root_mean_squared_error").fit(train_data)
            eval_dict = predictor.evaluate(test_data)
        elif task_hint == 'binary_classification':
            predictor = TabularPredictor(label=label, verbosity=0, eval_metric="roc_auc").fit(train_data)
            eval_dict = predictor.evaluate(test_data)
        elif task_hint == 'multi_classification':
            predictor = TabularPredictor(label=label, verbosity=0, eval_metric="log_loss").fit(train_data)
            eval_dict = predictor.evaluate(test_data)
        print(eval_dict)
