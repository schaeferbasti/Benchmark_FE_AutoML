import warnings

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.exceptions import UndefinedMetricWarning

from src.datasets.Datasets import preprocess_data, preprocess_target
from src.datasets.Splits import get_splits

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)

datasets = ["abalone"]
methods = ["original", "autofeat", "autogluon"]  # , "bioautoml", "boruta", "correlationbased", "featuretools", "featurewiz", "h2o", "macfe", "mafese", "mljar", "openfe"]

for name in datasets:
    for method in methods:
        print(f"\n****************************************\n{name} - {method}\n****************************************")
        execution_times = pd.read_csv(f"../datasets/feature_engineered_datasets/exec_times.csv")
        result = execution_times[(execution_times['Dataset'] == name) & (execution_times['Method'] == method)]
        # Extract the time value if a match is found
        if not result.empty:  # method has been executed on dataset
            exec_time = result['Time'].values[0]
        else:
            exec_time = 0 # no FE method executed on dataset -> raw dataset

        time_limit = 14400 - exec_time
        print(f"Time limit: {time_limit}")
        try:
            data = pd.read_csv(f'../datasets/feature_engineered_datasets/regression_{name}_{method}.csv')
            task_hint = 'regression'
        except:
            try:
                data = pd.read_csv(f'../datasets/feature_engineered_datasets/binary-classification_{name}_{method}.csv')
                task_hint = 'binary-classification'
            except:
                data = pd.read_csv(f'../datasets/feature_engineered_datasets/multi-classification_{name}_{method}.csv')
                task_hint = 'multi-classification'
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
            predictor = TabularPredictor(label=label, verbosity=0, eval_metric="root_mean_squared_error").fit(train_data, time_limit=time_limit, num_cpus=8)
            eval_dict = predictor.evaluate(test_data)
        elif task_hint == 'binary_classification':
            predictor = TabularPredictor(label=label, verbosity=0, eval_metric="roc_auc").fit(train_data, time_limit=time_limit, num_cpus=8)
            eval_dict = predictor.evaluate(test_data)
        elif task_hint == 'multi_classification':
            predictor = TabularPredictor(label=label, verbosity=0, eval_metric="log_loss").fit(train_data, time_limit=time_limit, num_cpus=8)
            eval_dict = predictor.evaluate(test_data)
        print(eval_dict)
