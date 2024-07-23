import warnings
import os

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.exceptions import UndefinedMetricWarning

from src.datasets.Datasets import preprocess_data, preprocess_target
from src.datasets.Splits import get_splits

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)

datasets = ["abalone"]
methods = ["original", "autofeat", "autogluon"]  # , "bioautoml", "boruta", "correlationbased", "featuretools", "featurewiz", "h2o", "macfe", "mafese", "mljar", "openfe"]

dataset_files = os.listdir("../datasets/feature_engineered_datasets/")
csv_files = []
for dataset_file in dataset_files:
    if dataset_file.endswith(".csv") and not dataset_file.__contains__("exec_times"):
        csv_files.append(dataset_file)
        core_name = dataset_file[:-len('.csv')]
        parts = core_name.split('_')
        task_hint = parts[0]
        dataset = parts[1]
        method = parts[2]

        print(f"\n****************************************\n{dataset} - {method}\n****************************************")
        execution_times = pd.read_csv(f"../datasets/feature_engineered_datasets/exec_times.csv")
        result = execution_times[(execution_times['Dataset'] == dataset) & (execution_times['Method'] == method)]
        # Extract the time value if a match is found
        if not result.empty:  # method has been executed on dataset
            exec_time = result['Time'].values[0]
        else:
            exec_time = 0  # no FE method executed on dataset -> raw dataset

        time_limit = 14400 - exec_time  # 4h in seconds - time needed for feature engineering
        max_memory_usage_ratio = 0.1  # share of total memory (we want to give 32GB to autogluon)
        num_cpus = 8

        print(f"Time limit: {time_limit}")
        try:
            data = pd.read_csv(f'../datasets/feature_engineered_datasets/regression_{dataset}_{method}.csv')
            task_hint = 'regression'
        except:
            try:
                data = pd.read_csv(f'../datasets/feature_engineered_datasets/binary-classification_{dataset}_{method}.csv')
                task_hint = 'binary'
            except:
                data = pd.read_csv(f'../datasets/feature_engineered_datasets/multi-classification_{dataset}_{method}.csv')
                task_hint = 'multiclass'
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

        eval_dict = None
        if task_hint == 'regression':
            predictor = TabularPredictor(label=label, verbosity=0, problem_type=task_hint, eval_metric="root_mean_squared_error").fit(train_data, time_limit=time_limit, num_cpus=num_cpus, ag_args_fit={max_memory_usage_ratio: max_memory_usage_ratio})
            eval_dict = predictor.evaluate(test_data)
        elif task_hint == 'binary':
            predictor = TabularPredictor(label=label, verbosity=0, problem_type=task_hint, eval_metric="roc_auc").fit(train_data, time_limit=time_limit, num_cpus=num_cpus, ag_args_fit={max_memory_usage_ratio: max_memory_usage_ratio})
            eval_dict = predictor.evaluate(test_data)
        elif task_hint == 'multiclass':
            predictor = TabularPredictor(label=label, verbosity=0, problem_type=task_hint, eval_metric="log_loss").fit(train_data, time_limit=time_limit, num_cpus=num_cpus, ag_args_fit={max_memory_usage_ratio: max_memory_usage_ratio})
            eval_dict = predictor.evaluate(test_data)
        print(eval_dict)
