import argparse
import warnings

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.exceptions import UndefinedMetricWarning

from src.datasets.Datasets import preprocess_data, preprocess_target
from src.datasets.Splits import get_splits

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)

def main(args):
    folds = 10
    csv_files = []
    dataset_file = args.method
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
        max_memory_usage_ratio = 1.0  # share of total memory (we want to give 32GB to autogluon and limit the cluster to this amount)
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

        for fold in range(folds):
            train_x, train_y, test_x, test_y = get_splits(X, y, fold)

            train_x, test_x = preprocess_data(train_x, test_x)
            train_y = preprocess_target(train_y)
            test_y = preprocess_target(test_y)

            train_data = pd.concat([train_x, train_y], axis=1)
            test_data = pd.concat([test_x, test_y], axis=1)
            train_data = TabularDataset(train_data)
            test_data = TabularDataset(test_data)

            path = "logs/autogluon_" + dataset + "_" + method + "_" + fold + ".csv"

            if task_hint == 'regression':
                predictor = TabularPredictor(label=label, verbosity=0, problem_type=task_hint, eval_metric="root_mean_squared_error", path=path).fit(
                    train_data=train_data, time_limit=time_limit, num_cpus=num_cpus, presets="best_quality", memory_limit=32)
                eval_dict = predictor.evaluate(test_data)
                leaderboard = predictor.leaderboard(test_data)
            elif task_hint == 'binary':
                predictor = TabularPredictor(label=label, verbosity=2, problem_type=task_hint, eval_metric="roc_auc", path=path).fit(
                    train_data=train_data, time_limit=time_limit, num_cpus=num_cpus, presets="best_quality", memory_limit=32)
                eval_dict = predictor.evaluate(test_data)
                leaderboard = predictor.leaderboard(test_data)
            elif task_hint == 'multiclass':
                predictor = TabularPredictor(label=label, verbosity=4, problem_type=task_hint, eval_metric="log_loss", path=path).fit(
                    train_data=train_data, time_limit=time_limit, num_cpus=num_cpus, presets="best_quality", memory_limit=32)
                eval_dict = predictor.evaluate(test_data)
                leaderboard = predictor.leaderboard(test_data)
            print(eval_dict)
            print(leaderboard)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run feature engineering methods')
    parser.add_argument('--method', type=str, required=True, help='Feature engineering method to use')
    args = parser.parse_args()
    main(args)