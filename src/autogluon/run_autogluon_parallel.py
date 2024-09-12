import argparse
import warnings
import os
from pathlib import Path

import pandas as pd
from src.autogluon.method.tabular.src.autogluon.tabular import (TabularDataset, TabularPredictor)
from sklearn.exceptions import UndefinedMetricWarning

from src.datasets.Datasets import preprocess_data, preprocess_target
from src.datasets.Splits import get_splits

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)

def main(args):
    folder = args.method
    path = f'src/datasets/feature_engineered_datasets/' + folder
    dataset_files = sorted(os.listdir(Path(path)))
    eval_df = pd.DataFrame()
    method = None
    dataset = None
    task_hint = None
    for dataset_file in dataset_files:
        core_name = dataset_file[:-len('.parquet')]
        parts = core_name.split('_')
        task_hint = parts[0]
        method = parts[-2]
        fold = parts[-1]
        parts.remove(task_hint)
        parts.remove(method)
        parts.remove(fold)
        dataset = '_'.join(parts)
        print(
            f"\n****************************************\n {task_hint} - {dataset} - {method} - {fold} \n****************************************")
        try:
            exec_path = f"src/datasets/feature_engineered_datasets/exec_times/exec_times_{dataset}_{method}_{fold}.parquet"
            execution_times = pd.read_parquet(Path(exec_path))
            result = execution_times[(execution_times['Dataset'] == dataset) & (execution_times['Method'] == method)]
            exec_time = result['Time'].values[0]
        except FileNotFoundError:
            exec_time = 0  # no FE method executed on dataset -> raw dataset
        num_cpus = 8
        memory_limit = 32
        time_limit = 14400 - exec_time  # 4h in seconds - time needed for feature engineering
        print(f"Time limit: {time_limit}, Exec time: {exec_time}")

        leaderboard = pd.DataFrame()
        eval_dict = None
        if time_limit >= 0:
            print("Read Data")
            data_path = f'src/datasets/feature_engineered_datasets/{task_hint}_{dataset}_{method}/{task_hint}_{dataset}_{method}_{fold}.parquet'
            data = pd.read_parquet(Path(data_path))
            try:
                label = data.columns[-1]
                X = data.drop(label, axis=1)
                y = data[label]
                print("Split data")
                train_x, train_y, test_x, test_y = get_splits(X, y, int(fold))
                train_x, test_x = preprocess_data(train_x, test_x)
                train_y = preprocess_target(train_y)
                test_y = preprocess_target(test_y)

                train_data = pd.concat([train_x, train_y], axis=1)
                test_data = pd.concat([test_x, test_y], axis=1)
                train_data = TabularDataset(train_data)
                test_data = TabularDataset(test_data)

                print("Use Tabular Predictor")
                if task_hint == 'regression':
                    print("Regression")
                    predictor = TabularPredictor(label=label, verbosity=4, problem_type=task_hint,
                                                 eval_metric="root_mean_squared_error").fit(
                        train_data=train_data, time_limit=time_limit, num_cpus=num_cpus, presets="best_quality",
                        memory_limit=memory_limit)
                    eval_dict = predictor.evaluate(test_data)
                    leaderboard = predictor.leaderboard(test_data)
                elif task_hint == 'binary-classification':
                    try:
                        task_hint = 'binary'
                        print("Binary Classification")
                        predictor = TabularPredictor(label=label, verbosity=4, problem_type=task_hint,
                                                     eval_metric="roc_auc").fit(
                            train_data=train_data, time_limit=time_limit, num_cpus=num_cpus, presets="best_quality",
                            memory_limit=memory_limit)
                        eval_dict = predictor.evaluate(test_data)
                        leaderboard = predictor.leaderboard(test_data)
                    except AssertionError:
                        task_hint = 'multiclass'
                        print("Multiclass Classification")
                        predictor = TabularPredictor(label=label, verbosity=4, problem_type=task_hint,
                                                     eval_metric="log_loss").fit(
                            train_data=train_data, time_limit=time_limit, num_cpus=num_cpus, presets="best_quality",
                            memory_limit=memory_limit)
                        eval_dict = predictor.evaluate(test_data)
                        leaderboard = predictor.leaderboard(test_data)
                elif task_hint == 'multiclass-classification':
                    task_hint = 'multiclass'
                    print("Multiclass Classification")
                    predictor = TabularPredictor(label=label, verbosity=4, problem_type=task_hint,
                                                 eval_metric="log_loss").fit(
                        train_data=train_data, time_limit=time_limit, num_cpus=num_cpus, presets="best_quality",
                        memory_limit=memory_limit)
                    eval_dict = predictor.evaluate(test_data)
                    leaderboard = predictor.leaderboard(test_data)
                print("Write leaderboard + Append eval_df")
                path_leaderboard = f"src/autogluon/results/leaderboard/leaderboard_{task_hint}_{dataset}_{method}_{fold}.parquet"
                leaderboard.to_parquet(Path(path_leaderboard))
                eval_df = eval_df._append(pd.DataFrame(eval_dict, index=[0]))
            except IndexError:
                print("No Data")
    print("Write eval_df")
    eval_path = f"src/autogluon/results/evaldict/evaldict_{task_hint}_{dataset}_{method}.parquet"
    eval_df.to_parquet(Path(eval_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run autogluon')
    parser.add_argument('--method', type=str, required=True, help='Feature engineering method to use')
    args = parser.parse_args()
    main(args)

