import argparse
import os
import time

import pandas as pd
from pynisher import limit, WallTimeoutException, MemoryLimitException

from src.datasets.Datasets import get_amlb_dataset, construct_dataframe

from src.feature_engineering.autofeat.Autofeat import get_autofeat_features
from src.feature_engineering.AutoGluon.AutoGluon import get_autogluon_features
from src.feature_engineering.BioAutoML.BioAutoML import get_bioautoml_features
from src.feature_engineering.Boruta.Boruta import get_boruta_features
from src.feature_engineering.CorrelationBasedFS.CorrelationBasedFS import get_correlationbased_features
from src.feature_engineering.Featuretools.Featuretools import get_featuretools_features
from src.feature_engineering.Featurewiz.Featurewiz import get_featurewiz_features
from src.feature_engineering.H2O.H2O import get_h2o_features
from src.feature_engineering.MACFE.MACFE import get_macfe_features
from src.feature_engineering.MAFESE.MAFESE import get_mafese_features
from src.feature_engineering.MLJAR.MLJAR import get_mljar_features
from src.feature_engineering.NFS.NFS import get_nfs_features
from src.feature_engineering.OpenFE.OpenFE import get_openFE_features


def main(args):
    task_id = args.method
    feature_engineering_methods = ["autofeat", "autogluon", "bioautoml", "boruta", "correlationbased", "featuretools",
                                   "featurewiz", "h2o", "macfe", "mafese", "mljar", "nfs", "openfe"]
    run_and_save(feature_engineering_methods, task_id)


def run_and_save(feature_engineering_methods, task_id):
    splits = 10
    for split in range(splits):
        train_x, train_y, test_x, test_y, name, task_hint = get_amlb_dataset(task_id, split)
        df = construct_dataframe(train_x, train_y, test_x, test_y)
        df.to_csv('src/datasets/feature_engineered_datasets/' + task_hint + "_" + name + '_original_' + str(split) + '.csv', index=False)
        df_times = pd.DataFrame()
        for method in feature_engineering_methods:
            if not os.path.isfile('src/datasets/feature_engineered_datasets/' + task_hint + '_' + name + '_' + method + '_' + str(split) + '.csv'):
                df_times = get_and_save_features(df_times, train_x, train_y, test_x, test_y, name, method, split, task_hint)
                df_times.to_csv('src/datasets/feature_engineered_datasets/exec_times/exec_times_' + name + '_' + method + '_' + str(split) + '.csv', index=False)


def get_and_save_features(df_times, train_x, train_y, test_x, test_y, name, method, split, task_hint):
    execution_time = 0
    df = pd.DataFrame()

    if method == "autofeat":
        try:
            fe = limit(get_autofeat_features, wall_time=(4, "h"), memory=(32, "GB"))
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, task_hint, 2, 5)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = pd.DataFrame()

    elif method == "autogluon":
        try:
            fe = limit(get_autogluon_features, wall_time=(4, "h"), memory=(32, "GB"))
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = pd.DataFrame()

    elif method == "bioautoml":
        estimations = 50
        try:
            fe = limit(get_bioautoml_features, wall_time=(4, "h"), memory=(32, "GB"))
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, estimations)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = pd.DataFrame()

    elif method == "boruta":
        try:
            fe = limit(get_boruta_features, wall_time=(4, "h"), memory=(32, "GB"))
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = pd.DataFrame()

    elif method == "correlationBasedFS":
        try:
            fe = limit(get_correlationbased_features, wall_time=(4, "h"), memory=(32, "GB"))
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = pd.DataFrame()

    elif method == "featuretools":
        try:
            fe = limit(get_featuretools_features, wall_time=(4, "h"), memory=(32, "GB"))
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, test_y, name)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = pd.DataFrame()

    elif method == "featurewiz":
        try:
            fe = limit(get_featurewiz_features, wall_time=(4, "h"), memory=(32, "GB"))
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = pd.DataFrame()

    elif method == "h2o":
        try:
            fe = limit(get_h2o_features, wall_time=(4, "h"), memory=(32, "GB"))
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = pd.DataFrame()

    elif method == "macfe":
        try:
            fe = limit(get_macfe_features, wall_time=(4, "h"), memory=(32, "GB"))
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, test_y, name)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = pd.DataFrame()

    elif method == "mafese":
        num_features = 50
        try:
            fe = limit(get_mafese_features, wall_time=(4, "h"), memory=(32, "GB"))
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, test_y, name, num_features)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = pd.DataFrame()

    elif method == "mljar":
        num_features = 50
        try:
            fe = limit(get_mljar_features, wall_time=(4, "h"), memory=(32, "GB"))
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, num_features)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = pd.DataFrame()

    elif method == "nfs":
        try:
            fe = limit(get_nfs_features, wall_time=(4, "h"), memory=(32, "GB"))
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, test_y, name)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = pd.DataFrame()

    elif method == "openfe":
        try:
            fe = limit(get_openFE_features, wall_time=(4, "h"), memory=(32, "GB"))
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, 1, name)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = pd.DataFrame()

    df.to_csv('src/datasets/feature_engineered_datasets/' + task_hint + '_' + name + '_' + method + '_' + split + '.csv', index=False)
    df_times = df_times._append({'Dataset': name, 'Method': method, 'Time': execution_time}, ignore_index=True)
    return df_times


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run feature engineering methods')
    parser.add_argument('--method', type=str, required=True, help='Feature engineering dataset to use')
    args = parser.parse_args()
    main(args)
