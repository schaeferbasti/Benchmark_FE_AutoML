import argparse
import time

import pandas as pd
from pynisher import limit, WallTimeoutException, MemoryLimitException

from datasets.Datasets import get_amlb_dataset, construct_dataframe
from AutoGluon.AutoGluon import get_autogluon_features
from BioAutoML.BioAutoML import get_bioautoml_features
from Boruta.Boruta import get_boruta_features
from CorrelationBasedFS.CorrelationBasedFS import get_correlationbased_features
from Featuretools.Featuretools import get_featuretools_features
from Featurewiz.Featurewiz import get_featurewiz_features
from H2O.H2O import get_h2o_features
from MACFE.MACFE import get_macfe_features
from MAFESE.MAFESE import get_mafese_features
from MLJAR.MLJAR import get_mljar_features
from OpenFE.OpenFE import get_openFE_features
from autofeat.Autofeat import get_autofeat_features


def main(args):
    task_id = args.method
    feature_engineering_methods = ["autofeat", "autogluon", "bioautoml", "boruta", "correlationbased", "featuretools",
                                   "featurewiz", "h2o", "macfe", "mafese", "mljar", "openfe"]
    run_and_save(feature_engineering_methods, task_id)


def run_and_save(feature_engineering_methods, task_id):
    train_x, train_y, test_x, test_y, name, task_hint = get_amlb_dataset(task_id)
    df = construct_dataframe(train_x, train_y, test_x, test_y)
    df.to_csv('datasets/feature_engineered_datasets/' + task_hint + "_" + name + '_original.csv', index=False)
    df_times = pd.DataFrame()
    for method in feature_engineering_methods:
        df_times = get_and_save_features(df_times, train_x, train_y, test_x, test_y, name, method, task_hint)
    df_times.to_csv('datasets/feature_engineered_datasets/exec_times.csv', index=False)


def get_and_save_features(df_times, train_x, train_y, test_x, test_y, name, method, task_hint):
    execution_time = 0
    df = None

    if method == "autofeat":
        fe = limit(get_autofeat_features, wall_time=(4, "h"), memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, task_hint, 2, 5)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "autogluon":
        fe = limit(get_autogluon_features, wall_time=(4, "h"), memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "bioautoml":
        estimations = 50
        fe = limit(get_bioautoml_features, wall_time=(4, "h"), memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, estimations)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "bortua":
        fe = limit(get_boruta_features, wall_time=(4, "h"), memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "correlationBasedFS":
        fe = limit(get_correlationbased_features, wall_time=(4, "h"), memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "featuretools":
        fe = limit(get_featuretools_features, wall_time=(4, "h"), memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, test_y, name)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "featurewiz":
        fe = limit(get_featurewiz_features, wall_time=(4, "h"), memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "h2o":
        fe = limit(get_h2o_features, wall_time=(4, "h"), memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "macfe":
        fe = limit(get_macfe_features, wall_time=(4, "h"), memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, test_y, name)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "mafese":
        num_features = 50
        fe = limit(get_mafese_features, wall_time=(4, "h"), memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, test_y, name, num_features)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "mljar":
        num_features = 50
        fe = limit(get_mljar_features, wall_time=(4, "h"), memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, num_features)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "openfe":
        fe = limit(get_openFE_features, wall_time=(4, "h"), memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, 1)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    df.to_csv('datasets/feature_engineered_datasets/' + task_hint + '_' + name + '_' + method + '.csv', index=False)
    df_times = df_times._append({'Dataset': name, 'Method': method, 'Time': execution_time}, ignore_index=True)
    return df_times


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run feature engineering methods')
    parser.add_argument('--method', type=str, required=True, help='Feature engineering method to use')
    args = parser.parse_args()
    main(args)
