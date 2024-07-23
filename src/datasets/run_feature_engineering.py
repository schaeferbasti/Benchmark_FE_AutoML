import time
import datetime

import pandas as pd
from pynisher import limit, WallTimeoutException, MemoryLimitException

from src.datasets.Datasets import get_amlb_dataset, construct_dataframe
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
from src.feature_engineering.OpenFE.OpenFE import get_openFE_features
from src.feature_engineering.autofeat.Autofeat import get_autofeat_features


def main():
    amlb_task_ids = [
        # regression
        359944
        ]
    """
    , 359929, 233212, 359937, 359950,)
            359938, 233213, 359942, 233211, 359936,
            359952, 359951, 359949, 233215, 360945,
            167210, 359943, 359941, 359946, 360933,
            360932, 359930, 233214, 359948, 359931,
            359932, 359933, 359934, 359939, 359945,
            359935, 317614, 359940,
            # classification
            190411, 359983, 189354, 189356, 10090,
            359979, 168868, 190412, 146818, 359982,
            359967, 359955, 359960, 359973, 359968,
            359992, 359959, 359957, 359977, 7593,
            168757, 211986, 168909, 189355, 359964,
            359954, 168910, 359976, 359969, 359970,
            189922, 359988, 359984, 360114, 359966,
            211979, 168911, 359981, 359962, 360975,
            3945, 360112, 359991, 359965, 190392,
            359961, 359953, 359990, 359980, 167120,
            359993, 190137, 359958, 190410, 359971,
            168350, 360113, 359956, 359989, 359986,
            359975, 359963, 359994, 359987, 168784,
            359972, 190146, 359985, 146820, 359974,
            2073
        ]
        """
    feature_engineering_methods = ["autofeat", "autogluon"]  # , "bioautoml", "boruta", "correlationbased", "featuretools", "featurewiz", "h2o", "macfe", "mafese", "mljar", "openfe"]
    for task_id in amlb_task_ids:
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
        fe = limit(get_autofeat_features, wall_time=(4, "h")) # , memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, task_hint, 2, 5)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "autogluon":
        fe = limit(get_autogluon_features, wall_time=(4, "h"))#, memory=(32, "GB"))
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
        fe = limit(get_bioautoml_features, wall_time=(4, "h"))#, memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, estimations)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "bortua":
        fe = limit(get_boruta_features, wall_time=(4, "h"))#, memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "correlationBasedFS":
        fe = limit(get_correlationbased_features, wall_time=(4, "h"))#, memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "featuretools":
        fe = limit(get_featuretools_features, wall_time=(4, "h"))#, memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, test_y, name)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "featurewiz":
        fe = limit(get_featurewiz_features, wall_time=(4, "h"))#, memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "h2o":
        fe = limit(get_h2o_features, wall_time=(4, "h"))#, memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "macfe":
        fe = limit(get_macfe_features, wall_time=(4, "h"))#, memory=(32, "GB"))
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
        fe = limit(get_mafese_features, wall_time=(4, "h"))#, memory=(32, "GB"))
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
        fe = limit(get_mljar_features, wall_time=(4, "h"))#, memory=(32, "GB"))
        try:
            start_time = time.time()  #
            train_x, test_x = fe(train_x, train_y, test_x, num_features)
            end_time = time.time()  #
            execution_time = end_time - start_time
            df = construct_dataframe(train_x, train_y, test_x, test_y)
        except (WallTimeoutException, MemoryLimitException):
            df = None

    elif method == "openfe":
        fe = limit(get_openFE_features, wall_time=(4, "h"))#, memory=(32, "GB"))
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
    main()
