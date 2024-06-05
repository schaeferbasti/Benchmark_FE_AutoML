from __future__ import annotations

import warnings
from pathlib import Path
import os.path

from amltk.optimization import Metric
from amltk.pipeline import Choice, Sequential, Split
from sklearn.metrics import get_scorer
from sklearn.preprocessing import *

from src.amltk.classifiers.Classifiers import *
from src.amltk.datasets.Datasets import *
from src.amltk.evaluation.Evaluator import get_cv_evaluator
from src.amltk.optimizer.RandomSearch import RandomSearch

from src.amltk.feature_engineering.AutoGluon import get_autogluon_features
from src.amltk.feature_engineering.Autofeat import get_autofeat_features
from src.amltk.feature_engineering.OpenFE import get_openFE_features

warnings.simplefilter(action='ignore', category=FutureWarning)


preprocessing = Split(
    {
        "numerical": Component(SimpleImputer, space={"strategy": ["mean", "median"]}),
        "categorical": [
            Component(
                OrdinalEncoder,
                config={
                    "categories": "auto",
                    "handle_unknown": "use_encoded_value",
                    "unknown_value": -1,
                    "encoded_missing_value": -2,
                },
            ),
            Choice(
                "passthrough",
                Component(
                    OneHotEncoder,
                    space={"max_categories": (2, 20)},
                    config={
                        "categories": "auto",
                        "drop": None,
                        "sparse_output": False,
                        "handle_unknown": "infrequent_if_exist",
                    },
                ),
                name="one_hot",
            ),
        ],
    },
    name="preprocessing",
)


def safe_dataframe(df, working_dir, dataset_name, fold_number, method_name):
    file_string = "results_" + str(dataset_name) + "_" + str(method_name) + "_fold_" + str(fold_number) + ".parquet"
    results_to = working_dir / file_string
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)
    print(f"Saving dataframe of results to path: {results_to}")
    df.to_parquet(results_to)


rf_classifier = get_rf_classifier()
rf_pipeline = Sequential(preprocessing, rf_classifier, name="rf_pipeline")

# works on dataset 2 (not for continuous data)
mlp_classifier = get_mlp_classifier()
mlp_pipeline = Sequential(preprocessing, mlp_classifier, name="mlp_pipeline")

# works on dataset 2 (not on continuous data)
svc_classifier = get_svc_classifier()
svc_pipeline = Sequential(preprocessing, svc_classifier, name="svc_pipeline")

# works on dataset 2 (not on continuous data)
knn_classifier = get_knn_classifier()
knn_pipeline = Sequential(preprocessing, knn_classifier, name="knn_pipeline")

lgbm_classifier = get_lgbm_classifier()
lgbm_classifier_pipeline = Sequential(preprocessing, lgbm_classifier, name="lgbm_classifier_pipeline")

lgbm_regressor = get_lgbm_regressor()
lgbm_regressor_pipeline = Sequential(preprocessing, lgbm_regressor, name="lgbm_regressor_pipeline")


def main() -> None:
    rerun = False                               # Decide if you want to re-execute the methods on a dataset or use the existing files
    debugging = False                           # Decide if you want ot raise trial exceptions
    feat_eng_steps = 2                          # Number of feature engineering steps for autofeat
    feat_sel_steps = 5                          # Number of feature selection steps for autofeat
    # working_dir = Path("src/amltk/results")   # Path if running on Cluster
    working_dir = Path("results")               # Path for local execution
    random_seed = 42                            # Set seed
    folds = 10                                  # Set number of folds (normal 10, test 1)

    # Choose set of datasets
    all_datasets = [1, 5, 14, 15, 16, 17, 18, 21, 22, 23, 24, 27, 28, 29, 31, 35, 36]
    small_datasets = [1, 5, 14, 16, 17, 18, 21, 27, 31, 35, 36]
    smallest_datasets = [14, 16, 17, 21, 35]  # n ~ 1000, p ~ 15
    big_datasets = [15, 22, 23, 24, 28, 29]
    test_new_method_datasets = [16]

    optimizer_cls = RandomSearch
    pipeline = lgbm_classifier_pipeline

    metric_definition = Metric(
        "accuracy",
        minimize=False,
        bounds=(0, 1),
        fn=get_scorer("accuracy")
    )

    per_process_memory_limit = None  # (4, "GB")  # NOTE: May have issues on Mac
    per_process_walltime_limit = None  # (60, "s")

    if debugging:
        max_trials = 1  # don't care about quality of the found model
        max_time = 600  # 10 minutes
        n_workers = 4
        # raise an error with traceback, something went wrong
        on_trial_exception = "raise"
        display = True
        wait_for_all_workers_to_finish = False
    else:
        max_trials = 50  # trade-off between exploration and resource usage
        max_time = 3600  # one hour
        n_workers = 50
        # Just mark the trial as fail and move on to the next one
        on_trial_exception = "continue"
        display = True
        wait_for_all_workers_to_finish = False

    df_methods = pd.DataFrame()
    df_methods_datasets = pd.DataFrame()
    df_methods_datasets_folds = pd.DataFrame()

    for fold in range(folds):
        print("\n\n\n*******************************\n Fold " + str(fold) + "\n*******************************\n")
        inner_fold_seed = random_seed + fold
        # Iterate over all chosen datasets
        for option in small_datasets:
            # Get train test split dataset
            train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=option)
            """
            ############## Original Data ##############
            Use original data without feature engineering

            """
            print("Original Data")
            file_name = "results_" + str(name) + "_original_fold_" + str(fold) + ".parquet"
            file = working_dir / file_name
            print("\n\n\n*******************************\n" + str(file_name) + "\n*******************************\n")
            if rerun or not os.path.isfile(file):
                print("Run Original Method on Dataset")
                evaluator = get_cv_evaluator(train_x, train_y, test_x, test_y, inner_fold_seed,
                                             on_trial_exception, task_hint)

                history_original = pipeline.optimize(
                    target=evaluator.fn,
                    metric=metric_definition,
                    optimizer=optimizer_cls,
                    seed=inner_fold_seed,
                    process_memory_limit=per_process_memory_limit,
                    process_walltime_limit=per_process_walltime_limit,
                    working_dir=working_dir,
                    max_trials=max_trials,
                    timeout=max_time,
                    display=display,
                    wait=wait_for_all_workers_to_finish,
                    n_workers=n_workers,
                    on_trial_exception=on_trial_exception,
                )
                df_original = history_original.df()
                df_methods = df_methods._append(df_original)
                safe_dataframe(df_original, working_dir, name, fold, "original")
            else:
                # Read dataset from parquet (without executing all FE methods on the dataset)
                print("Read from Parquet")
                df_original = pd.read_parquet(file, engine='pyarrow')
                df_methods = df_methods._append(df_original)

            """
            ############## Feature Engineering with autofeat ##############
            Use Feature Engineering from autofeat
        
            """
            print("\n\nautofeat Data")
            file_name = "results_" + str(name) + "_autofeat_fold_" + str(fold) + ".parquet"
            file = working_dir / file_name
            print("\n\n\n*******************************\n" + str(file_name) + "\n*******************************\n")
            if rerun or not os.path.isfile(file):
                print("Run autofeat Method on Dataset")
                train_x_autofeat, test_x_autofeat = get_autofeat_features(train_x, train_y, test_x, task_hint,
                                                                          feat_eng_steps, feat_sel_steps)

                evaluator = get_cv_evaluator(train_x_autofeat, train_y, test_x_autofeat, test_y, inner_fold_seed,
                                             on_trial_exception, task_hint)

                history_autofeat = pipeline.optimize(
                    target=evaluator.fn,
                    metric=metric_definition,
                    optimizer=optimizer_cls,
                    seed=inner_fold_seed,
                    process_memory_limit=per_process_memory_limit,
                    process_walltime_limit=per_process_walltime_limit,
                    working_dir=working_dir,
                    max_trials=max_trials,
                    timeout=max_time,
                    display=display,
                    wait=wait_for_all_workers_to_finish,
                    n_workers=n_workers,
                    on_trial_exception=on_trial_exception,
                )
                df_autofeat = history_autofeat.df()
                df_methods = df_methods._append(df_autofeat)
                safe_dataframe(df_autofeat, working_dir, name, fold, "autofeat")
            else:
                # Read dataset from parquet (without executing all FE methods on the dataset)
                print("Read from Parquet")
                df_autofeat = pd.read_parquet(file, engine='pyarrow')
                df_methods = df_methods._append(df_autofeat)

            """
            ############## Feature Engineering with OpenFE ##############
            Use Feature Generation and Selection implemented by the OpenFE paper
        
            """
            print("\n\nOpenFE Data")
            file_name = "results_" + str(name) + "_openfe_fold_" + str(fold) + ".parquet"
            file = working_dir / file_name
            print("\n\n\n*******************************\n" + str(file_name) + "\n*******************************\n")
            if rerun or not os.path.isfile(file):
                print("Run OpenFE Method on Dataset")
                train_x_openfe, test_x_openfe = get_openFE_features(train_x, train_y, test_x, 1)

                evaluator = get_cv_evaluator(train_x_openfe, train_y, test_x_openfe, test_y, inner_fold_seed,
                                             on_trial_exception, task_hint)

                history_openFE = pipeline.optimize(
                    target=evaluator.fn,
                    metric=metric_definition,
                    optimizer=optimizer_cls,
                    seed=inner_fold_seed,
                    process_memory_limit=per_process_memory_limit,
                    process_walltime_limit=per_process_walltime_limit,
                    working_dir=working_dir,
                    max_trials=max_trials,
                    timeout=max_time,
                    display=display,
                    wait=wait_for_all_workers_to_finish,
                    n_workers=n_workers,
                    on_trial_exception=on_trial_exception,
                )
                df_openFE = history_openFE.df()
                df_methods = df_methods._append(df_openFE)
                safe_dataframe(df_autofeat, working_dir, name, fold, "openfe")
            else:
                # Read dataset from parquet (without executing all FE methods on the dataset)
                print("Read from Parquet")
                df_openFE = pd.read_parquet(file, engine='pyarrow')
                df_methods = df_methods._append(df_openFE)

            """
            ############## Feature Engineering with AutoGluon ##############
            Use AutoGluon Feature Generation and Selection

            """

            print("\n\nAutoGluon Data")
            file_name = "results_" + str(name) + "_autogluon_fold_" + str(fold) + ".parquet"
            file = working_dir / file_name
            print("\n\n\n*******************************\n" + str(file_name) + "\n*******************************\n")
            if rerun or not os.path.isfile(file):
                print("Run AutoGluon Method on Dataset")
                train_x_autogluon, test_x_autogluon = get_autogluon_features(train_x, train_y, test_x)

                evaluator = get_cv_evaluator(train_x_autogluon, train_y, test_x_autogluon, test_y, inner_fold_seed,
                                             on_trial_exception,
                                             task_hint)

                history_autogluon = pipeline.optimize(
                    target=evaluator.fn,
                    metric=metric_definition,
                    optimizer=optimizer_cls,
                    seed=inner_fold_seed,
                    process_memory_limit=per_process_memory_limit,
                    process_walltime_limit=per_process_walltime_limit,
                    working_dir=working_dir,
                    max_trials=max_trials,
                    timeout=max_time,
                    display=display,
                    wait=wait_for_all_workers_to_finish,
                    n_workers=n_workers,
                    on_trial_exception=on_trial_exception,
                )
                df_autogluon = history_autogluon.df()
                df_methods = df_methods._append(df_autogluon)
                safe_dataframe(df_autogluon, working_dir, name, fold, "autogluon")
            else:
                # Read dataset from parquet (without executing all FE methods on the dataset)
                print("Read from Parquet")
                df_autogluon = pd.read_parquet(file, engine='pyarrow')
                df_methods = df_methods._append(df_autogluon)
            # Append DF with all methods to methods_datasets and save it
            df_methods_datasets = df_methods_datasets._append(df_methods)
            safe_dataframe(df_methods, working_dir, name, fold, "all_methods")
        # Append DF with all methods & datasets to methods_datasets_folds and save it
        df_methods_datasets_folds = df_methods_datasets_folds._append(df_methods_datasets)
        safe_dataframe(df_methods_datasets, working_dir, "all_datasets", fold, "all_methods")
    # Safe dataframe containing results on all methods, datasets and over all folds
    safe_dataframe(df_methods_datasets_folds, working_dir, "all_datasets", "all_folds", "all_methods")


if __name__ == "__main__":
    main()
