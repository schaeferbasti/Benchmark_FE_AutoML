from __future__ import annotations

import warnings
import os.path
from pathlib import Path
import argparse
import os

from amltk.optimization import Metric
from amltk.pipeline import Choice, Sequential, Split
from sklearn.metrics import get_scorer
from sklearn.preprocessing import *

from src.amltk.classifiers.Classifiers import *
from src.amltk.datasets.Datasets import *
from src.amltk.evaluation.Evaluator import get_cv_evaluator
from src.amltk.optimizer.RandomSearch import RandomSearch

from src.amltk.feature_engineering.autofeat.Autofeat import get_autofeat_features
from src.amltk.feature_engineering.AutoGluon.AutoGluon import get_autogluon_features
from src.amltk.feature_engineering.BioAutoML.BioAutoML import get_bioautoml_features
from src.amltk.feature_engineering.Boruta.Boruta import get_boruta_features
# CAAFE
from src.amltk.feature_engineering.CorrelationBasedFS.CorrelationBasedFS import get_correlationbased_features
# DIFER
# ExploreKit
from src.amltk.feature_engineering.Featuretools.Featuretools import get_featuretools_features
# from src.amltk.feature_engineering.Featurewiz.Featurewiz import get_featurewiz_features
from src.amltk.feature_engineering.H2O.H2O import get_h2o_features
from src.amltk.feature_engineering.MLJAR.MLJAR import get_mljar_features
from src.amltk.feature_engineering.OpenFE.OpenFE import get_openFE_features

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


def safe_dataframe(df, working_dir, dataset_name, fold_number, method_name, classifier_name):
    file_string = f"results_{dataset_name}_{method_name}_{classifier_name}_{fold_number}.parquet"
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

knn_classifier = get_knn_classifier()
knn_pipeline = Sequential(preprocessing, knn_classifier, name="knn_pipeline")

lgbm_classifier = get_lgbm_classifier()
lgbm_classifier_pipeline = Sequential(preprocessing, lgbm_classifier, name="lgbm_classifier_pipeline")


def main(args):
    method = args.method

    rerun = False  # Decide if you want to re-execute the methods on a dataset or use the existing files
    debugging = False  # Decide if you want ot raise trial exceptions
    feat_eng_steps = 2  # Number of feature engineering steps for autofeat
    feat_sel_steps = 5  # Number of feature selection steps for autofeat
    n_jobs = 1  # Number of jobs for OpenFE
    num_features = 500  # Number of features for MLJAR
    working_dir = Path("src/amltk/results")  # Path if running on Cluster
    # working_dir = Path("results")  # Path for local execution
    random_seed = 42  # Set seed
    folds = 10  # Set number of folds (normal 10, test 1)

    # Choose set of datasets
    all_datasets = [1, 5, 14, 15, 16, 17, 18, 21, 22, 23, 24, 27, 28, 29, 31, 35, 36]  # 17
    small_datasets = [1, 5, 14, 16, 17, 18, 21, 27, 31, 35, 36]
    smallest_datasets = [14, 16, 17, 21, 35]  # n ~ 1000, p ~ 15
    big_datasets = [15, 22, 23, 24, 28, 29]
    test_new_method_datasets = [18]  # [16]

    optimizer_cls = RandomSearch
    pipelines = [lgbm_classifier_pipeline, knn_pipeline, svc_pipeline, mlp_pipeline, rf_classifier]

    metric_definition = Metric(
        "roc_auc_ovo",
        minimize=False,
        bounds=(0, 1),
        fn=get_scorer("roc_auc_ovo")
    )

    per_process_memory_limit = None  # (4, "GB")  # NOTE: May have issues on Mac
    per_process_walltime_limit = None  # (60, "s")

    if debugging:
        max_trials = 1  # don't care about quality of the found model
        max_time = 600  # 10 minutes
        n_workers = 20
        # raise an error with traceback, something went wrong
        on_trial_exception = "raise"
        display = True
        wait_for_all_workers_to_finish = False
    else:
        max_trials = 100000  # trade-off between exploration and resource usage
        max_time = 3600  # one hour
        n_workers = 4
        # Just mark the trial as fail and move on to the next one
        on_trial_exception = "continue"
        display = True
        wait_for_all_workers_to_finish = False

    for fold in range(folds):
        print(f"\n\n\n*******************************\n Fold {fold}\n*******************************\n")
        inner_fold_seed = random_seed + fold
        for pipeline in pipelines:
            try:
                if method.startswith("original"):
                    print("Original Data")
                    option = method[-2:]
                    try:
                        int(option)
                    except ValueError as e:
                        option = method[-1:]
                    int(option)
                    pipeline_name = pipeline.name
                    train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=option)
                    file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                    print(file_name)
                    file = working_dir / file_name
                    if rerun or not os.path.isfile(file):
                        evaluator = get_cv_evaluator(train_x, train_y, test_x, test_y, inner_fold_seed,
                                                     on_trial_exception, task_hint)
                        print("Evaluator done")
                        history = pipeline.optimize(
                            target=evaluator.fn,
                            metric=metric_definition,
                            optimizer=optimizer_cls,
                            seed=inner_fold_seed,
                            max_trials=max_trials,
                            timeout=max_time,
                            display=display,
                            wait=wait_for_all_workers_to_finish,
                            n_workers=n_workers,
                            on_trial_exception=on_trial_exception
                        )
                        print("History done")
                        df = history.df()
                        safe_dataframe(df, working_dir, name, fold, method, pipeline_name)
                        print("Saved to df")

                elif method.startswith("autofeat"):
                    print("autofeat Data")
                    option = method[-2:]
                    try:
                        int(option)
                    except ValueError as e:
                        option = method[-1:]
                    int(option)
                    pipeline_name = pipeline.name
                    train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=option)
                    file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                    file = working_dir / file_name
                    if rerun or not os.path.isfile(file):
                        train_x, test_x = get_autofeat_features(train_x, train_y, test_x, task_hint, feat_eng_steps, feat_sel_steps)
                        evaluator = get_cv_evaluator(train_x, train_y, test_x, test_y, inner_fold_seed,
                                                     on_trial_exception, task_hint)
                        history = pipeline.optimize(
                            target=evaluator.fn,
                            metric=metric_definition,
                            optimizer=optimizer_cls,
                            seed=inner_fold_seed,
                            max_trials=max_trials,
                            timeout=max_time,
                            display=display,
                            wait=wait_for_all_workers_to_finish,
                            n_workers=n_workers,
                            on_trial_exception=on_trial_exception
                        )
                        df = history.df()
                        safe_dataframe(df, working_dir, name, fold, method, pipeline_name)

                elif method.startswith("autogluon"):
                    print("autogluon Data")
                    option = method[-2:]
                    try:
                        int(option)
                    except ValueError as e:
                        option = method[-1:]
                    int(option)
                    pipeline_name = pipeline.name
                    train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=option)
                    file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                    file = working_dir / file_name
                    if rerun or not os.path.isfile(file):
                        train_x, test_x = get_autogluon_features(train_x, train_y, test_x)
                        evaluator = get_cv_evaluator(train_x, train_y, test_x, test_y, inner_fold_seed,
                                                     on_trial_exception, task_hint)
                        history = pipeline.optimize(
                            target=evaluator.fn,
                            metric=metric_definition,
                            optimizer=optimizer_cls,
                            seed=inner_fold_seed,
                            max_trials=max_trials,
                            timeout=max_time,
                            display=display,
                            wait=wait_for_all_workers_to_finish,
                            n_workers=n_workers,
                            on_trial_exception=on_trial_exception
                        )
                        df = history.df()
                        safe_dataframe(df, working_dir, name, fold, method, pipeline_name)


                elif method.startswith("bioautoml"):
                    print("BioAutoML Data")
                    option = method[-2:]
                    try:
                        int(option)
                    except ValueError as e:
                        option = method[-1:]
                    int(option)
                    pipeline_name = pipeline.name
                    train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=option)
                    file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                    file = working_dir / file_name
                    if rerun or not os.path.isfile(file):
                        train_x, test_x = get_bioautoml_features(train_x, train_y, test_x)
                        evaluator = get_cv_evaluator(train_x, train_y, test_x, test_y, inner_fold_seed,
                                                     on_trial_exception,
                                                     task_hint)
                        history = pipeline.optimize(
                            target=evaluator.fn,
                            metric=metric_definition,
                            optimizer=optimizer_cls,
                            seed=inner_fold_seed,
                            max_trials=max_trials,
                            timeout=max_time,
                            display=display,
                            wait=wait_for_all_workers_to_finish,
                            n_workers=n_workers,
                            on_trial_exception=on_trial_exception
                        )
                        df = history.df()
                        safe_dataframe(df, working_dir, name, fold, method, pipeline_name)


                elif method.startswith("boruta"):
                    print("boruta Data")
                    option = method[-2:]
                    try:
                        int(option)
                    except ValueError as e:
                        option = method[-1:]
                    int(option)
                    pipeline_name = pipeline.name
                    train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=option)
                    file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                    file = working_dir / file_name
                    if rerun or not os.path.isfile(file):
                        train_x, test_x = get_boruta_features(train_x, train_y, test_x)
                        evaluator = get_cv_evaluator(train_x, train_y, test_x, test_y, inner_fold_seed,
                                                     on_trial_exception, task_hint)
                        history = pipeline.optimize(
                            target=evaluator.fn,
                            metric=metric_definition,
                            optimizer=optimizer_cls,
                            seed=inner_fold_seed,
                            max_trials=max_trials,
                            timeout=max_time,
                            display=display,
                            wait=wait_for_all_workers_to_finish,
                            n_workers=n_workers,
                            on_trial_exception=on_trial_exception
                        )
                        df = history.df()
                        safe_dataframe(df, working_dir, name, fold, method, pipeline_name)

                elif method.startswith("correlationBasedFS"):
                    print("CorrelationBasedFS Data")
                    option = method[-2:]
                    try:
                        int(option)
                    except ValueError as e:
                        option = method[-1:]
                    int(option)
                    pipeline_name = pipeline.name
                    train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=option)
                    file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                    file = working_dir / file_name
                    if rerun or not os.path.isfile(file):
                        train_x, test_x = get_correlationbased_features(train_x, train_y, test_x)
                        evaluator = get_cv_evaluator(train_x, train_y, test_x, test_y, inner_fold_seed,
                                                     on_trial_exception, task_hint)
                        history = pipeline.optimize(
                            target=evaluator.fn,
                            metric=metric_definition,
                            optimizer=optimizer_cls,
                            seed=inner_fold_seed,
                            max_trials=max_trials,
                            timeout=max_time,
                            display=display,
                            wait=wait_for_all_workers_to_finish,
                            n_workers=n_workers,
                            on_trial_exception=on_trial_exception
                        )
                        df = history.df()
                        safe_dataframe(df, working_dir, name, fold, method, pipeline_name)

                elif method.startswith("featuretools"):
                    print("Featuretools Data")
                    option = method[-2:]
                    try:
                        int(option)
                    except ValueError as e:
                        option = method[-1:]
                    int(option)
                    pipeline_name = pipeline.name
                    train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=option)
                    file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                    file = working_dir / file_name
                    if rerun or not os.path.isfile(file):
                        train_x, test_x = get_featuretools_features(train_x, train_y, test_x, test_y, name)
                        evaluator = get_cv_evaluator(train_x, train_y, test_x, test_y, inner_fold_seed,
                                                     on_trial_exception, task_hint)
                        history = pipeline.optimize(
                            target=evaluator.fn,
                            metric=metric_definition,
                            optimizer=optimizer_cls,
                            seed=inner_fold_seed,
                            max_trials=max_trials,
                            timeout=max_time,
                            display=display,
                            wait=wait_for_all_workers_to_finish,
                            n_workers=n_workers,
                            on_trial_exception=on_trial_exception
                        )
                        df = history.df()
                        safe_dataframe(df, working_dir, name, fold, method, pipeline_name)

                elif method.startswith("h2o"):
                    print("H2O Data")
                    option = method[-2:]
                    try:
                        int(option)
                    except ValueError as e:
                        option = method[-1:]
                    int(option)
                    pipeline_name = pipeline.name
                    train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=option)
                    file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                    file = working_dir / file_name
                    if rerun or not os.path.isfile(file):
                        train_x, test_x = get_h2o_features(train_x, train_y, test_x)
                        evaluator = get_cv_evaluator(train_x, train_y, test_x, test_y, inner_fold_seed,
                                                     on_trial_exception, task_hint)
                        history = pipeline.optimize(
                            target=evaluator.fn,
                            metric=metric_definition,
                            optimizer=optimizer_cls,
                            seed=inner_fold_seed,
                            max_trials=max_trials,
                            timeout=max_time,
                            display=display,
                            wait=wait_for_all_workers_to_finish,
                            n_workers=n_workers,
                            on_trial_exception=on_trial_exception
                        )
                        df = history.df()
                        safe_dataframe(df, working_dir, name, fold, method, pipeline_name)

                elif method.startswith("mljar"):
                    print("MLJAR Data")
                    option = method[-2:]
                    try:
                        int(option)
                    except ValueError as e:
                        option = method[-1:]
                    int(option)
                    pipeline_name = pipeline.name
                    train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=option)
                    file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                    file = working_dir / file_name
                    if rerun or not os.path.isfile(file):
                        train_x, test_x = get_mljar_features(train_x, train_y, test_x, num_features)
                        evaluator = get_cv_evaluator(train_x, train_y, test_x, test_y, inner_fold_seed,
                                                     on_trial_exception, task_hint)
                        history = pipeline.optimize(
                            target=evaluator.fn,
                            metric=metric_definition,
                            optimizer=optimizer_cls,
                            seed=inner_fold_seed,
                            max_trials=max_trials,
                            timeout=max_time,
                            display=display,
                            wait=wait_for_all_workers_to_finish,
                            n_workers=n_workers,
                            on_trial_exception=on_trial_exception
                        )
                        df = history.df()
                        safe_dataframe(df, working_dir, name, fold, method, pipeline_name)

                elif method.startswith("openfe"):
                    print("OpenFE Data")
                    option = method[-2:]
                    try:
                        int(option)
                    except ValueError as e:
                        option = method[-1:]
                    int(option)
                    pipeline_name = pipeline.name
                    train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=option)
                    file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                    file = working_dir / file_name
                    if rerun or not os.path.isfile(file):
                        train_x, test_x = get_openFE_features(train_x, train_y, test_x, 1)
                        evaluator = get_cv_evaluator(train_x, train_y, test_x, test_y, inner_fold_seed,
                                                     on_trial_exception, task_hint)
                        history = pipeline.optimize(
                            target=evaluator.fn,
                            metric=metric_definition,
                            optimizer=optimizer_cls,
                            seed=inner_fold_seed,
                            max_trials=max_trials,
                            timeout=max_time,
                            display=display,
                            wait=wait_for_all_workers_to_finish,
                            n_workers=n_workers,
                            on_trial_exception=on_trial_exception
                        )
                        df = history.df()
                        safe_dataframe(df, working_dir, name, fold, method, pipeline_name)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run feature engineering methods')
    parser.add_argument('--method', type=str, required=True, help='Feature engineering method to use')
    args = parser.parse_args()
    main(args)
