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


def safe_dataframe(df, working_dir, dataset_name, fold_number, method_name):
    file_string = f"results_{dataset_name}_{method_name}_{fold_number}.parquet"
    results_to = working_dir / file_string
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)
    print(f"Saving dataframe of results to path: {results_to}")
    df.to_parquet(results_to)


lgbm_classifier = get_lgbm_classifier()
lgbm_classifier_pipeline = Sequential(preprocessing, lgbm_classifier, name="lgbm_classifier_pipeline")


def main(args):
    method = args.method
    dataset = args.dataset
    fold = args.fold

    rerun = True  # Decide if you want to re-execute the methods on a dataset or use the existing files
    debugging = False  # Decide if you want ot raise trial exceptions
    feat_eng_steps = 2  # Number of feature engineering steps for autofeat
    feat_sel_steps = 5  # Number of feature selection steps for autofeat
    n_jobs = 1  # Number of jobs for OpenFE
    num_features = 500  # Number of features for MLJAR
    working_dir = Path("src/amltk/results")  # Path if running on Cluster
    # working_dir = Path("results")  # Path for local execution
    random_seed = 42  # Set seed
    folds = 10  # Set number of folds (normal 10, test 1)

    optimizer_cls = RandomSearch
    pipeline = lgbm_classifier_pipeline

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

    print(f"\n\n\n*******************************\n Fold {fold}\n*******************************\n")
    inner_fold_seed = random_seed + fold
    train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=dataset)

    if method == "original":
        print("Original Data")
        file_name = f"results_{name}_{method}_{fold}.parquet"
        file = working_dir / file_name
        if rerun or not os.path.isfile(file):
            evaluator = get_cv_evaluator(train_x, train_y, test_x, test_y, inner_fold_seed, on_trial_exception,
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
            safe_dataframe(history, working_dir, name, fold, method)
            print(f"Finished fold {fold} for {name} with method {method}")
        else:
            print(f"Results for fold {fold} for {name} with method {method} already exist. Skipping execution.")

    else:
        feat_methods = {
            "autofeat": get_autofeat_features,
            "autogluon": get_autogluon_features,
            "bioautoml": get_bioautoml_features,
            "boruta": get_boruta_features,
            "correlationBasedFS": get_correlationbased_features,
            "featuretools": get_featuretools_features,
            "h2o": get_h2o_features,
            "mljar": get_mljar_features,
            "openfe": get_openFE_features,
        }

        if method not in feat_methods:
            raise ValueError(f"Unknown method: {method}")

        print(f"Using feature engineering method: {method}")
        feature_eng_func = feat_methods[method]
        file_name = f"results_{name}_{method}_{fold}.parquet"
        file = working_dir / file_name

        if rerun or not os.path.isfile(file):
            # Apply feature engineering method
            train_x_feat, test_x_feat = feature_eng_func(train_x, train_y, test_x, test_y, inner_fold_seed, feat_eng_steps, feat_sel_steps, n_jobs, num_features)
            evaluator = get_cv_evaluator(train_x_feat, train_y, test_x_feat, test_y, inner_fold_seed, on_trial_exception, task_hint)
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
            safe_dataframe(df, working_dir, name, fold, method)
            print(f"Finished fold {fold} for {name} with method {method}")
        else:
            print(f"Results for fold {fold} for {name} with method {method} already exist. Skipping execution.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automated ML pipeline with specified method, dataset, and fold.")
    parser.add_argument("--method", type=str, required=True, help="Feature engineering method to use.")
    parser.add_argument("--dataset", type=int, required=True, help="Dataset to use.")
    parser.add_argument("--fold", type=int, required=True, help="Fold number to use.")
    args = parser.parse_args()
    main(args)
