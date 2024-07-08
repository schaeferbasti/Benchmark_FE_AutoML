import argparse
import pandas as pd
import os
from pathlib import Path
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
from src.amltk.feature_engineering.FETCH.FETCH import get_xxx_features
from src.amltk.feature_engineering.H2O.H2O import get_h2o_features
from src.amltk.feature_engineering.OpenFE.OpenFE import get_openFE_features


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

lgbm_classifier = get_lgbm_classifier()
lgbm_classifier_pipeline = Sequential(preprocessing, lgbm_classifier, name="lgbm_classifier_pipeline")

def safe_dataframe(df, working_dir, dataset_name, fold_number, method_name):
    file_string = f"results_{dataset_name}_{method_name}_fold_{fold_number}.parquet"
    results_to = working_dir / file_string
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)
    print(f"Saving dataframe of results to path: {results_to}")
    df.to_parquet(results_to)


def main():
    parser = argparse.ArgumentParser(description='Run feature engineering methods')
    parser.add_argument('--method', type=str, required=True, help='Feature engineering method to use')
    args = parser.parse_args()

    method = args.method

    rerun = True
    debugging = True
    feat_eng_steps = 2
    feat_sel_steps = 5
    working_dir = Path("results")
    random_seed = 42
    folds = 1
    test_new_method_datasets = [18]

    optimizer_cls = RandomSearch
    pipeline = lgbm_classifier_pipeline

    metric_definition = Metric(
        "roc_auc_ovo",
        minimize=False,
        bounds=(0, 1),
        fn=get_scorer("roc_auc_ovo")
    )

    if debugging:
        max_trials = 1
        max_time = 600
        n_workers = 20
        on_trial_exception = "raise"
        display = True
        wait_for_all_workers_to_finish = False
    else:
        max_trials = 50
        max_time = 3600
        n_workers = 50
        on_trial_exception = "continue"
        display = True
        wait_for_all_workers_to_finish = False

    for fold in range(folds):
        print(f"\n\n\n*******************************\n Fold {fold}\n*******************************\n")
        inner_fold_seed = random_seed + fold
        for option in test_new_method_datasets:
            train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=option)

            if method == "original":
                print("Original Data")
                file_name = f"results_{name}_original_fold_{fold}.parquet"
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
                    df = history.df()
                    safe_dataframe(df, working_dir, name, fold, "original")

            elif method == "autofeat":
                print("autofeat Data")
                file_name = f"results_{name}_autofeat_fold_{fold}.parquet"
                file = working_dir / file_name
                if rerun or not os.path.isfile(file):
                    train_x_autofeat, test_x_autofeat = get_autofeat_features(train_x, train_y, test_x, task_hint,
                                                                              feat_eng_steps, feat_sel_steps)
                    evaluator = get_cv_evaluator(train_x_autofeat, train_y, test_x_autofeat, test_y, inner_fold_seed,
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
                    safe_dataframe(df, working_dir, name, fold, "autofeat")
            """
            elif method == "autogluon":
                print("autogluon Data")
                file_name = f"results_{name}_autogluon_fold_{fold}.parquet"
                file = working_dir / file_name
                if rerun or not os.path.isfile(file):
                    train_x_autogluon, test_x_autogluon = get_autogluon_features(train_x, train_y, test_x)
                    evaluator = get_cv_evaluator(train_x_autogluon, train_y, test_x_autogluon, test_y,
                                                 inner_fold_seed,
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
                    safe_dataframe(df, working_dir, name, fold, "autogluon")
            """
