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
from src.datasets.Datasets import *
from src.amltk.evaluation.Evaluator import get_cv_evaluator
from src.amltk.optimizer.RandomSearch import RandomSearch

from src.feature_engineering.autofeat.Autofeat import get_autofeat_features
from src.feature_engineering.AutoGluon.AutoGluon import get_autogluon_features
from src.feature_engineering.BioAutoML.BioAutoML import get_bioautoml_features
from src.feature_engineering.Boruta.Boruta import get_boruta_features
from src.feature_engineering.CorrelationBasedFS.CorrelationBasedFS import get_correlationbased_features
from src.feature_engineering.Featurewiz.Featurewiz import get_featurewiz_features
from src.feature_engineering.H2O.H2O import get_h2o_features
from src.feature_engineering.MACFE.MACFE import get_macfe_features
from src.feature_engineering.MAFESE.MAFESE import get_mafese_features
from src.feature_engineering.MLJAR.MLJAR import get_mljar_features
from src.feature_engineering.OpenFE.OpenFE import get_openFE_features

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


lgbm_classifier = get_lgbm_classifier()
lgbm_classifier_pipeline = Sequential(preprocessing, lgbm_classifier, name="lgbm_classifier_pipeline")


def main(args):
    method_dataset = args.method_dataset

    rerun = True  # Decide if you want to re-execute the methods on a dataset or use the existing files
    debugging = False  # Decide if you want ot raise trial exceptions
    feat_eng_steps = 2  # Number of feature engineering steps for autofeat
    feat_sel_steps = 5  # Number of feature selection steps for autofeat
    num_features = 500  # Number of features for MLJAR
    estimations = 50  # Number of estimations for BioAutoML
    working_dir = Path("src/amltk/results/files")  # Path
    random_seed = 42  # Set seed
    folds = 10  # Set number of repetitions

    optimizer_cls = RandomSearch
    pipeline = lgbm_classifier_pipeline

    metric_definition = Metric(
        "roc_auc_ovo",
        minimize=False,
        bounds=(0, 1),
        fn=get_scorer("roc_auc_ovo")
    )

    if debugging:
        max_trials = 1  # don't care about quality of the found model
        max_time = 300  # 5 minutes
        n_workers = 4
        # Raise an error with traceback, something went wrong
        on_trial_exception = "raise"
        display = True
        wait_for_all_workers_to_finish = False
    else:
        max_trials = 100000
        max_time = 3600  # one hour
        n_workers = 4
        # Mark the trial as fail and move on to the next one
        on_trial_exception = "continue"
        display = True
        wait_for_all_workers_to_finish = False

    # Iterate over 10 folds (10 repetitions)
    for fold in range(folds):
        print(f"\n\n\n*******************************\n Fold {fold}\n*******************************\n")
        inner_fold_seed = random_seed + fold
        try:
            # Run AMLTK for Original Dataset
            if method_dataset.startswith("original"):
                print("Original Data")
                dataset = method_dataset[-2:]
                try:
                    int(dataset)
                except ValueError:
                    dataset = method_dataset[-1:]
                dataset = int(dataset)
                method = method_dataset.replace(dataset, "")
                pipeline_name = pipeline.name
                train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=dataset)
                file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                existing_file = "src/amltk/results/files/" + file_name
                if rerun or not os.path.isfile(existing_file):
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
                    if history.df() is None:
                        df = pd.DataFrame()
                    else:
                        df = history.df()
                    safe_dataframe(df, working_dir, name, fold, method, pipeline_name)
            # Do Feature Engineering with Autofeat and run AMLTK for feature-engineered Dataset
            elif method_dataset.startswith("autofeat"):
                print("autofeat Data")
                dataset = method_dataset[-2:]
                try:
                    int(dataset)
                except ValueError:
                    dataset = method_dataset[-1:]
                dataset = int(dataset)
                method = method_dataset.replace(dataset, "")
                pipeline_name = pipeline.name
                try:
                    train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=dataset)
                    print(name)
                    file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                    file = working_dir / file_name
                    if rerun or not os.path.isfile(file):
                        train_x, test_x = get_autofeat_features(train_x, train_y, test_x, task_hint, feat_eng_steps,
                                                                feat_sel_steps)
                        evaluator = get_cv_evaluator(train_x, train_y, test_x, test_y, inner_fold_seed,
                                                     on_trial_exception, task_hint)
                except:
                    train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=dataset)
                    print(name)
                    file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                    file = working_dir / file_name
                    if rerun or not os.path.isfile(file):
                        train_x, test_x = get_autofeat_features(train_x, train_y, test_x, task_hint, feat_eng_steps - 1,
                                                                feat_sel_steps)
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
                    if history.df() is None:
                        df = pd.DataFrame()
                    else:
                        df = history.df()
                    safe_dataframe(df, working_dir, name, fold, method, pipeline_name)
            # Do Feature Engineering with Autogluon and run AMLTK for feature-engineered Dataset
            elif method_dataset.startswith("autogluon"):
                print("autogluon Data")
                dataset = method_dataset[-2:]
                try:
                    int(dataset)
                except ValueError:
                    dataset = method_dataset[-1:]
                dataset = int(dataset)
                method = method_dataset.replace(dataset, "")
                pipeline_name = pipeline.name
                train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=dataset)
                print(name)
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
                    if history.df() is None:
                        df = pd.DataFrame()
                    else:
                        df = history.df()
                    safe_dataframe(df, working_dir, name, fold, method, pipeline_name)
            # Do Feature Engineering with BioAutoML and run AMLTK for feature-engineered Dataset
            elif method_dataset.startswith("bioautoml"):
                print("BioAutoML Data")
                dataset = method_dataset[-2:]
                try:
                    int(dataset)
                except ValueError:
                    dataset = method_dataset[-1:]
                dataset = int(dataset)
                method = method_dataset.replace(dataset, "")
                pipeline_name = pipeline.name
                train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=dataset)
                print(name)
                file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                file = working_dir / file_name
                if rerun or not os.path.isfile(file):
                    continuous = True
                    train_x, test_x = get_bioautoml_features(train_x, train_y, test_x, estimations, continuous)
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
                    if history.df() is None:
                        df = pd.DataFrame()
                    else:
                        df = history.df()
                    safe_dataframe(df, working_dir, name, fold, method, pipeline_name)
            # Do Feature Engineering with Boruta and run AMLTK for feature-engineered Dataset
            elif method_dataset.startswith("boruta"):
                print("boruta Data")
                dataset = method_dataset[-2:]
                try:
                    int(dataset)
                except ValueError:
                    dataset = method_dataset[-1:]
                dataset = int(dataset)
                method = method_dataset.replace(dataset, "")
                pipeline_name = pipeline.name
                train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=dataset)
                print(name)
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
                    if history.df() is None:
                        df = pd.DataFrame()
                    else:
                        df = history.df()
                    safe_dataframe(df, working_dir, name, fold, method, pipeline_name)
            # Do Feature Engineering with CorrelationBasedFS and run AMLTK for feature-engineered Dataset
            elif method_dataset.startswith("cfs"):
                print("CorrelationBasedFS Data")
                dataset = method_dataset[-2:]
                try:
                    int(dataset)
                except ValueError:
                    dataset = method_dataset[-1:]
                dataset = int(dataset)
                method = method_dataset.replace(dataset, "")
                pipeline_name = pipeline.name
                train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=dataset)
                print(name)
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
                    if history.df() is None:
                        df = pd.DataFrame()
                    else:
                        df = history.df()
                    safe_dataframe(df, working_dir, name, fold, method, pipeline_name)
            # Do Feature Engineering with Featurewiz and run AMLTK for feature-engineered Dataset
            elif method_dataset.startswith("featurewiz"):
                print("Featurewiz Data")
                dataset = method_dataset[-2:]
                try:
                    int(dataset)
                except ValueError:
                    dataset = method_dataset[-1:]
                dataset = int(dataset)
                method = method_dataset.replace(dataset, "")
                pipeline_name = pipeline.name
                train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=dataset)
                print(name)
                file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                file = working_dir / file_name
                if rerun or not os.path.isfile(file):
                    train_x, test_x = get_featurewiz_features(train_x, train_y, test_x)
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
                    if history.df() is None:
                        df = pd.DataFrame()
                    else:
                        df = history.df()
                    safe_dataframe(df, working_dir, name, fold, method, pipeline_name)
            # Do Feature Engineering with H2O and run AMLTK for feature-engineered Dataset
            elif method_dataset.startswith("h2o"):
                print("H2O Data")
                dataset = method_dataset[-2:]
                try:
                    int(dataset)
                except ValueError:
                    dataset = method_dataset[-1:]
                dataset = int(dataset)
                method = method_dataset.replace(dataset, "")
                pipeline_name = pipeline.name
                train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=dataset)
                print(name)
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
                    if history.df() is None:
                        df = pd.DataFrame()
                    else:
                        df = history.df()
                    safe_dataframe(df, working_dir, name, fold, method, pipeline_name)
            # Do Feature Engineering with MACFE and run AMLTK for feature-engineered Dataset
            elif method_dataset.startswith("macfe"):
                print("MACFE Data")
                dataset = method_dataset[-2:]
                try:
                    int(dataset)
                except ValueError:
                    dataset = method_dataset[-1:]
                dataset = int(dataset)
                method = method_dataset.replace(dataset, "")
                pipeline_name = pipeline.name
                train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=dataset)
                print(name)
                file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                file = working_dir / file_name
                if rerun or not os.path.isfile(file):
                    train_x, test_x = get_macfe_features(train_x, train_y, test_x, test_y, name)
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
                    if history.df() is None:
                        df = pd.DataFrame()
                    else:
                        df = history.df()
                    safe_dataframe(df, working_dir, name, fold, method, pipeline_name)
            # Do Feature Engineering with MAFESE and run AMLTK for feature-engineered Dataset
            elif method_dataset.startswith("mafese"):
                print("MAFESE Data")
                dataset = method_dataset[-2:]
                try:
                    int(dataset)
                except ValueError:
                    dataset = method_dataset[-1:]
                dataset = int(dataset)
                method = method_dataset.replace(dataset, "")
                pipeline_name = pipeline.name
                train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=dataset)
                print(name)
                file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                file = working_dir / file_name
                if rerun or not os.path.isfile(file):
                    train_x, test_x = get_mafese_features(train_x, train_y, test_x, test_y, name, num_features)
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
                    if history.df() is None:
                        df = pd.DataFrame()
                    else:
                        df = history.df()
                    safe_dataframe(df, working_dir, name, fold, method, pipeline_name)
            # Do Feature Engineering with MLJAR and run AMLTK for feature-engineered Dataset
            elif method_dataset.startswith("mljar"):
                print("MLJAR Data")
                dataset = method_dataset[-2:]
                try:
                    int(dataset)
                except ValueError:
                    dataset = method_dataset[-1:]
                dataset = int(dataset)
                method = method_dataset.replace(dataset, "")
                pipeline_name = pipeline.name
                train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=dataset)
                print(name)
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
                    if history.df() is None:
                        df = pd.DataFrame()
                    else:
                        df = history.df()
                    safe_dataframe(df, working_dir, name, fold, method, pipeline_name)
            # Do Feature Engineering with OpenFE and run AMLTK for feature-engineered Dataset
            elif method_dataset.startswith("openfe"):
                print("OpenFE Data")
                dataset = method_dataset[-2:]
                try:
                    int(dataset)
                except ValueError:
                    dataset = method_dataset[-1:]
                dataset = int(dataset)
                method = method_dataset.replace(dataset, "")
                pipeline_name = pipeline.name
                train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=dataset)
                print(name)
                file_name = f"results_{name}_{method}_{pipeline_name}_{fold}.parquet"
                file = working_dir / file_name
                if rerun or not os.path.isfile(file):
                    train_x, test_x = get_openFE_features(train_x, train_y, test_x, 1, name)
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
                    if history.df() is None:
                        df = pd.DataFrame()
                    else:
                        df = history.df()
                    safe_dataframe(df, working_dir, name, fold, method, pipeline_name)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run feature engineering methods')
    parser.add_argument('--method', type=str, required=True, help='Feature engineering method to use')
    args = parser.parse_args()
    main(args)
