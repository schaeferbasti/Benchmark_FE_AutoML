from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd
from ConfigSpace import Categorical, Integer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import get_scorer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from lib.random_search import RandomSearch
from amltk.optimization.trial import Metric, Trial
from amltk.pipeline import Choice, Component, Node, Sequential, Split, request
from amltk.sklearn import CVEvaluation

from openfe import OpenFE, transform
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from ucimlrepo import fetch_ucirepo


def get_fold(
        openml_task_id: int,
        fold: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | pd.Series,
    pd.DataFrame | pd.Series,
]:
    """Get the data for a specific fold of an OpenML task.

    Args:
        openml_task_id: The OpenML task id.
        fold: The fold number.
        n_splits: The number of splits that will be applied. This is used
            to resample training data such that enough at least instance for each class is present for
            every stratified split.
        seed: The random seed to use for reproducibility of resampling if necessary.
    """
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    train_idx, test_idx = task.get_train_test_split_indices(fold=fold)
    X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    return X_train, X_test, y_train, y_test


def get_dataset(option, openml_task_id, outer_fold_number) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | pd.Series,
    pd.DataFrame | pd.Series,
]:
    """Get the data from OpenFE.

    Args:
        openml_task_id: The OpenML task id.
        fold: The fold number.
        n_splits: The number of splits that will be applied. This is used
            to resample training data such that enough at least instance for each class is present for
            every stratified split.
        seed: The random seed to use for reproducibility of resampling if necessary.
    """
    # california-housing dataset from OpenFE example
    if option == 1:
        data = fetch_california_housing(as_frame=True).frame
        label = data[['MedHouseVal']]
        train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=1)
        return train_x, test_x, train_y, test_y
    # cylinder-bands dataset from OpenFE benchmark
    elif option == 2:
        return get_fold(openml_task_id=openml_task_id, fold=outer_fold_number)
    # balance-scale dataset from OpenFE benchmark (not working)
    elif option == 3:
        balance_scale = fetch_ucirepo(id=12)
        X = balance_scale.data.features
        y = balance_scale.data.targets
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=20)
        return train_X, test_X, train_y, test_y
    # black-friday (long execution time!!)
    elif option == 4:
        train = pd.read_csv(r'datasets/black-friday/train.csv', delimiter=',', header=None, skiprows=1,
                            names=['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
                                   'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
                                   'Product_Category_2', 'Product_Category_3', 'Purchase'])
        train_y = train[['Purchase']]
        train_X = train.drop(['Purchase'], axis=1)
        test = pd.read_csv(r'datasets/black-friday/test.csv', delimiter=',', header=None, skiprows=1,
                            names=['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
                                   'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
                                   'Product_Category_2', 'Product_Category_3', 'Purchase'])
        test_y = test[['Purchase']]
        test_X = test.drop(['Purchase'], axis=1)
        return train_X, test_X, train_y, test_y


def get_openFE_features(train_x, test_x, train_y, n_jobs):
    openFE = OpenFE()
    features = openFE.fit(data=train_x, label=train_y, n_jobs=n_jobs)  # generate new features
    train_x, test_x = transform(train_x, test_x, features, n_jobs=n_jobs)
    return train_x, test_x


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


def rf_config_transform(config: Mapping[str, Any], _: Any) -> dict[str, Any]:
    new_config = dict(config)
    if new_config["class_weight"] == "None":
        new_config["class_weight"] = None
    return new_config


# NOTE: This space should not be used for evaluating how good this RF is
# vs other algorithms
rf_classifier = Component(
    item=RandomForestClassifier,
    config_transform=rf_config_transform,
    space={
        "criterion": ["gini", "entropy"],
        "max_features": Categorical(
            "max_features",
            list(np.logspace(0.1, 1, base=10, num=10) / 10),
            ordered=True,
        ),
        "min_samples_split": Integer("min_samples_split", bounds=(2, 20), default=2),
        "min_samples_leaf": Integer("min_samples_leaf", bounds=(1, 20), default=1),
        "bootstrap": Categorical("bootstrap", [True, False], default=True),
        "class_weight": ["balanced", "balanced_subsample", "None"],
        "min_impurity_decrease": (1e-9, 1e-1),
    },
    config={
        "random_state": request(
            "random_state",
            default=None,
        ),  # Will be provided later by the `Trial`
        "n_estimators": 512,
        "max_depth": None,
        "min_weight_fraction_leaf": 0.0,
        "max_leaf_nodes": None,
        "warm_start": False,  # False due to no iterative fit used here
        "n_jobs": 1,
    },
)

rf_pipeline = Sequential(preprocessing, rf_classifier, name="rf_pipeline")


def do_something_after_a_split_was_evaluated(
        trial: Trial,
        fold: int,
        info: CVEvaluation.PostSplitInfo,
) -> CVEvaluation.PostSplitInfo:
    return info


def do_something_after_a_complete_trial_was_evaluated(
        report: Trial.Report,
        pipeline: Node,
        info: CVEvaluation.CompleteEvalInfo,
) -> Trial.Report:
    return report


def main() -> None:
    optimizer_cls = RandomSearch
    working_dir = Path("results").absolute()
    results_to = working_dir / "results.parquet"
    metric_definition = Metric(
        "accuracy",
        minimize=False,
        bounds=(0, 1),
        fn=get_scorer("accuracy"),
    )

    per_process_memory_limit = None  # (4, "GB")  # NOTE: May have issues on Mac
    per_process_walltime_limit = None  # (60, "s")

    debugging = False
    if debugging:
        max_trials = 1
        max_time = 30
        n_workers = 1
        # raise an error with traceback, something went wrong
        on_trial_exception = "raise"
        display = True
        wait_for_all_workers_to_finish = True
    else:
        max_trials = 10
        max_time = 300
        n_workers = 4
        # Just mark the trial as fail and move on to the next one
        on_trial_exception = "continue"
        display = True
        wait_for_all_workers_to_finish = False

    random_seed = 42
    openml_task_id = 1797
    task_hint = "classification"
    outer_fold_number = 0  # Only run the first outer fold, wrap this in a loop if needs be, with a unique history file for each one)
    inner_fold_seed = random_seed + outer_fold_number

    # Evaluation of the original data
    X_original, X_test_original, y, y_test = get_dataset(2, openml_task_id, outer_fold_number)
    evaluator = CVEvaluation(
        # Provide data, number of times to split, cross-validation and a hint of the task type
        X_original,
        y,
        splitter="cv",
        n_splits=8,
        task_hint=task_hint,
        # Seeding for reproducibility
        random_state=inner_fold_seed,
        # Provide test data to get test scores
        X_test=X_test_original,
        y_test=y_test,
        # Record training scores
        train_score=True,
        # Where to store things
        working_dir="logs/log.txt",
        # What to do when something goes wrong.
        on_error="raise" if on_trial_exception == "raise" else "fail",
        # Whether you want models to be store on disk under working_dir
        store_models=False,
        # A callback to be called at the end of each split
        post_split=do_something_after_a_split_was_evaluated,
        # Some callback that is called at the end of all fold evaluations
        post_processing=do_something_after_a_complete_trial_was_evaluated,
        # Whether the post_processing callback requires models will required models, i.e.
        # to compute some bagged average over all fold models. If `False` will discard models eagerly
        # to save space.
        post_processing_requires_models=False,
        # This handles edge cases related to stratified splitting when there are too
        # few instances of a specific class. May wish to disable if your passing extra fit params
        #rebalance_if_required_for_stratified_splitting=True,
        # Extra parameters requested by sklearn models/group splitters or metrics,
        # such as `sample_weight`
        params=None,
    )

    # Here we just use the `optimize` method to set up and run an optimization loop
    # with `n_workers`. Please either look at the source code for `optimize` or
    # refer to the `Scheduler` and `Optimizer` guide if you need more fine-grained control.
    # If you need to evaluate a certain configuration, you can create your own `Trial` object.

    # trial = Trial.create(name=...., info=None, config=..., bucket=..., seed=..., metrics=metric_def)
    # report = evaluator.evaluate(trial, rf_pipeline)
    # print(report)

    history_original = rf_pipeline.optimize(
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

    # Evaluation of the feature engineered data from OpenFE
    X_openFE, X_test_openFE = get_openFE_features(X_original, X_test_original, y, 1)
    evaluator = CVEvaluation(
        # Provide data, number of times to split, cross-validation and a hint of the task type
        X_openFE,
        y,
        splitter="cv",
        n_splits=8,
        task_hint=task_hint,
        # Seeding for reproducibility
        random_state=inner_fold_seed,
        # Provide test data to get test scores
        X_test=X_test_openFE,
        y_test=y_test,
        # Record training scores
        train_score=True,
        # Where to store things
        working_dir="logs/log.txt",
        # What to do when something goes wrong.
        on_error="raise" if on_trial_exception == "raise" else "fail",
        # Whether you want models to be store on disk under working_dir
        store_models=False,
        # A callback to be called at the end of each split
        post_split=do_something_after_a_split_was_evaluated,
        # Some callback that is called at the end of all fold evaluations
        post_processing=do_something_after_a_complete_trial_was_evaluated,
        # Whether the post_processing callback requires models will required models, i.e.
        # to compute some bagged average over all fold models. If `False` will discard models eagerly
        # to save space.
        post_processing_requires_models=False,
        # This handles edge cases related to stratified splitting when there are too
        # few instances of a specific class. May wish to disable if your passing extra fit params
        # rebalance_if_required_for_stratified_splitting=True,
        # Extra parameters requested by sklearn models/group splitters or metrics,
        # such as `sample_weight`
        params=None,
    )

    # Here we just use the `optimize` method to set up and run an optimization loop
    # with `n_workers`. Please either look at the source code for `optimize` or
    # refer to the `Scheduler` and `Optimizer` guide if you need more fine-grained control.
    # If you need to evaluate a certain configuration, you can create your own `Trial` object.

    # trial = Trial.create(name=...., info=None, config=..., bucket=..., seed=..., metrics=metric_def)
    # report = evaluator.evaluate(trial, rf_pipeline)
    # print(report)

    history_openFE = rf_pipeline.optimize(
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
    df_openFE = history_openFE.df()

    df = pd.concat([df_original, df_openFE], axis=0)
    # Assign some new information to the dataframe
    df.assign(
        outer_fold=outer_fold_number,
        inner_fold_seed=inner_fold_seed,
        task_id=openml_task_id,
        max_trials=max_trials,
        max_time=max_time,
        optimizer=optimizer_cls.__name__,
        n_workers=n_workers,
    )
    print(df)
    print(f"Saving dataframe of results to path: {results_to}")
    df.to_parquet(results_to)


if __name__ == "__main__":
    main()