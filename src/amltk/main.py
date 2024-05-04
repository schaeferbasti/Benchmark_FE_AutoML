from __future__ import annotations

import warnings
from pathlib import Path

from amltk.optimization import Metric, Trial
from amltk.pipeline import Choice, Sequential, Split, Node
from amltk.sklearn import CVEvaluation
from sklearn.impute import SimpleImputer
from sklearn.metrics import get_scorer
from sklearn.preprocessing import *

from src.amltk.classifiers.get_classifiers import *
from src.amltk.datasets.get_datasets import *
from src.amltk.evaluation.get_evaluator import get_cv_evaluator
from src.amltk.feature_engineering.open_fe import get_openFE_features
from src.amltk.feature_engineering.own_method import get_sklearn_features
from src.amltk.optimizer.random_search import RandomSearch

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_dataset(option, openml_task_id, outer_fold_number) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | pd.Series,
    pd.DataFrame | pd.Series,
]:
    # california-housing dataset from OpenFE example
    if option == 1:
        test_x, test_y, train_x, train_y = get_california_housing_dataset()
        return train_x, test_x, train_y, test_y
    # cylinder-bands dataset from OpenFE benchmark
    elif option == 2:
        return get_cylinder_folds_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
    # balance-scale dataset from OpenFE benchmark (not working)
    elif option == 3:
        test_X, test_y, train_X, train_y = get_balance_scale_dataset()
        return train_X, test_X, train_y, test_y
    # black-friday dataset from AMLB (long execution time)
    elif option == 4:
        test_X, test_y, train_X, train_y = get_black_friday_dataset()
        return train_X, test_X, train_y, test_y


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
    outer_fold_number = 0  # Only run the first outer fold, wrap this in a loop if needs be, with a unique history file for each one
    inner_fold_seed = random_seed + outer_fold_number

    pipeline = rf_pipeline

    """
    ############## Original Data ##############
    Use original data without feature engineering

    """
    # Get original data
    X_original, X_test_original, y, y_test = get_dataset(option=2, openml_task_id=openml_task_id,
                                                         outer_fold_number=outer_fold_number)

    evaluator = get_cv_evaluator(X_original, X_test_original, inner_fold_seed, on_trial_exception, task_hint, y, y_test)

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

    """
    ############## Feature Engineering with OpenFE ##############
    Use Feature Generation and Selection implemented by the OpenFE paper

    """
    X_openFE, X_test_openFE = get_openFE_features(X_original, X_test_original, y, 1)

    evaluator = get_cv_evaluator(X_openFE, X_test_openFE, inner_fold_seed, on_trial_exception, task_hint, y, y_test)

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

    """
    ############## Feature Engineering with sklearn ##############
    Use self-implemented Feature Generation and Selection with the usage of the sklearn library

    """
    X_sklearn, X_test_sklearn = get_sklearn_features(X_original, X_test_original, y, y_test)

    evaluator = get_cv_evaluator(X_sklearn, X_test_sklearn, inner_fold_seed, on_trial_exception, task_hint, y, y_test)

    history_sklearn = pipeline.optimize(
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

    # Append Dataframes to one + print and save it to parquet
    df_original = history_original.df()
    df_openFE = history_openFE.df()
    df_sklearn = history_sklearn.df()

    df = pd.concat([df_original, df_openFE, df_sklearn], axis=0)
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
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)
    print(f"Saving dataframe of results to path: {results_to}")
    df.to_parquet(results_to)


if __name__ == "__main__":
    main()
