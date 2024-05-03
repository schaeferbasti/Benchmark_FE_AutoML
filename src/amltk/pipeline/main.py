from __future__ import annotations
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping, Iterable, Sequence
from typing import Any, Literal, overload

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from typing_extensions import override, Self

import numpy as np
import pandas as pd

from ConfigSpace import Categorical, Integer, ConfigurationSpace

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import get_scorer
from sklearn.preprocessing import *
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from amltk.pipeline import Choice, Component, Sequential, Split, request, Node
from amltk.sklearn import CVEvaluation
from amltk.optimization import Metric, Optimizer, Trial
from amltk.randomness import as_int, randuid
from amltk.store import PathBucket
from amltk.types import Seed

import openml

from openfe import OpenFE, transform

from ucimlrepo import fetch_ucirepo

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_fold(
        openml_task_id: int,
        fold: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | pd.Series,
    pd.DataFrame | pd.Series,
]:
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
    # black-friday dataset from AMLB (long execution time)
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


def get_openFE_features(train_x, test_x, train_y, n_jobs) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    openFE = OpenFE()
    features = openFE.fit(data=train_x, label=train_y, n_jobs=n_jobs)  # generate new features
    train_x, test_x = transform(train_x, test_x, features, n_jobs=n_jobs)
    return train_x, test_x


def get_sklearn_features(train_x, test_x, train_y, test_y) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    # ****** COMBINE TWO FEATURES (following the Expand & Reduce Strategy) ******
    # 1. Feature Generation
    #   a. Create Polynomial Features (PolynomialFeatures)
    #   b. Dimensionality Reduction (PCA, TruncatedSVD)
    #   c. Custom Feature Engineering (FunctionTransformer, TransformerMixin)
    # 2. Feature Selection
    #   a. SelectKBest
    #   b. SelectPercentile

    columns_train_x = train_x.columns
    columns_test_x = test_x.columns

    # Preprocessing
    # Numerize categorical columns
    cat_columns = train_x.select_dtypes(['category']).columns
    train_x[cat_columns] = train_x[cat_columns].apply(lambda x: pd.factorize(x, use_na_sentinel=True)[0])
    cat_columns = test_x.select_dtypes(['category']).columns
    test_x[cat_columns] = test_x[cat_columns].apply(lambda x: pd.factorize(x, use_na_sentinel=True)[0])

    # Replace NaN and negative values by mean
    imp1 = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp1.fit(train_x)
    train_x = imp1.transform(train_x)
    imp2 = SimpleImputer(missing_values=-1, strategy='mean')
    imp2.fit(train_x)
    train_x = imp2.transform(train_x)
    imp3 = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp3.fit(test_x)
    test_x = imp3.transform(test_x)
    imp4 = SimpleImputer(missing_values=-1, strategy='mean')
    imp4.fit(test_x)
    test_x = imp4.transform(test_x)

    # Generate Polynomial features
    # pf = PolynomialFeatures(degree=2, interaction_only=True)
    # train_x = pf.fit_transform(train_x)
    # test_x = pf.fit_transform(test_x)

    # Normalize
    train_x = normalize(train_x, axis=0)
    test_x = normalize(test_x, axis=0)

    # Binarize
    # train_x = binarize(train_x)
    # test_x = binarize(test_x)

    # Quantile Transformer
    qt = QuantileTransformer(random_state=0)
    # train_x = qt.fit_transform(train_x)
    # test_x = qt.fit_transform(test_x)

    # Transform to DataFrame again
    train_x = pd.DataFrame(train_x, columns=columns_train_x)
    test_x = pd.DataFrame(test_x, columns=columns_test_x)

    # Select Best Features
    # train_x = SelectKBest(chi2).fit_transform(train_x, train_y)
    # test_x = SelectKBest(chi2).fit_transform(test_x, test_y)

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

# works on dataset 2 (not for continuous data)
mlp_classifier = Component(
    item=MLPClassifier,
    space={
        "activation": ["identity", "logistic", "relu"],
        "alpha": (0.0001, 0.1),
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "epsilon": (1e-9, 1e-3),
        "momentum": (0.0, 1.0)
    },
    config={
        "random_state": request(
            "random_state",
            default=None,
        )
    }
)

mlp_pipeline = Sequential(preprocessing, mlp_classifier, name="mlp_pipeline")

# works on dataset 2 (not on continuous data)
svc_classifier = Component(
    item=SVC,
    config_transform=rf_config_transform,
    space={
        "C": (0.1, 10.0),
        "gamma": ["scale", "auto"],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
    },
    config={
        "class_weight": "balanced",
        "degree": 3,
        "probability": True,
        "random_state": request(
            "random_state",
            default=None,
        ),
    }
)
svc_pipeline = Sequential(preprocessing, svc_classifier, name="svc_pipeline")

# works on dataset 2 (not on continuous data)
knn_classifier = Component(
    item=KNeighborsClassifier,
    space={
        "n_neighbors": (2, 8),
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    },
    config={
        "leaf_size": 30,
        "metric": "minkowski",
        "n_jobs": 1,
    }
)
knn_pipeline = Sequential(preprocessing, knn_classifier, name="knn_pipeline")

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
    outer_fold_number = 0  # Only run the first outer fold, wrap this in a loop if needs be, with a unique history file for each one
    inner_fold_seed = random_seed + outer_fold_number

    # Evaluation of the original data
    X_original, X_test_original, y, y_test = get_dataset(option=2, openml_task_id=openml_task_id,
                                                         outer_fold_number=outer_fold_number)
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
        # Whether the post_processing callback requires models will require models, i.e.
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

    # Feature Engineering with OpenFE
    # X_openFE, X_test_openFE = get_openFE_features(X_original, X_test_original, y, 1)

    # Feature Engineering with sklearn
    """Works only with numerical data sets"""
    X_openFE, X_test_openFE = get_sklearn_features(X_original, X_test_original, y, y_test)

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
        working_dir="logs",
        # What to do when something goes wrong.
        on_error="raise" if on_trial_exception == "raise" else "fail",
        # Whether you want models to be store on disk under working_dir
        store_models=False,
        # A callback to be called at the end of each split
        post_split=do_something_after_a_split_was_evaluated,
        # Some callback that is called at the end of all fold evaluations
        post_processing=do_something_after_a_complete_trial_was_evaluated,
        # Whether the post_processing callback requires models will require models, i.e.
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
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)
    print(f"Saving dataframe of results to path: {results_to}")
    df.to_parquet(results_to)


class RandomSearch(Optimizer[None]):
    """An optimizer that uses ConfigSpace for random search."""

    def __init__(
            self,
            *,
            space: ConfigurationSpace,
            bucket: PathBucket | None = None,
            metrics: Metric | Sequence[Metric],
            seed: Seed | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            space: The search space to search over.
            bucket: The bucket given to trials generated by this optimizer.
            metrics: The metrics to optimize. Unused for RandomSearch.
            seed: The seed to use for the optimization.
        """
        metrics = metrics if isinstance(metrics, Sequence) else [metrics]
        super().__init__(metrics=metrics, bucket=bucket)
        seed = as_int(seed)
        space.seed(seed)
        self._counter = 0
        self.seed = seed
        self.space = space

    @override
    @classmethod
    def create(
            cls,
            *,
            space: ConfigurationSpace | Node,
            metrics: Metric | Sequence[Metric],
            bucket: PathBucket | str | Path | None = None,
            seed: Seed | None = None,
    ) -> Self:
        """Create a random search optimizer.

        Args:
            space: The node to optimize
            metrics: The metrics to optimize
            bucket: The bucket to store the results in
            seed: The seed to use for the optimization
        """
        seed = as_int(seed)
        match bucket:
            case None:
                bucket = PathBucket(
                    f"{cls.__name__}-{datetime.now().isoformat()}",
                )
            case str() | Path():
                bucket = PathBucket(bucket)
            case bucket:
                bucket = bucket  # noqa: PLW0127

        if isinstance(space, Node):
            space = space.search_space(parser=cls.preferred_parser())

        return cls(
            space=space,
            seed=seed,
            bucket=bucket,
            metrics=metrics,
        )

    @overload
    def ask(self, n: int) -> Iterable[Trial[None]]:
        ...

    @overload
    def ask(self, n: None = None) -> Trial[None]:
        ...

    @override
    def ask(
            self,
            n: int | None = None,
    ) -> Trial[None] | Iterable[Trial[None]]:
        """Ask the optimizer for a new config.

        Args:
            n: The number of configs to ask for. If `None`, ask for a single config.


        Returns:
            The trial info for the new config.
        """
        if n is None:
            configs = [self.space.sample_configuration()]
        else:
            configs = self.space.sample_configuration(n)

        trials: list[Trial[None]] = []
        for config in configs:
            self._counter += 1
            randuid_seed = self.seed + self._counter
            unique_name = f"trial-{randuid(4, seed=randuid_seed)}-{self._counter}"
            trial: Trial[None] = Trial.create(
                name=unique_name,
                config=dict(config),
                info=None,
                seed=self.seed,
                bucket=self.bucket / unique_name,
                metrics=self.metrics,
            )
            trials.append(trial)

        if n is None:
            return trials[0]

        return trials

    @override
    def tell(self, report: Trial.Report[None]) -> None:
        """Tell the optimizer about the result of a trial.

        Does nothing for random search.

        Args:
            report: The report of the trial.
        """

    @override
    @classmethod
    def preferred_parser(cls) -> Literal["configspace"]:
        """The preferred parser for this optimizer."""
        return "configspace"


if __name__ == "__main__":
    main()
