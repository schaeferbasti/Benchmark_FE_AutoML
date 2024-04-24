from typing import Any

import openml
from amltk import Sequential, Split, Component, Trial, Node, PathBucket, Scheduler, Metric, History
from amltk.optimization.optimizers.smac import SMACOptimizer
from openfe import OpenFE, transform
from amltk.sklearn import split_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import mean_squared_error, accuracy_score

import lightgbm as lgb

import pandas as pd

def get_openfe_features(train_x, test_x, train_y, n_jobs):
    openFE = OpenFE()
    features = openFE.fit(data=train_x, label=train_y, n_jobs=n_jobs)  # generate new features
    train_x, test_x = transform(train_x, test_x, features, n_jobs=n_jobs)
    return train_x, test_x


def calc_openfe_score(train_x, test_x, train_y, test_y, seed, n_jobs):
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'seed': seed}
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(train_x, train_y, eval_set=[(test_x, test_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = pd.DataFrame(gbm.predict(test_x), index=test_x.index)
    score = mean_squared_error(test_y, pred)
    return score

def get_data(seed):
    # OpenFE with OpenML Dataset
    dataset = openml.datasets.get_dataset(31, download_data=True, download_features_meta_data=False, download_qualities=False)
    target_name = dataset.default_target_attribute
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_name)
    _y = LabelEncoder().fit_transform(y)
    data = split_data(X, _y, splits={"train": 0.6, "val": 0.2, "test": 0.2}, seed=seed)  # type: ignore
    X_train, y_train = data["train"]
    X_val, y_val = data["test"]
    X_test, y_test = data["test"]
    return X_train, y_train, X_val, y_val, X_test, y_test


def target_function(
    train_x, val_x, test_x, train_y, val_y, test_y,
    trial: Trial,
    _pipeline: Node,
    data_bucket: PathBucket
) -> Trial.Report:
    trial.store({"config.json": trial.config})
    # Load in data
    """
    with trial.profile("data-loading"):
        X_train, X_val, X_test, y_train, y_val, y_test = (
            data_bucket["X_train.csv"].load(),
            data_bucket["X_val.csv"].load(),
            data_bucket["X_test.csv"].load(),
            data_bucket["y_train.npy"].load(),
            data_bucket["y_val.npy"].load(),
            data_bucket["y_test.npy"].load(),
        )
    """
    X_train, X_val, X_test, y_train, y_val, y_test = train_x, val_x, test_x, train_y, val_y, test_y

    # Configure the pipeline with the trial config before building it.
    sklearn_pipeline = _pipeline.configure(trial.config).build("sklearn")

    # Fit the pipeline, indicating when you want to start the trial timing
    try:
        with trial.profile("fit"):
            sklearn_pipeline.fit(X_train, y_train)
    except Exception as e:
        return trial.fail(e)

    # Make our predictions with the model
    with trial.profile("predictions"):
        train_predictions = sklearn_pipeline.predict(X_train)
        val_predictions = sklearn_pipeline.predict(X_val)
        test_predictions = sklearn_pipeline.predict(X_test)

    with trial.profile("probabilities"):
        val_probabilites = sklearn_pipeline.predict_proba(X_val)

    # Save the scores to the summary of the trial
    with trial.profile("scoring"):
        train_acc = float(accuracy_score(train_predictions, y_train))
        val_acc = float(accuracy_score(val_predictions, y_val))
        test_acc = float(accuracy_score(test_predictions, y_test))

    trial.summary["train/acc"] = train_acc
    trial.summary["val/acc"] = val_acc
    trial.summary["test/acc"] = test_acc

    # Save all of this to the file system
    trial.store(
        {
            "model.pkl": sklearn_pipeline,
            "val_probabilities.npy": val_probabilites,
            "val_predictions.npy": val_predictions,
            "test_predictions.npy": test_predictions,
        },
    )

    # Finally report the success
    return trial.success(accuracy=val_acc)


if __name__ == "__main__":
    seed = 42
    train_x, test_x, val_x, val_y, train_y, test_y = get_data(seed)

    # OpenFE
    """
    n_jobs = 4
    train_x, test_x = get_openfe_features(train_x, test_x, train_y, n_jobs)
    print(train_x.head(5))
    score = calc_openfe_score(train_x, test_x, train_y, test_y, seed, n_jobs)
    print("Score: " + str(score))
    """

    # AutoML-Toolkit
    pipeline = (
            Sequential(name="Pipeline")
            >> Split(
        {
            "categorical": [
                SimpleImputer(strategy="constant", fill_value="missing"),
                OneHotEncoder(drop="first"),
            ],
            "numerical": Component(
                SimpleImputer,
                space={"strategy": ["mean", "median"]},
            ),
        },
        name="feature_preprocessing",
    )
            >> Component(
        RandomForestClassifier,
        space={
            "n_estimators": (10, 100),
            "max_features": (0.0, 1.0),
            "criterion": ["gini", "entropy", "log_loss"],
        },
    )
    )

    print(pipeline)
    print(pipeline.search_space("configspace"))

    bucket = PathBucket("example-hpo", clean=True, create=True)
    data_bucket = bucket / "data"
    data_bucket.store(
        {
            "X_train.csv": train_x,
            "X_val.csv": val_x,
            "X_test.csv": test_x,
            "y_train.npy": train_y,
            "y_val.npy": val_y,
            "y_test.npy": test_y,
        },
    )

    print(bucket)
    print(dict(bucket))
    print(dict(data_bucket))

    scheduler = Scheduler.with_processes(2)
    optimizer = SMACOptimizer.create(
        space=pipeline,  # <!> (1)!
        metrics=Metric("accuracy", minimize=False, bounds=(0.0, 1.0)),
        bucket=bucket,
        seed=seed,
    )

    task = scheduler.task(target_function)
    print(task)


    @scheduler.on_start
    def launch_initial_tasks() -> None:
        """When we start, launch `n_workers` tasks."""
        trial = optimizer.ask()
        task.submit(trial, _pipeline=pipeline, data_bucket=data_bucket)


    @task.on_result
    def tell_optimizer(_, report: Trial.Report) -> None:
        """When we get a report, tell the optimizer."""
        optimizer.tell(report)


    trial_history = History()

    @task.on_result
    def add_to_history(_, report: Trial.Report) -> None:
        """When we get a report, print it."""
        trial_history.add(report)


    @task.on_result
    def launch_another_task(*_: Any) -> None:
        """When we get a report, evaluate another trial."""
        if scheduler.running():
            trial = optimizer.ask()
            task.submit(trial, _pipeline=pipeline, data_bucket=data_bucket)


    @task.on_exception
    def stop_scheduler_on_exception(*_: Any) -> None:
        scheduler.stop()


    @task.on_cancelled
    def stop_scheduler_on_cancelled(_: Any) -> None:
        scheduler.stop()


    scheduler.run(timeout=5)

    print("Trial history:")
    history_df = trial_history.df()
    print(history_df)