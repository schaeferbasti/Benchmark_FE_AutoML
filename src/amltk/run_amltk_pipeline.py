from __future__ import annotations

import warnings
from pathlib import Path
import os.path

from amltk.optimization import Metric
from amltk.pipeline import Choice, Sequential, Split
from sklearn.metrics import get_scorer
from sklearn.preprocessing import *

from src.amltk.classifiers.Classifiers import *
from src.datasets.Datasets import *
from src.amltk.evaluation.Evaluator import get_cv_evaluator
from src.amltk.optimizer.RandomSearch import RandomSearch

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


def safe_dataframe(df, working_dir, dataset_name, fold_number, method_name):
    file_string = "results_" + str(dataset_name) + "_" + str(method_name) + "_" + str(fold_number) + ".parquet"
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
    rerun = False        # Decide if you want to re-execute the methods on a dataset or use the existing files
    debugging = False    # Decide if you want ot raise trial exceptions

    feat_eng_steps = 2  # Number of feature engineering steps for autofeat
    feat_sel_steps = 5  # Number of feature selection steps for autofeat
    estimations = 50    # Number of estimations for BioAutoML, default = 50
    num_features = 20   # Number of Features for MAFESE in range(1, 20)

    working_dir = Path("src/amltk/results/third_try")  # Path
    random_seed = 42  # Set seed
    folds = 1  # Set number of folds (normal 10, test 1)

    # Choose set of datasets
    all_datasets = [1, 5, 14, 15, 16, 17, 18, 21, 22, 23, 24, 27, 28, 29, 31, 35, 36]  # 17

    small_datasets = [1, 5, 14, 16, 17, 18, 21, 27, 31, 35, 36]
    smallest_datasets = [14, 16, 17, 21, 35]  # n ~ 1000, p ~ 15
    big_datasets = [15, 22, 23, 24, 28, 29]
    test_new_method_datasets = [17]  # [18]  # [18]  # [16]

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
        max_trials = 100  # trade-off between exploration and resource usage
        max_time = 3600  # one hour
        n_workers = 4
        # Just mark the trial as fail and move on to the next one
        on_trial_exception = "continue"
        display = True
        wait_for_all_workers_to_finish = False

    for fold in range(folds):
        print("\n\n\n*******************************\n Fold " + str(fold) + "\n*******************************\n")
        inner_fold_seed = random_seed + fold
        # Iterate over all chosen datasets
        for option in all_datasets:
            # Get train test split dataset
            train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=option)

            # ############# Feature Engineering with xxx ############# #
            file_name = "results_" + str(name) + "_xxx_" + str(fold) + ".parquet"
            file = working_dir / file_name
            print("\n\n\n*******************************\n" + str(file_name) + "\n*******************************\n")
            if rerun or not os.path.isfile(file):
                print("Run OpenFE Method on Dataset")
                # train_x, train_y, test_x, test_y = get_splits(train_x, train_y, test_x, test_y)
                train_x_xxx, test_x_xxx = get_openFE_features(train_x, train_y, test_x, 1)
                evaluator = get_cv_evaluator(train_x_xxx, train_y, test_x_xxx, test_y, inner_fold_seed,
                                             on_trial_exception, task_hint)
                history_xxx = pipeline.optimize(
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
                df_xxx = history_xxx.df()
                safe_dataframe(df_xxx, working_dir, name, fold, "openfe")
            else:
                print("File exists, going for next method")


if __name__ == "__main__":
    main()
