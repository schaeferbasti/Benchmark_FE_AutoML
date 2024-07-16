import numpy as np

from amltk.pipeline import Component, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from typing import Any
from collections.abc import Mapping
from ConfigSpace import Categorical, Integer, Float

from lightgbm import LGBMClassifier, LGBMRegressor


def rf_config_transform(config: Mapping[str, Any], _: Any) -> dict[str, Any]:
    new_config = dict(config)
    if new_config["class_weight"] == "None":
        new_config["class_weight"] = None
    return new_config


def get_rf_classifier():
    return Component(
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


def get_mlp_classifier():
    return Component(
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


def get_svc_classifier():
    return Component(
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


def get_knn_classifier():
    return Component(
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


def get_lgbm_classifier():
    return Component(
        item=LGBMClassifier,
        name="lgbm-classifier",
        config={
            "random_state": request("random_state"),
            "n_jobs": 1,
            "verbosity": -1,
        },
        space={
            "n_estimators": Integer("n_estimators", (200, 5000), default=200),
            "learning_rate": Float(
                "learning_rate",
                (5e-3, 0.1),
                default=0.05,
                log=True,
            ),
            "feature_fraction": Float(
                "feature_fraction",
                (0.4, 1.0),
                default=1.0,
            ),
            "min_data_in_leaf": Integer("min_data_in_leaf", (2, 60), default=20),
            "num_leaves": Integer("num_leaves", (16, 255), default=31),
            "extra_trees": Categorical("extra_trees", [False, True]),
        },
    )


def get_lgbm_regressor():
    return Component(
        item=LGBMRegressor,
        name="lgbm-regressor",
        config={
            "random_state": request("random_state"),
            "n_jobs": 1,
            "verbosity": -1,

        },
        space={
            "n_estimators": Integer("n_estimators", (32, 512), default=128),
            "learning_rate": Float(
                "learning_rate",
                (5e-3, 0.1),
                default=0.05,
                log=True,
            ),
            "feature_fraction": Float(
                "feature_fraction",
                (0.4, 1.0),
                default=1.0,
            ),
            "min_data_in_leaf": Integer("min_data_in_leaf", (2, 60), default=20),
            "num_leaves": Integer("num_leaves", (16, 255), default=31),
            "extra_trees": Categorical("extra_trees", [False, True]),
        },
    )
