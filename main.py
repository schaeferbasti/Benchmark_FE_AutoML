import openml
from openfe import OpenFE, transform
from amltk.sklearn import split_data
from sklearn.preprocessing import LabelEncoder

from sklearn.datasets import fetch_california_housing, load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

import pandas as pd

def get_openFE_features(train_x, test_x, train_y, n_jobs):
    openFE = OpenFE()
    features = openFE.fit(data=train_x, label=train_y, n_jobs=n_jobs)  # generate new features
    train_x, test_x = transform(train_x, test_x, features, n_jobs=n_jobs)
    return train_x, test_x


def calc_openFE_score(train_x, test_x, train_y, test_y):
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'seed': 1}
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(train_x, train_y, eval_set=[(test_x, test_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = pd.DataFrame(gbm.predict(test_x), index=test_x.index)
    score = mean_squared_error(test_y, pred)
    return score


def get_data(option):
    if option == 1:
        # OpenFE with OpenML Dataset
        dataset = openml.datasets.get_dataset(31, download_data=True, download_features_meta_data=False,
                                              download_qualities=False, )
        target_name = dataset.default_target_attribute
        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_name)
        _y = LabelEncoder().fit_transform(y)
        data = split_data(X, _y, splits={"train": 0.6, "val": 0.2, "test": 0.2}, seed=42)  # type: ignore
        train_x, train_y = data["train"]
        val_x, val_y = data["val"]
        test_x, test_y = data["test"]
        print(len(train_x), len(test_x), len(train_y), len(test_y))
        return train_x, train_y, test_x, test_y
    elif option == 2:
        # OpenFE with Iris Plants Dataset
        data = load_iris(as_frame=True).frame
        label = data[['target']]
        train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=1)
        print(len(train_x), len(test_x), len(train_y), len(test_y))
        return train_x, test_x, train_y, test_y
    elif option == 3:
        # OpenFE with California Housing Dataset
        data = fetch_california_housing(as_frame=True).frame
        label = data[['MedHouseVal']]
        train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=1)
        print(len(train_x), len(test_x), len(train_y), len(test_y))
        return train_x, test_x, train_y, test_y


def apply_openFE(train_x, test_x, train_y, test_y, n_jobs):
    print(train_x)
    train_x, test_x = get_openFE_features(train_x, test_x, train_y, n_jobs)
    print(train_x)
    score = calc_openFE_score(train_x, test_x, train_y, test_y)
    print("Score: " + str(score))


if __name__ == "__main__":
    n_jobs = 4

    # OpenFE with OpenML Dataset
    train_x, test_x, train_y, test_y = get_data(option=1)
    apply_openFE(train_x, test_x, train_y, test_y, n_jobs)
    # OpenFE with Iris Plants Dataset
    train_x, test_x, train_y, test_y = get_data(option=2)
    apply_openFE(train_x, test_x, train_y, test_y, n_jobs)
    # OpenFE with California Housing Dataset
    train_x, test_x, train_y, test_y = get_data(option=3)
    apply_openFE(train_x, test_x, train_y, test_y, n_jobs)


