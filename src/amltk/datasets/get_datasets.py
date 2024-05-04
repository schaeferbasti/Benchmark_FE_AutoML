import pandas as pd
import openml
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def get_california_housing_dataset():
    data = fetch_california_housing(as_frame=True).frame
    label = data[['MedHouseVal']]
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=1)
    return test_x, test_y, train_x, train_y


def get_cylinder_folds_dataset(
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


def get_balance_scale_dataset():
    balance_scale = fetch_ucirepo(id=12)
    X = balance_scale.data.features
    y = balance_scale.data.targets
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=20)
    return test_X, test_y, train_X, train_y


def get_black_friday_dataset():
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
    return test_X, test_y, train_X, train_y