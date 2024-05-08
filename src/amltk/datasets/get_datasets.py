import numpy as np
import pandas as pd
import openml
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def get_california_housing_dataset() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    data = fetch_california_housing(as_frame=True).frame
    label = data[['MedHouseVal']]
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=1)
    return train_x, train_y, test_x, test_y


def get_cylinder_bands_dataset(
        openml_task_id: int,
        fold: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
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
    train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
    test_x, test_y = X.iloc[test_idx], y.iloc[test_idx]
    return train_x, train_y, test_x, test_y


def get_balance_scale_dataset() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    balance_scale = fetch_ucirepo(id=12)
    X = balance_scale.data.features
    y = balance_scale.data.targets
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=20)
    return train_x, train_y, test_x, test_y


def get_black_friday_dataset() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    train = pd.read_csv(r'datasets/black-friday/train.csv', delimiter=',', header=None, skiprows=1,
                        names=['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
                               'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
                               'Product_Category_2', 'Product_Category_3', 'Purchase'])
    train_y = train[['Purchase']]
    train_x = train.drop(['Purchase'], axis=1)
    test = pd.read_csv(r'datasets/black-friday/test.csv', delimiter=',', header=None, skiprows=1,
                       names=['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
                              'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
                              'Product_Category_2', 'Product_Category_3', 'Purchase'])
    test_y = test[['Purchase']]
    test_x = test.drop(['Purchase'], axis=1)

    return train_x, train_y, test_x, test_y


def preprocess_dataframe(df) -> pd.DataFrame:
    if len(df.shape) == 1:
        df = pd.DataFrame(df, columns=['value', 'band_type'])
    cat_columns = df.select_dtypes(['category']).columns
    obj_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x, use_na_sentinel=True)[0])
    df[obj_columns] = df[obj_columns].apply(lambda x: pd.factorize(x, use_na_sentinel=True)[0])
    imp_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
    df = imp_nan.fit_transform(df)
    imp_m1 = SimpleImputer(missing_values=-1, strategy='mean')
    df = imp_m1.fit_transform(df)
    df = pd.DataFrame(df).fillna(0)
    return df
