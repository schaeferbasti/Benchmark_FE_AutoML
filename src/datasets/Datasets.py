import numpy as np
import pandas as pd
import openml
from pandas import Series, DataFrame
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


def get_openml_dataset(
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


def preprocess_data(train_x, test_x) -> (pd.DataFrame, pd.DataFrame):
    cols = train_x.columns
    cat_columns = train_x.select_dtypes(['category']).columns
    obj_columns = train_x.select_dtypes(['object']).columns
    train_x[cat_columns] = train_x[cat_columns].apply(lambda x: pd.factorize(x, use_na_sentinel=True)[0])
    test_x[cat_columns] = test_x[cat_columns].apply(lambda x: pd.factorize(x, use_na_sentinel=True)[0])
    train_x[obj_columns] = train_x[obj_columns].apply(lambda x: pd.factorize(x, use_na_sentinel=True)[0])
    test_x[obj_columns] = test_x[obj_columns].apply(lambda x: pd.factorize(x, use_na_sentinel=True)[0])
    imp_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
    train_x = imp_nan.fit_transform(train_x)
    test_x = imp_nan.transform(test_x)
    imp_m1 = SimpleImputer(missing_values=-1, strategy='mean')
    train_x = imp_m1.fit_transform(train_x)
    test_x = imp_m1.transform(test_x)
    train_x = pd.DataFrame(train_x).fillna(0)
    test_x = pd.DataFrame(test_x).fillna(0)
    train_x.columns = cols
    test_x.columns = cols
    return train_x, test_x


def preprocess_target(label) -> DataFrame:
    label = pd.DataFrame(label)
    cols = label.columns
    imp_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
    label = imp_nan.fit_transform(label)
    label = pd.DataFrame(label).fillna(0)
    label.columns = cols
    return label


def get_amlb_dataset(openml_task_id) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | pd.Series,
    pd.DataFrame | pd.Series,
    str,
    str
]:
    outer_fold_number = 0
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    train_idx, test_idx = task.get_train_test_split_indices(fold=0)
    name = task.get_dataset().name
    if task.task_type == "Supervised Classification":
        task_hint = "classification"
    else:
        task_hint = "regression"
    X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
    train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
    test_x, test_y = X.iloc[test_idx], y.iloc[test_idx]
    return train_x, train_y, test_x, test_y, name, task_hint


def get_dataset(option) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | pd.Series,
    pd.DataFrame | pd.Series,
    str,
    str
]:

    """All Datasets from the AutoML Benchmark with a number of 1000-5000 samples"""
    #### REGRESSION ####
    # abalone (n = 4177, p = 9) (yes)
    if option == 1:
        name = "abalone_dataset"
        task_hint = "regression"
        openml_task_id = 359944
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # house prices nominal (n = 1460, p = 80) (no)
    elif option == 2:
        name = "house_prices_nominal_dataset"
        task_hint = "regression"
        openml_task_id = 359951
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # Mercedes Benz Greener Manufacturing (n = 4209, p = 377) (no)
    elif option == 3:
        name = "mercedes_dataset"
        task_hint = "regression"
        openml_task_id = 233215
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # MIP-2016-regression (n = 1090, p = 145) (no)
    elif option == 4:
        name = "mip_dataset"
        task_hint = "regression"
        openml_task_id = 360945
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # moneyball (n = 1232, p = 15) (yes)
    elif option == 5:
        name = "moneyball_dataset"
        task_hint = "regression"
        openml_task_id = 167210
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # quake (n = 2178, p = 4) (no)
    elif option == 6:
        name = "quake_dataset"
        task_hint = "regression"
        openml_task_id = 359930
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # Santander transaction value (n = 4459, p = 4992) (no)
    elif option == 7:
        name = "santander_dataset"
        task_hint = "regression"
        openml_task_id = 233214
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # SAT11-HAND-runtime-regression (n = 4440, p = 117) (no)
    elif option == 8:
        name = "sat11_dataset"
        task_hint = "regression"
        openml_task_id = 359948
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # socmob (n = 1156, p = 6) (no)
    elif option == 9:
        name = "socmob_dataset"
        task_hint = "regression"
        openml_task_id = 359932
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # space ga (n = 3107, p = 7) (no)
    elif option == 10:
        name = "space_ga_dataset"
        task_hint = "regression"
        openml_task_id = 359933
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # us crime (n = 1994, p = 127) (no)
    elif option == 11:
        name = "us_crime_dataset"
        task_hint = "regression"
        openml_task_id = 359945
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name

    #### CLASSIFICATION ####
    # ada (n = 4147, p = 49) (no)
    elif option == 12:
        name = "ada_dataset"
        task_hint = "classification"
        openml_task_id = 190411
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # amazon-commerce-reviews (n = 1500, p = 10001) (no)
    elif option == 13:
        name = "amazon_commerce_reviews_dataset"
        task_hint = "classification"
        openml_task_id = 10090
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # australian dataset (n = 690, p = 15) (yes)
    elif option == 14:
        name = "australian_dataset"
        task_hint = "classification"
        openml_task_id = 146818
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # bioresponse (n = 3751, p = 1777) (yes, too many features)
    elif option == 15:
        name = "bioresponse_dataset"
        task_hint = "classification"
        openml_task_id = 359967
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # blood transfusion service center dataset (n = 748, p = 5) (yes)
    elif option == 16:
        name = "blood_transfusion_service_center_dataset"
        task_hint = "classification"
        openml_task_id = 359955
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # car (n = 1728, p = 7) (yes)
    elif option == 17:
        name = "car_dataset"
        task_hint = "classification"
        openml_task_id = 359960
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # churn (n = 5000, p = 21) (yes)
    elif option == 18:
        name = "churn_dataset"
        task_hint = "classification"
        openml_task_id = 359968
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # cmc (n = 1473, p = 10) (no)
    elif option == 19:
        name = "cmc_dataset"
        task_hint = "classification"
        openml_task_id = 359959
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # cnae-9 (n = 1080, p = 857) (no)
    elif option == 20:
        name = "cnae-9_dataset"
        task_hint = "classification"
        openml_task_id = 359957
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # credit g dataset (n = 1000, p = 21) (yes)
    elif option == 21:
        name = "credit_g_dataset"
        task_hint = "classification"
        openml_task_id = 168757
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # dna (n = 3186, p = 181) (yes)
    elif option == 22:
        name = "dna_dataset"
        task_hint = "classification"
        openml_task_id = 359964
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # gina (n = 3153, p = 971) (yes)
    elif option == 23:
        name = "gina_dataset"
        task_hint = "classification"
        openml_task_id = 189922
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # Internet-Advertisements (n = 3279, p = 1559) (yes)
    elif option == 24:
        name = "internet_advertisements_dataset"
        task_hint = "classification"
        openml_task_id = 359966
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # jasmine (n = 2984, p = 145) (no)
    elif option == 25:
        name = "jasmine_dataset"
        task_hint = "classification"
        openml_task_id = 168911
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # kc1 (n = 2109, p = 22) (no)
    elif option == 26:
        name = "kc1_dataset"
        task_hint = "classification"
        openml_task_id = 359962
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # kr-vs-kp (n = 3196, p = 37) (yes)
    elif option == 27:
        name = "kr_vs_kp_dataset"
        task_hint = "classification"
        openml_task_id = 359965
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # madeline (n = 3140, p = 260) (yes)
    elif option == 28:
        name = "madeline_dataset"
        task_hint = "classification"
        openml_task_id = 190392
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # mfeat-factors (n = 2000, p = 217) (yes)
    elif option == 29:
        name = "mfeat_factors_dataset"
        task_hint = "classification"
        openml_task_id = 359961
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # ozone-level-8hr (n = 2534, p = 73) (no)
    elif option == 30:
        name = "ozone_level_dataset"
        task_hint = "classification"
        openml_task_id = 190137
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # pc4 (n = 1458, p = 38) (yes)
    elif option == 31:
        name = "pc4_dataset"
        task_hint = "classification"
        openml_task_id = 359958
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # qsar-biodeg (n = 1055, p = 42) (no)
    elif option == 32:
        name = "qsar_biodeg_dataset"
        task_hint = "classification"
        openml_task_id = 359956
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # segment (n = 2310, p = 20) (no)
    elif option == 33:
        name = "segment_dataset"
        task_hint = "classification"
        openml_task_id = 359963
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # steel-plates-fault (n = 1941, p = 28) (no)
    elif option == 34:
        name = "steel_plates_fault_dataset"
        task_hint = "classification"
        openml_task_id = 168784
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # wilt (n = 4839, p = 6) (yes)
    elif option == 35:
        name = "wilt_dataset"
        task_hint = "classification"
        openml_task_id = 146820
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # wine-quality-white (n = 4898, p = 12) (yes)
    elif option == 36:
        name = "wine_quality_white_dataset"
        task_hint = "classification"
        openml_task_id = 359974
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # yeast (n = 1484, p = 9) (no)
    elif option == 37:
        name = "yeast_dataset"
        task_hint = "classification"
        openml_task_id = 2073
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name


def get_old_dataset(option) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | pd.Series,
    pd.DataFrame | pd.Series,
    str,
    str
]:
    """ Not working datasets """
    # california-housing dataset from OpenFE example (no)
    if option == 1:
        name = "california_housing_dataset"
        task_hint = "regression"
        train_x, train_y, test_x, test_y = get_california_housing_dataset()
        return train_x, train_y, test_x, test_y, task_hint, name
    # cylinder-bands dataset from OpenFE benchmark (no)
    elif option == 2:
        name = "cylinder_bands_dataset"
        task_hint = "classification"
        openml_task_id = 1797
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # balance-scale dataset from OpenFE benchmark (no)
    elif option == 3:
        name = "balance_scale_dataset"
        task_hint = "classification"
        train_x, train_y, test_x, test_y = get_balance_scale_dataset()
        return train_x, train_y, test_x, test_y, task_hint, name
    # black-friday dataset from AMLB (no, too big)
    elif option == 4:
        name = "black_friday_dataset"
        task_hint = "classification"
        train_x, train_y, test_x, test_y = get_black_friday_dataset()
        return train_x, train_y, test_x, test_y, task_hint, name
    # boston dataset (no)
    elif option == 5:
        name = "boston_dataset"
        task_hint = "regression"
        openml_task_id = 359950
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # sensory dataset (no)
    elif option == 6:
        name = "sensory_dataset"
        task_hint = "regression"
        openml_task_id = 359931
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        return train_x, train_y, test_x, test_y, task_hint, name
    # eucalyptus dataset (no)
    elif option == 7:
        name = "eucalyptus_dataset"
        task_hint = "classification"
        openml_task_id = 359954
        outer_fold_number = 0
        train_x, train_y, test_x, test_y = get_openml_dataset(openml_task_id=openml_task_id, fold=outer_fold_number)
        openml.tasks.get_task(openml_task_id=openml_task_id)
        return train_x, train_y, test_x, test_y, task_hint, name

def construct_dataframe(train_x, train_y, test_x, test_y):
    df_train = pd.concat([train_x, train_y], axis=1)
    df_test = pd.concat([test_x, test_y], axis=1)
    df = pd.concat([df_train, df_test], axis=0)
    return df
