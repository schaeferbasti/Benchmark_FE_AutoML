# https://github.com/fuyuanlyu/AutoFS-in-CTR/tree/main/LPFS
import pickle
import os
import numpy as np
import pandas as pd
from causalnex.structure import DAGClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.feature_engineering.MACFE.method.transform import transform_unary, transform_binary, transform_scaler


def get_macfe_features(train_x, train_y, test_x, test_y, name) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    d_list = [1, 2, 3, 4]
    s_list = [.2, .4, .6, .8, 1]  # top 20%, top 40%, ...
    df_train = pd.concat([train_x, train_y], axis=1)
    df_test = pd.concat([test_x, test_y], axis=1)
    df_original = pd.concat([df_train, df_test], axis=0)

    # START - Automated Feature Engineering Process
    print(f"\n*** Starting MACFE, d:{d_list}, s:{s_list} ***\n")
    print(f"Working on {name}")

    print("Original Dim: ", df_train.shape[1] - 1)
    TRM_dataset, TRM_binary_dataset, TRM_scaler = get_TRMs()

    df_fe = pd.DataFrame()
    df_selected_list = feature_selection(df_original, s_list)
    for s, df_selected in zip(s_list, df_selected_list):
        df_engineered_list = feature_construction(df_selected, d_list, TRM_dataset, TRM_binary_dataset)
        print("Evaluation...")
        for d, df_engineered in zip(d_list, df_engineered_list):
            df_fe = pd.concat([df_selected, df_engineered], axis=1)

    X = df_fe.drop(columns=["class"])
    y = df_fe["class"]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1)
    return train_x, test_x


def feature_construction(df_original, d_list, TRM_dataset, TRM_binary_dataset):
    df_engineered_list = []
    print("Construction...")
    df_engineered = df_original.copy(deep=True)

    max_d = max(d_list)

    for d_i in range(1, max_d + 1):
        df_engineered = _feature_construction_step(df_engineered, TRM_dataset, TRM_binary_dataset)
        # Drop Original Features from engineered ones
        df_engineered_d = df_engineered.drop(df_original.columns, axis=1)

        if (d_i in d_list):
            print(f"d:{d_i} done.")
            # Add df to the testing list if current d in in d_list
            df_engineered_list.append(df_engineered_d)

    return df_engineered_list


def feature_selection(df, s_list):
    print("Selection...")
    X, y = preprocess_dataset(df)
    y = pd.Series(y, name="class")

    dag = DAGClassifier(
        alpha=0.01,
        beta=0.5,
        hidden_layer_units=[5],
        fit_intercept=True,
        standardize=True
    )
    X = X.astype(float)
    y = y.astype(int)
    dag.fit(X.values, y)

    # List to save features for each "s"
    df_selected_list = []

    for threshold in s_list:
        # Select top (1 - threshold)%
        _threshold = np.quantile(dag.feature_importances_[0], (1.0 - threshold))
        selection_idx = np.where(dag.feature_importances_[0] >= _threshold)[0]
        X_selected = X.iloc[:, selection_idx]

        _column_names = np.array(df.columns.tolist()[:-1])
        _column_names = _column_names[selection_idx]

        # Create new selected DataFrame
        df_e = pd.DataFrame(X_selected)
        df_e.columns = _column_names
        df_e['class'] = y
        df_selected_list.append(df_e)
        print(f's:{threshold}, Dim:{df_e.shape[1] - 1}')

    return df_selected_list


def feature_scaler(df, TRM_scaler):
    X = df.drop(['class'], axis=1)
    y = df['class']

    column_names = X.columns.tolist().copy()

    df_scaled = transform_scaler(
        X.values,
        y.values,
        column_names,
        TRM_scaler)
    return df_scaled


def preprocess_dataset(df):
    # Preprocess data
    df = df.replace('?', np.nan)

    # df = df.fillna(df.mean())

    X_raw = df.iloc[:, 0:-1]
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X_raw = X_raw.select_dtypes(include=numerics)
    X_raw = X_raw.fillna(X_raw.mean())
    X_raw = X_raw.astype('float32')

    le = LabelEncoder()
    y_raw = df.iloc[:, -1].values
    y_raw = le.fit_transform(y_raw)

    return X_raw, y_raw


def get_TRMs():
    with open("src/feature_engineering/MACFE/data/TRM_set.pkl", 'rb') as f:
        TRM_set = pickle.load(f)
    TRM_dataset = list()
    for i in range(len(TRM_set)):
        TRM_dataset.append(
            np.append(TRM_set[i]['encoding'].ravel(), TRM_set[i]['top_t_index']))
    TRM_dataset = np.array(TRM_dataset)

    with open('src/feature_engineering/MACFE/data/TRM_binary_set_maxf1f2.pkl', 'rb') as f:
        TRM_binary_set = pickle.load(f)
    TRM_binary_dataset = list()
    for i in range(len(TRM_binary_set)):
        TRM_binary_dataset.append(
            np.append(TRM_binary_set[i]['encoding'].ravel(), TRM_binary_set[i]['top_t_index']))
    TRM_binary_dataset = np.array(TRM_binary_dataset)

    with open('src/feature_engineering/MACFE/data/TRM_scaler_set.pkl', 'rb') as f:
        TRM_scaler_set = pickle.load(f)
    TRM_scaler_dataset = list()
    for i in range(len(TRM_scaler_set)):
        TRM_scaler_dataset.append(
            np.append(TRM_scaler_set[i]['encoding'].ravel(), TRM_scaler_set[i]['top_t_index']))
    TRM_scaler_dataset = np.array(TRM_scaler_dataset)

    return TRM_dataset, TRM_binary_dataset, TRM_scaler_dataset


def _feature_construction_step(df, TRM_dataset, TRM_binary_dataset):
    X, y = preprocess_dataset(df)
    _column_names = df.columns.tolist()[:-1].copy()

    X_new_unary, _column_names_unary = transform_unary(X.values, y, _column_names, TRM_dataset)
    X_new_binary, _column_names_binary = transform_binary(X.values, y, _column_names, TRM_binary_dataset)

    # New DF with novel features
    df_e = df.copy()
    df_e = df_e.drop(['class'], axis=1)

    if (X_new_unary is not None):
        df_unary = pd.DataFrame(X_new_unary, columns=_column_names_unary)
        df_e = pd.concat([df_e, df_unary], axis=1)

    if (X_new_binary is not None):
        df_binary = pd.DataFrame(X_new_binary, columns=_column_names_binary)
        df_e = pd.concat([df_e, df_binary], axis=1)

    df_e['class'] = y

    return df_e
