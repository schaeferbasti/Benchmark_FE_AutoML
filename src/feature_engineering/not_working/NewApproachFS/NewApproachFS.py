# https://github.com/tud-zih-energy/pymit/tree/master
import numpy as np
import pandas as pd
from src.feature_engineering.NewApproachFS import pymit



def get_xxx_features(train_x, train_y, test_x, test_y) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    df_train = pd.concat([train_x, train_y], axis=1)
    df_test = pd.concat([test_x, test_y], axis=1)
    data = pd.concat([df_train, df_test], axis=0)
    labels = data.columns
    bins = 10
    expected_features = [241, 338, 378, 105, 472]#, 475, 433, 64, 128, 442, 453, 336, 48, 493, 281, 318, 153, 28, 451, 455]

    [num_examples, num_features] = data.shape
    data_discrete = np.zeros([num_examples, num_features])
    for i in range(num_features):
        _, bin_edges = pymit._lib.histogram(data[:, i], bins=bins)
        data_discrete[:, i] = pymit._lib.digitize(data[:, i], bin_edges, right=False)

    max_features = len(expected_features)
    selected_features = []
    j_h = 0
    hjmi = None

    for i in range(0, max_features):
        jmi = np.zeros([num_features], dtype=np.float)
        for X_k in range(num_features):
            if X_k in selected_features:
                continue
            jmi_1 = pymit.I(data_discrete[:, X_k], labels, bins=[bins, 2])
            jmi_2 = 0
            for X_j in selected_features:
                tmp1 = pymit.I(data_discrete[:, X_k], data_discrete[:, X_j], bins=[bins, bins])
                tmp2 = pymit.I_cond(data_discrete[:, X_k], data_discrete[:, X_j], labels, bins=[bins, bins, 2])
                jmi_2 += tmp1 - tmp2
            if len(selected_features) == 0:
                jmi[X_k] = j_h + jmi_1
            else:
                jmi[X_k] = j_h + jmi_1 - jmi_2/len(selected_features)
        f = jmi.argmax()
        j_h = jmi[f]
        if hjmi is None or (j_h - hjmi)/hjmi > 0.03:
            hjmi = j_h
            selected_features.append(f)
        else:
            break

    return train_x, test_x
