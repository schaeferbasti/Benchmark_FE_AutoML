# https://github.com/TjuJianyu/NFS

import pandas as pd
from src.feature_engineering.NFS.method.Main_sequence import main


def get_nfs_features(train_x, train_y, test_x, test_y, task_hint) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    df_train = pd.concat([train_x, train_y], axis=1)
    df_test = pd.concat([test_x, test_y], axis=1)
    df_original = pd.concat([df_train, df_test], axis=0)
    infos, name = main(train_x, train_y, task_hint)
    print(infos)
    print(name)
    return train_x, test_x
