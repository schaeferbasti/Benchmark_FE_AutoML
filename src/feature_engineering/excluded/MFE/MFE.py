# https://github.com/fuyuanlyu/AutoFS-in-CTR/tree/main/LPFS
import numpy as np
import pandas as pd
from pymfe.mfe import MFE


def get_mfe_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    mfe = MFE()
    mfe.fit(np.array(train_x), np.array(train_y))
    extract = mfe.extract()
    print(extract)
    extract_names = mfe.extract_metafeature_names()
    print(extract_names)

    mtfs_all = MFE.valid_metafeatures()
    for mtf in mtfs_all:
        print(mtf.metafeature_name)
    return train_x, test_x
