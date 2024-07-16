# https://github.com/fuyuanlyu/AutoFS-in-CTR/tree/main/LPFS
import numpy as np
import pandas as pd
from pymfe.mfe import MFE


def get_xxx_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    mfe = MFE()
    mfe.fit(np.array(train_x), np.array(train_y))
    train_x = pd.DataFrame(train_x, np.array(mfe.extract()[1]))
    mfe.fit(np.array(test_x))
    test_x = pd.DataFrame(test_x, np.array(mfe.extract()[1]))
    return train_x, test_x
