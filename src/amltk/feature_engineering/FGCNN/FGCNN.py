# https://github.com/numb3r33/fgcnn
import pandas as pd
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.losses import BCELossFlat
from fastai.optimizer import ranger
from fastai.tabular.learner import TabularLearner

from src.amltk.feature_engineering.FGCNN.method.fgcnn.data import *
from src.amltk.feature_engineering.FGCNN.method.fgcnn.model import *
from src.amltk.feature_engineering.FGCNN.method.fgcnn.train import *



def get_xxx_features(train_x, train_y, test_x, test_y) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    df_train = pd.concat([train_x, train_y], axis=1)
    df_test = pd.concat([test_x, test_y], axis=1)
    df = pd.concat([df_train, df_test], axis=0)

    emb_szs = get_emb_sz(df_train, k=40)

    m = FGCNN(emb_szs=emb_szs,
              conv_kernels=[14, 16, 18, 20],
              kernels=[3, 3, 3, 3],
              dense_layers=[4096, 2048, 1024, 512],
              h=7,
              hp=2
              )

    learn = TabularLearner(df, m, loss_func=BCELossFlat(), opt_func=ranger)
    learn.fit_flat_cos(1, 2e-4, cbs=EarlyStoppingCallback())
    return train_x, test_x
