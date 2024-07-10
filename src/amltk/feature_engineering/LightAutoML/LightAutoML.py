# https://github.com/sb-ai-lab/LightAutoML

import pandas as pd
import src.amltk.feature_engineering.LightAutoML.method.lightautoml.pipelines.features.lgb_pipeline as lgb


def get_xxx_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    train_df = pd.DataFrame(train_x, train_y)
    lgb_pipeline = lgb.LGBSimpleFeatures()
    train_x = lgb_pipeline.fit_transform(train_df)
    test_x = lgb_pipeline.transform(test_x)
    train_x = pd.DataFrame(train_x)
    test_x = pd.DataFrame(test_x)
    return train_x, test_x
