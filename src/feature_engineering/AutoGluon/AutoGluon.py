# https://github.com/autogluon/autogluon

import pandas as pd
from autogluon.features.generators import AutoMLInterpretablePipelineFeatureGenerator, AutoMLPipelineFeatureGenerator, IdentityFeatureGenerator


def get_autogluon_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    # feature_generator = IdentityFeatureGenerator()
    # feature_generator = AutoMLInterpretablePipelineFeatureGenerator()
    feature_generator = AutoMLPipelineFeatureGenerator()
    train_x = feature_generator.fit_transform(train_x, train_y)
    test_x = feature_generator.transform(test_x)

    return train_x, test_x
