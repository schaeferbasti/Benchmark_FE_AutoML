# https://github.com/sb-ai-lab/LightAutoML
import numpy as np
import pandas as pd

import src.feature_engineering.excluded.LightAutoML.method.lightautoml.pipelines.features.lgb_pipeline as lgb
from src.feature_engineering.excluded.LightAutoML.method.lightautoml.dataset.base import LAMLDataset
from src.feature_engineering.excluded.LightAutoML.method.lightautoml.dataset.roles import TargetRole, CategoryRole, NumericRole, DatetimeRole, FoldsRole
from src.feature_engineering.excluded.LightAutoML.method.lightautoml.dataset.utils import roles_parser


def get_xxx_features(train_x, train_y, test_x, task_hint) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    # columns = train_x.columns
    # columns.append("target")
    data = pd.DataFrame(train_x, train_y)
    features = data.columns
    roles = extract_roles(train_x)

    laml_dataset = LAMLDataset(data, features, roles_parser(roles))
    lgb_pipeline = lgb.LGBSimpleFeatures()
    train_x = lgb_pipeline.fit_transform(laml_dataset)
    test_x = lgb_pipeline.transform(test_x)
    train_x = pd.DataFrame(train_x)
    test_x = pd.DataFrame(test_x)
    return train_x, test_x


def extract_roles(df):
    roles = {}
    for column in df.columns:
        if column == "target":
            if TargetRole() in roles.keys():
                roles[TargetRole()].append(column)
            else:
                roles[TargetRole()] = [column]
        elif df[column].dtype == "category":
            if CategoryRole(dtype=str) in roles.keys():
                roles[CategoryRole(dtype=str)].append(column)
            else:
                roles[CategoryRole(dtype=str)] = [column]
        elif np.issubdtype(df[column].dtype, np.number):
            if NumericRole(np.float32) in roles.keys():
                roles[NumericRole(np.float32)].append(column)
            else:
                roles[NumericRole(np.float32)] = [column]
        else:
            if CategoryRole(dtype=str) in roles.keys():
                roles[CategoryRole(dtype=str)].append(column)
            else:
                roles[CategoryRole(dtype=str)] = [column]
    return roles
