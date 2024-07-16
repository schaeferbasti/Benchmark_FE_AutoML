from enum import Enum
from typing import List

import pandas as pd

from src.feature_engineering.ExploreKit.method.Data.Dataset import Dataset


class operatorType(Enum):
    Unary = 1
    Binary = 2
    GroupByThen = 3
    TimeBasedGroupByThen = 4

class outputType(Enum):
    Numeric = 1
    Discrete = 2
    Date = 3

class Operator:

    @staticmethod
    def getSeriesType(column: pd.Series) -> outputType:
        if pd.api.types.is_integer_dtype(column):
            return outputType.Discrete
        elif pd.api.types.is_float_dtype(column):
            return outputType.Numeric
        elif pd.api.types.is_datetime64_any_dtype(column):
            return outputType.Date
        elif pd.api.types.is_object_dtype(column):
            raise Exception("Can't handle object type columns")
        else:
            raise Exception('Unknown column type')

    def __init__(self):
        pass

    def getName(self) -> str:
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")

    def getType(self) -> operatorType:
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")

    def getOutputType(self) -> outputType:
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")

    def processTrainingSet(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns: List[pd.Series]):
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")

    def generate(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns: List[pd.Series]) -> pd.Series:
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")

    def isApplicable(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns: List[pd.Series]) -> bool:
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")