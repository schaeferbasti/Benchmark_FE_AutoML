from typing import List

import pandas as pd

from src.feature_engineering.excluded.ExploreKit.method.Data.Dataset import Dataset
from src.feature_engineering.excluded.ExploreKit.method.Operators.Operator import Operator, operatorType, outputType


class UnaryOperator(Operator):

    def __init__(self):
        super().__init__()
        self.abc: List[int]

    def getType(self) -> operatorType:
        return operatorType.Unary

    def requiredInputType(self) -> outputType:
        raise NotImplementedError("UnaryOperator is an abstract class")

    def isApplicable(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns: List[pd.Series]) -> bool:
        # if there are any target columns or if there is more than one source column, return false
        if len(sourceColumns) != 1 or (targetColumns != None and len(targetColumns) != 0):
            return False
        else:
            return True

    def generateName(self, sourceColumns: List[pd.Series],  targetColumns: List[pd.Series]) -> str:
        return f"({sourceColumns[0].name})"
