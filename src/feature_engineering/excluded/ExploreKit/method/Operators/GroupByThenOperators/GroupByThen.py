from typing import List

import numpy as np
import pandas as pd

from src.feature_engineering.ExploreKit.method.Data.Dataset import Dataset
from src.feature_engineering.ExploreKit.method.Evaluation.OperationAssignmentAncestorsSingleton import OperationAssignmentAncestorsSingleton
from src.feature_engineering.ExploreKit.method.Operators.Operator import outputType, operatorType, Operator


class GroupByThen(Operator):

    def __init__(self):
        super().__init__()
        self.valuePerKey = {}

    def processTrainingSet(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns: List[pd.Series]):
        df = pd.concat(sourceColumns, axis=1)
        for indice in dataset.getIndicesOfTrainingInstances():
            targetValue = targetColumns[0].iloc[indice]
            if np.isinf(targetValue) or np.isnan(targetValue):
                continue
            key = tuple(df.iloc[indice,:].values)
            self.valuePerKey.setdefault(key, []).append(targetValue)

    def isApplicable(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns: List[pd.Series]) -> bool:
        if (targetColumns is None) or len(targetColumns) != 1:
            return False
        for column in sourceColumns:
            if Operator.getSeriesType(column) != outputType.Discrete:
                return False
        return True

    def getType(self) -> operatorType:
        return operatorType.GroupByThen

    def getOutputType(self) -> outputType:
        return outputType.Numeric

    def generateName(self, sourceColumns: List[pd.Series], targetColumns: List[pd.Series]):
        sourceAtts = "Source({0})".format(';'.join([col.name for col in sourceColumns]))

        targetAtts = "Target({0})".format(';'.join([col.name for col in targetColumns]))

        finalString = f"{sourceAtts}_{targetAtts}"
        return finalString

