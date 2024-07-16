from typing import List

import pandas as pd
import numpy as np

from src.feature_engineering.ExploreKit.method.Data.Dataset import Dataset
from src.feature_engineering.ExploreKit.method.Evaluation.OperationAssignmentAncestorsSingleton import OperationAssignmentAncestorsSingleton
from src.feature_engineering.ExploreKit.method.Operators.BinaryOperators.BinaryOperator import BinaryOperator
from src.feature_engineering.ExploreKit.method.Operators.Operator import outputType


class SubtractBinaryOperator(BinaryOperator):

    def generate(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns):
        newColumn = sourceColumns[0].sub(targetColumns[0]).replace([np.inf, -np.inf], np.nan).fillna(0)
        newColumn.name = 'Subtract' + self.generateName(sourceColumns, targetColumns)
        oaAncestors = OperationAssignmentAncestorsSingleton()
        oaAncestors.addAssignment(newColumn.name, sourceColumns, targetColumns)
        return newColumn

    def processTrainingSet(self, dataset: Dataset, sourceColumns: pd.Series, targetColumns):
        pass

    def getOutputType(self) -> outputType:
        return outputType.Numeric

    def getName(self) -> str:
        return 'SubtractBinaryOperator'

