from typing import List

import numpy as np
import pandas as pd

from src.feature_engineering.excluded.ExploreKit.method.Data.Dataset import Dataset
from src.feature_engineering.excluded.ExploreKit.method.Evaluation.OperationAssignmentAncestorsSingleton import OperationAssignmentAncestorsSingleton
from src.feature_engineering.excluded.ExploreKit.method.Operators.BinaryOperators.BinaryOperator import BinaryOperator
from src.feature_engineering.excluded.ExploreKit.method.Operators.Operator import outputType, operatorType


class AddBinaryOperator(BinaryOperator):

    def generate(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns):
        newColumn = sourceColumns[0].add(targetColumns[0]).replace([np.inf, -np.inf], np.nan).fillna(0)
        newColumn.name = 'Add' + self.generateName(sourceColumns, targetColumns)
        oaAncestors = OperationAssignmentAncestorsSingleton()
        oaAncestors.addAssignment(newColumn.name, sourceColumns, targetColumns)
        return newColumn

    def processTrainingSet(self, dataset: Dataset, sourceColumns: pd.Series, targetColumns):
        pass

    def getType(self) -> operatorType:
        return operatorType.Binary

    def getOutputType(self) -> outputType:
        return outputType.Numeric

    def getName(self) -> str:
        return 'AddBinaryOperator'

