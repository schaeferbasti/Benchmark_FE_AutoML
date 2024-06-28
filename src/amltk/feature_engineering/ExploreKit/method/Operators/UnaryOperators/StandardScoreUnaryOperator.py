from typing import List

import numpy as np
import pandas as pd

from Data.Dataset import Dataset
from Utils.Logger import Logger
from Operators.Operator import Operator, operatorType, outputType
from Operators.UnaryOperators.UnaryOperator import UnaryOperator


class StandardScoreUnaryOperator(UnaryOperator):

    def __init__(self, upperBoundPerBin):
        self.upperBoundPerBin = np.array(upperBoundPerBin)
        self.avg = 0
        self.stdev = 0
        Operator.__init__(self)

    def processTrainingSet(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns):
        col = sourceColumns[0]
        col = col.iloc[dataset.getIndicesOfTrainingInstances()].replace([np.inf, -np.inf], np.nan).fillna(0)
        self.avg = col.mean()
        self.stdev = col.std()

    def generate(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns: list) -> pd.Series:
        if self.stdev == 0:
            newCol = pd.Series(np.zeros(sourceColumns[0].shape[0],), dtype=np.float_)
        else:
            newCol = (sourceColumns[0] - self.avg) / self.stdev
            newCol = newCol.replace([np.inf, -np.inf], np.nan).fillna(0)

        newCol.name = 'StandardScoreUnaryOperator' + self.generateName(sourceColumns, targetColumns)
        return newCol

    def getType(self) -> operatorType:
        return operatorType.Unary

    def getOutputType(self) -> outputType:
        return outputType.Numeric

    def getName(self) -> str:
        return 'StandardScoreUnaryOperator'

    def requiredInputType(self):
        return outputType.Numeric

    def isApplicable(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns: List[pd.Series]) -> bool:
        if super().isApplicable(dataset, sourceColumns, targetColumns):
            return Operator.getSeriesType(sourceColumns[0]) == outputType.Numeric
        return False

