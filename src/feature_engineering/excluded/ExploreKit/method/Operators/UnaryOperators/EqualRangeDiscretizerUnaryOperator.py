from typing import List

import numpy as np
import pandas as pd

from src.feature_engineering.ExploreKit.method.Data.Dataset import Dataset
from src.feature_engineering.ExploreKit.method.Utils.Logger import Logger
from src.feature_engineering.ExploreKit.method.Operators.Operator import Operator, operatorType, outputType
from src.feature_engineering.ExploreKit.method.Operators.UnaryOperators.UnaryOperator import UnaryOperator


class EqualRangeDiscretizerUnaryOperator(UnaryOperator):

    def __init__(self, upperBoundPerBin):
        self.upperBoundPerBin = np.array(upperBoundPerBin)
        Operator.__init__(self)

    def processTrainingSet(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns):
        trainIndices = dataset.getIndicesOfTrainingInstances()
        # if len(sourceColumns.shape) != 1: #Not Series
        #     values = sourceColumns.loc[trainIndices].iloc[:, 0]
        # else:
        #     values = sourceColumns.loc[trainIndices]
        values = sourceColumns[0].values
        minVal = np.nanmin(values)
        maxVal = np.nanmax(values)
        self.upperBoundPerBin = np.linspace(minVal, maxVal, self.upperBoundPerBin.shape[0])
    # def processTrainingSet(self, dataset: Dataset, sourceColumns: list, targetColumns:list):
    #     # minVal = Double.MAX_VALUE;
    #     # double maxVal = Double.MIN_VALUE;
    #
    #     columnInfo = sourceColumns.get(0)
    #     val = columnInfo.getColumn().getValue(0)
    #     minVal = maxVal = val
    #     for i in range(dataset.getNumOfTrainingDatasetRows()):
    #         # TODO: j instead of i
    #         j = dataset.getIndicesOfTrainingInstances().get(i)
    #         val = columnInfo.getColumn().getValue(j)
    #         if (not np.isnan(val)) and (not np.isinf(val)):
    #             minVal = min(minVal, val)
    #             maxVal = max(maxVal, val)
    #         else:
    #             x=5
    #
    #     rng = (maxVal-minVal)/len(self.upperBoundPerBin)
    #     currentVal = minVal
    #     for i in range(len(self.upperBoundPerBin)):
    #         self.upperBoundPerBin[i] = currentVal + rng
    #         currentVal += rng

    def generate(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns: list) -> pd.Series:
        try:
            sourceColumn = sourceColumns[0]
            values = sourceColumn.values
            discretized = np.digitize(values, self.upperBoundPerBin) - 1
            name='EqualRangeDiscretizer('+str(sourceColumn.name)+')'
            return pd.Series(data=discretized, name=name, index=sourceColumn.index, dtype='int')
        except Exception as ex:
            Logger.Error('error in EqualRangeDiscretizer: '+str(ex))
            return None
        # try:
        #     # DiscreteColumn column = DiscreteColumn(dataset.getNumOfInstancesPerColumn(), upperBoundPerBin.length)
        #     column = np.empty(dataset.getNumOfInstancesPerColumn(), dtype=np.dtype(int))
        #     # this is the number of rows we need to work on - not the size of the vector
        #     numOfRows = dataset.getNumberOfRows()
        #     columnInfo = sourceColumns.get(0)
        #     for i in range(numOfRows):
        #         # if (dataset.getIndices().size() == i) {
        #         #     int x = 5;
        #         # }
        #         j = dataset.getIndices()[i]
        #         binIndex = self.GetBinIndex(columnInfo.getColumn().getValue(j))
        #         column.setValue(j, binIndex)
        #
        #     # now we generate the name of the new attribute
        #     attString = "EqualRangeDiscretizer(" + columnInfo.getName()+ ")"
        #     return pd.DataFrame({attString: column})
        #     # new ColumnInfo(column, sourceColumns, targetColumns, this.getClass(), attString);
        #
        # except Exception as ex:
        #     Logger.Error("error in EqualRangeDiscretizer:  " +  ex)
        #     return None

    def GetBinIndex(self, value: float):
        for i in range(len(self.upperBoundPerBin)):
            if self.upperBoundPerBin[i] > value:
                return i

        return len(self.upperBoundPerBin) - 1

    def getType(self) -> operatorType:
        return operatorType.Unary

    def getOutputType(self) -> outputType:
        return outputType.Numeric

    def getName(self) -> str:
        return 'EqualRangeDiscretizerUnaryOperator'

    def requiredInputType(self):
        return outputType.Numeric

    def isApplicable(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns: List[pd.Series]) -> bool:
        if super().isApplicable(dataset, sourceColumns, targetColumns):
            return Operator.getSeriesType(sourceColumns[0]) == outputType.Numeric
        return False

