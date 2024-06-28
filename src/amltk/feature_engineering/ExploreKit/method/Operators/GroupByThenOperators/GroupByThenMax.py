from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.amltk.feature_engineering.ExploreKit.method.Data.Dataset import Dataset
from src.amltk.feature_engineering.ExploreKit.method.Operators.GroupByThenOperators.GroupByThen import GroupByThen
from src.amltk.feature_engineering.ExploreKit.method.Evaluation.OperationAssignmentAncestorsSingleton import OperationAssignmentAncestorsSingleton
from src.amltk.feature_engineering.ExploreKit.method.Operators.Operator import Operator, outputType


class GroupByThenMax(GroupByThen):
    def __init__(self):
        super().__init__()
        self.missingValuesVal = 0
        self.maxValuePerKey: Dict[Tuple, float] = {}

    def processTrainingSet(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns: List[pd.Series]):
        super().processTrainingSet(dataset, sourceColumns, targetColumns)
        for key, vals in self.valuePerKey.items():
            val = max(vals)
            self.maxValuePerKey[key] = val

        # now we compute the "missing values val" - the value for samples in the test set for which we don't have a values based on the training set
        self.missingValuesVal = np.mean(list(self.maxValuePerKey.values()))

     # Generates the values of the new attribute. The values are generated BOTH for the training and test folds
     # (but the values are calculated based ONLY on the training set)
    def generate(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns: List[pd.Series]) -> pd.Series:
        column = []

        for i in range(dataset.getNumOfInstancesPerColumn()):
            # int j = dataset.getIndices().get(i);
            sourceValues = tuple(col.iloc[i] for col in sourceColumns)
            if sourceValues not in self.maxValuePerKey:
                column.append(self.missingValuesVal)
            else:
                column.append(self.maxValuePerKey[sourceValues])

        # now we generate the name of the new attribute
        attString = self.generateName(sourceColumns, targetColumns)
        finalString = self.getName() + attString + ")"

        # ColumnInfo newColumnInfo = new ColumnInfo(column, sourceColumns, targetColumns, this.getClass(), finalString);
        newColumn = pd.Series(column, dtype=float, name=finalString)
        oaas = OperationAssignmentAncestorsSingleton()
        oaas.addAssignment(finalString, sourceColumns, targetColumns)
        return newColumn

    def isApplicable(self,  dataset: Dataset, sourceColumns: List[pd.Series], targetColumns: List[pd.Series]) -> bool:
        if super().isApplicable(dataset, sourceColumns, targetColumns):
            if Operator.getSeriesType(targetColumns[0]) == outputType.Numeric:
                return True

        return False

    def getOutputType(self):
        return outputType.Numeric

    def getName(self) -> str:
        return "GroupByThenMax"
