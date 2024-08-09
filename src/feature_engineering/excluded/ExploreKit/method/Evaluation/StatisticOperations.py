from typing import List

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, chi2_contingency

from src.feature_engineering.excluded.ExploreKit.method.Data.Dataset import Dataset
from src.feature_engineering.excluded.ExploreKit.method.Operators.UnaryOperators.EqualRangeDiscretizerUnaryOperator import EqualRangeDiscretizerUnaryOperator
from src.feature_engineering.excluded.ExploreKit.method.Operators.Operator import Operator, outputType
from src.feature_engineering.excluded.ExploreKit.method.Properties import Properties


class StatisticOperations:

    # The function receives two lists of features and returns a list of each possible pairs Paired T-Test values
    @staticmethod
    def _calculatePairedTTestValues(list1: List[pd.Series], list2: List[pd.Series]):
        tTestValues = []
        for ci1 in list1:
            if not pd.api.types.is_float_dtype(ci1):
                raise Exception("Unable to process non-numeric columns - list 1")

            for ci2 in list2:
                if not pd.api.types.is_float_dtype(ci2):
                    raise Exception("Unable to process non-numeric columns - list 2")

                (statistic, pvalue) = ttest_rel(ci1,ci2)
                testValue = np.abs(statistic)
                if not np.isnan(testValue):
                    tTestValues.append(testValue)
        return tTestValues

    @staticmethod
    def calculatePairedTTestValues(list1: List[pd.Series], columnInfo: pd.Series):
        tempList = []
        tempList.append(columnInfo)
        return StatisticOperations._calculatePairedTTestValues(list1, tempList)

    # Calculates the Chi-Square test values among all the possible combonation of elements in the two provided list.
    # Also supports numeric attributes, a discretized versions of which will be used in the calculation.
    @staticmethod
    def calculateChiSquareTestValues(list1: List[pd.Series], list2: List[pd.Series], dataset: Dataset) -> List[float]:
        bins = [0] * Properties.equalRangeDiscretizerBinsNumber
        erduo = EqualRangeDiscretizerUnaryOperator(bins)
        chiSquareValues = []

        for ci1 in list1:
            if not pd.api.types.is_integer_dtype(ci1) and not pd.api.types.is_float_dtype(ci1):
                raise Exception("unsupported column type")

            for ci2 in list2:
                if not pd.api.types.is_integer_dtype(ci2) and not pd.api.types.is_float_dtype(ci2):
                    raise Exception("unsupported column type")

                if pd.api.types.is_float_dtype(ci1):
                    tempColumn1 = StatisticOperations.discretizeNumericColumn(dataset, ci1,erduo)
                else:
                    tempColumn1 = ci1

                if pd.api.types.is_float_dtype(ci2):
                    tempColumn2 = StatisticOperations.discretizeNumericColumn(dataset, ci2,erduo)
                else:
                    tempColumn2 = ci2

                chiSquareTestVal, p, dof, expected = chi2_contingency(StatisticOperations.generateDiscreteAttributesCategoryIntersection(tempColumn1,tempColumn2))

                if not np.isnan(chiSquareTestVal) and not np.isinf(chiSquareTestVal):
                    chiSquareValues.append(chiSquareTestVal)
        return chiSquareValues


    # Receives a numeric column and returns its discretized version
    @staticmethod
    def discretizeNumericColumn(dataset: Dataset, columnInfo: pd.Series,
                                discretizer: EqualRangeDiscretizerUnaryOperator):
        if (discretizer == None):
            bins = np.zeros(Properties.equalRangeDiscretizerBinsNumber)
            discretizer = EqualRangeDiscretizerUnaryOperator(bins)

        tempColumnsList = []
        tempColumnsList.append(columnInfo)
        discretizer.processTrainingSet(dataset, tempColumnsList, None)
        discretizedAttribute = discretizer.generate(dataset, tempColumnsList, None)
        return discretizedAttribute


    # Used to generate the data structure required to conduct the Chi-Square test on two data columns
    @staticmethod
    def generateDiscreteAttributesCategoryIntersection(col1: pd.Series, col2: pd.Series) -> np.ndarray:
        assert Operator.getSeriesType(col1) != outputType.Numeric
        assert Operator.getSeriesType(col2) != outputType.Numeric

        if col1.shape[0] != col2.shape[0]:
            raise Exception("Columns do not have the same number of instances")

        return StatisticOperations._generateChiSuareIntersectionMatrix(col1, col2)

    @staticmethod
    def _generateChiSuareIntersectionMatrix(col1: pd.Series, col2: pd.Series) -> np.ndarray:
        # intersectionsMatrix = np.zeros((col1NumOfValues, col2NumOfValues), dtype=np.int)
        # for i in range(col1Values.shape[0]):
        #     intersectionsMatrix[col1Values[i]][col2Values[i]] += 1
        # return intersectionsMatrix
        crosstab = pd.crosstab(col1, col2)
        return crosstab.values
