from typing import Dict, List

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_rel

from src.feature_engineering.ExploreKit.method.Evaluation.AttributeInfo import AttributeInfo
from src.feature_engineering.ExploreKit.method.Data.Dataset import Dataset
from src.feature_engineering.ExploreKit.method.Evaluation.InformationGainFilterEvaluator import InformationGainFilterEvaluator
from src.feature_engineering.ExploreKit.method.Evaluation.StatisticOperations import StatisticOperations
from src.feature_engineering.ExploreKit.method.Evaluation.OperatorAssignment import OperatorAssignment
from src.feature_engineering.ExploreKit.method.Operators import Operator
from src.feature_engineering.ExploreKit.method.Operators.UnaryOperators.UnaryOperator import UnaryOperator


class OperatorAssignmentBasedAttributes:

    def __init__(self):
        self.numOfSources: int
        self.numOfNumericSources: int
        self.numOfDiscreteSources: int = 0
        self.numOfDateSources: int = 0
        self.operatorTypeIdentifier: int # The type of the operator: unary, binary etc.
        self.operatorIdentifier: int
        self.discretizerInUse: int # 0 if none is used, otherwise the type of the discretizer (enumerated) TODO: check if this is applies before or after the operator itself
        self.normalizerInUse: int # 0 if none is used, otherwise the type of the normalizer (enumerated) TODO: check if this is applies before or after the operator itself

        # statistics on the values of discrete source attributes
        self.maxNumOfDiscreteSourceAttribtueValues: float
        self.minNumOfDiscreteSourceAttribtueValues: float
        self.avgNumOfDiscreteSourceAttribtueValues: float
        self.stdevNumOfDiscreteSourceAttribtueValues: float

        # atatistics on the values of the target attribute (currently for numeric values)
        self.maxValueOfNumericTargetAttribute: float
        self.minValueOfNumericTargetAttribute: float
        self.avgValueOfNumericTargetAttribute: float
        self.stdevValueOfNumericTargetAttribute: float

        # statistics on the value of the numeric source attribute (currently we only support cases where it's the only source attribute)
        self.maxValueOfNumericSourceAttribute: float
        self.minValueOfNumericSourceAttribute: float
        self.avgValueOfNumericSourceAttribute: float
        self.stdevValueOfNumericSourceAttribute: float

        # Paired-T amd Chi-Square tests on the source and target attributes
        self.chiSquareTestValueForSourceAttributes: float
        self.pairedTTestValueForSourceAndTargetAttirbutes: float  # used for numeric single source attirbute and numeric target

        self.maxChiSquareTsetForSourceAndTargetAttributes: float # we discretize all the numeric attributes for this one
        self.minChiSquareTsetForSourceAndTargetAttributes: float
        self.avgChiSquareTsetForSourceAndTargetAttributes: float
        self.stdevChiSquareTsetForSourceAndTargetAttributes: float

        # Calculate the similarity of the source attributes to other attibures in the dataset (discretuze all the numeric ones)
        self.maxChiSquareTestvalueForSourceDatasetAttributes: float
        self.minChiSquareTestvalueForSourceDatasetAttributes: float
        self.avgChiSquareTestvalueForSourceDatasetAttributes: float
        self.stdevChiSquareTestvalueForSourceDatasetAttributes: float

        ##########################################################
        # statistics on the generated attributes
        self.isOutputDiscrete: int #if  not, it's 0
        # If the generated attribute is discrete, count the number of possible values. If numeric, the value is set to 0
        self.numOfDiscreteValues: int

        self.IGScore: float

        # If the generated attribute is numeric, calculate the Paired T-Test statistics for it and the dataset's numeric attributes
        self.maxPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes: float = -1
        self.minPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes: float = -1
        self.avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes: float = -1
        self.stdevPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes: float = -1

        # The Chi-Squared test of the (discretized if needed) generate attribute and the dataset's discrete attributes
        self.maxChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes: float
        self.minChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes: float
        self.avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes: float
        self.stdevChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes: float

        # the Chi-Squared test of the (discretized if needed) generate attribute and ALL the dataset's attributes (discrete and numeric)
        self.maxChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes: float
        self.minChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes: float
        self.avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes: float
        self.stdevChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes: float

        ##########################################################################

        # self.probDiffScoreForTopMiscallasiffiedInstancesInMinorityClass: Dict[float, float]
        # self.probDiffScoreForTopMiscallasiffiedInstancesInMajorityClass: Dict[float, float]


    def _is_column_numeric(self, col: pd.Series):
        return pd.api.types.is_float_dtype(col)

    def _is_column_discrete(self, col: pd.Series):
        return pd.api.types.is_integer_dtype(col)

    def _is_column_date(self, col: pd.Series):
        return pd.api.types.is_datetime64_any_dtype(col)

    def _is_series_in_list(self, columns: List[pd.Series], column: pd.Series) -> bool:
        if columns == None:
            return False
        return any(column.name == col.name for col in columns)

    # Generates the meta-feautres for the "parents" of the generated attribute. These are the meta-features that DO NOT require
    # calculating the values of the generated attribute to be calculated
    def getOperatorAssignmentBasedMetaFeatures(self, dataset: Dataset, oa: OperatorAssignment) -> Dict[int,AttributeInfo]:
        try:
            # Calling the procedures that calculate the attributes of the OperatorAssignment obejct and the source and target attribtues
            try:
                self._ProcessOperatorAssignment(dataset, oa)
            except Exception as ex:
                x=5

            try:
                self._processSourceAndTargetAttributes(dataset, oa)
            except Exception as ex:
                x = 5

            try:
                self._performStatisticalTestsOnSourceAndTargetAttributes(dataset, oa)
            except Exception as ex:
                x = 5

            try:
                self._performStatisticalTestOnOperatorAssignmentAndDatasetAtributes(dataset, oa)
            except Exception as ex:
                x = 5

            return self.generateInstanceAttributesMap(True, False)

        except Exception as ex:
            return None

    # Generates the meta-features that require the values of the generated attribute in order to be calculated.
    def getGeneratedAttributeValuesMetaFeatures(self, dataset: Dataset, oa: OperatorAssignment, generatedAttribute: pd.Series) -> Dict[int, AttributeInfo]:
        try:
            datasetReplica = dataset.replicateDataset()
            datasetReplica.addColumn(generatedAttribute)
            tempList = []
            tempList.append(generatedAttribute)

            # IGScore
            try:
                igfe = InformationGainFilterEvaluator()
                igfe.initFilterEvaluator(tempList)
                self.IGScore = igfe.produceScore(datasetReplica, None, dataset, None, None)
            except Exception as ex:
                x = 5

            # Calling the procedures that calculate statistics on the candidate attribute
            try:
                self.processGeneratedAttribute(dataset, oa, generatedAttribute)
            except Exception as ex:
                x = 5

            return self.generateInstanceAttributesMap(False, True)
        except Exception:
            return None

    # @param addValuesFreeMetaFeatures If true, will add all the meta-feautres that are not reliant on the values of the generated attribute (i.e. they rely on the "parents" and the operator assignment)
    # @param addValueDependentMetaFeatures If true, will add all the meta-features that are reliant on the values of the generated attribute
    def generateInstanceAttributesMap(self, addValuesFreeMetaFeatures: bool, addValueDependentMetaFeatures: bool) -> Dict[int,AttributeInfo]:
        attributes: Dict[int,AttributeInfo] = {}

        if addValuesFreeMetaFeatures:
            try:
                attributes[len(attributes)] = AttributeInfo("numOfSources", Operator.outputType.Numeric, self.numOfSources, -1)
                attributes[len(attributes)] = AttributeInfo("numOfNumericSources", Operator.outputType.Numeric, self.numOfNumericSources, -1)
                attributes[len(attributes)] = AttributeInfo("numOfDiscreteSources", Operator.outputType.Numeric, self.numOfDiscreteSources, -1)
                attributes[len(attributes)] = AttributeInfo("numOfDateSources", Operator.outputType.Numeric, self.numOfDateSources, -1)
                attributes[len(attributes)] = AttributeInfo("operatorTypeIdentifier", Operator.outputType.Discrete, self.operatorTypeIdentifier, 4)
                attributes[len(attributes)] = AttributeInfo("operatorIdentifier", Operator.outputType.Discrete, self.operatorIdentifier, 30)
                attributes[len(attributes)] = AttributeInfo("discretizerInUse", Operator.outputType.Discrete, self.discretizerInUse, 2)
                attributes[len(attributes)] = AttributeInfo("normalizerInUse", Operator.outputType.Discrete, self.normalizerInUse, 2)
                attributes[len(attributes)] = AttributeInfo("maxNumOfDiscreteSourceAttribtueValues", Operator.outputType.Numeric, self.maxNumOfDiscreteSourceAttribtueValues, -1)
                attributes[len(attributes)] = AttributeInfo("minNumOfDiscreteSourceAttribtueValues", Operator.outputType.Numeric, self.minNumOfDiscreteSourceAttribtueValues, -1)
                attributes[len(attributes)] = AttributeInfo("avgNumOfDiscreteSourceAttribtueValues", Operator.outputType.Numeric, self.avgNumOfDiscreteSourceAttribtueValues, -1)
                attributes[len(attributes)] = AttributeInfo("stdevNumOfDiscreteSourceAttribtueValues", Operator.outputType.Numeric, self.stdevNumOfDiscreteSourceAttribtueValues, -1)
                attributes[len(attributes)] = AttributeInfo("maxValueOfNumericTargetAttribute", Operator.outputType.Numeric, self.maxValueOfNumericTargetAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("minValueOfNumericTargetAttribute", Operator.outputType.Numeric, self.minValueOfNumericTargetAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("avgValueOfNumericTargetAttribute", Operator.outputType.Numeric, self.avgValueOfNumericTargetAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("stdevValueOfNumericTargetAttribute", Operator.outputType.Numeric, self.stdevValueOfNumericTargetAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("maxValueOfNumericSourceAttribute", Operator.outputType.Numeric, self.maxValueOfNumericSourceAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("minValueOfNumericSourceAttribute", Operator.outputType.Numeric, self.minValueOfNumericSourceAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("avgValueOfNumericSourceAttribute", Operator.outputType.Numeric, self.avgValueOfNumericSourceAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("stdevValueOfNumericSourceAttribute", Operator.outputType.Numeric, self.stdevValueOfNumericSourceAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("chiSquareTestValueForSourceAttributes", Operator.outputType.Numeric, self.chiSquareTestValueForSourceAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("pairedTTestValueForSourceAndTargetAttirbutes", Operator.outputType.Numeric, self.pairedTTestValueForSourceAndTargetAttirbutes, -1)
                attributes[len(attributes)] = AttributeInfo("maxChiSquareTsetForSourceAndTargetAttributes", Operator.outputType.Numeric, self.maxChiSquareTsetForSourceAndTargetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("minChiSquareTsetForSourceAndTargetAttributes", Operator.outputType.Numeric, self.minChiSquareTsetForSourceAndTargetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("avgChiSquareTsetForSourceAndTargetAttributes", Operator.outputType.Numeric, self.avgChiSquareTsetForSourceAndTargetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("stdevChiSquareTsetForSourceAndTargetAttributes", Operator.outputType.Numeric, self.stdevChiSquareTsetForSourceAndTargetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("maxChiSquareTestvalueForSourceDatasetAttributes", Operator.outputType.Numeric, self.maxChiSquareTestvalueForSourceDatasetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("minChiSquareTestvalueForSourceDatasetAttributes", Operator.outputType.Numeric, self.minChiSquareTestvalueForSourceDatasetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("avgChiSquareTestvalueForSourceDatasetAttributes", Operator.outputType.Numeric, self.avgChiSquareTestvalueForSourceDatasetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("stdevChiSquareTestvalueForSourceDatasetAttributes", Operator.outputType.Numeric, self.stdevChiSquareTestvalueForSourceDatasetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("isOutputDiscrete", Operator.outputType.Discrete, self.isOutputDiscrete, 2)
                attributes[len(attributes)] = AttributeInfo("numOfDiscreteValues", Operator.outputType.Numeric, self.numOfDiscreteValues, -1) # TODO: in the future, this one will have to move to the other group

            except Exception:
                x = 5

        if addValueDependentMetaFeatures:
            try:
                attributes[len(attributes)] = AttributeInfo("IGvalue", Operator.outputType.Numeric, self.IGScore, -1)
                attributes[len(attributes)] = AttributeInfo("probDiffScore", Operator.outputType.Numeric, -1, -1)
                attributes[len(attributes)] = AttributeInfo("maxPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes", Operator.outputType.Numeric, self.maxPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("minPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes", Operator.outputType.Numeric, self.minPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes", Operator.outputType.Numeric, self.avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("stdevPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes", Operator.outputType.Numeric, self.stdevPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("maxChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes", Operator.outputType.Numeric, self.maxChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("minChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes", Operator.outputType.Numeric, self.minChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes", Operator.outputType.Numeric, self.avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("stdevChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes", Operator.outputType.Numeric, self.stdevChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("maxChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes", Operator.outputType.Numeric, self.maxChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("minChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes", Operator.outputType.Numeric, self.minChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes", Operator.outputType.Numeric, self.avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("stdevChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes", Operator.outputType.Numeric, self.stdevChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes, -1)

            except Exception:
                x = 5

        return attributes

    # Used to calculate statistics on the correlation of the generates attribute and the attributes of the dataset.
    # The attributes that were used to generate the feature are excluded.
    def processGeneratedAttribute(self, dataset: Dataset, oa: OperatorAssignment, generatedAttribute: pd.Series):
        # IMPORTANT: make sure that the source and target attributes are not included in these calculations
        discreteColumns: List[pd.Series] = dataset.getAllColumnsOfType(self._is_column_discrete, False)
        numericColumns: List[pd.Series] = dataset.getAllColumnsOfType(self._is_column_numeric, False)

        # The paired T-Tests for the dataset's numeric attributes
        if Operator.Operator.getSeriesType(generatedAttribute) == Operator.outputType.Numeric:
            pairedTTestScores = StatisticOperations.calculatePairedTTestValues(self._filterOperatorAssignmentAttributes(numericColumns, oa), generatedAttribute)
            if len(pairedTTestScores) > 0:
                self.maxPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = np.max(pairedTTestScores)
                self.minPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = np.min(pairedTTestScores)
                self.avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = np.mean(pairedTTestScores)
                self.stdevPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = np.std(pairedTTestScores)
            else:
                self.maxPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = 0
                self.minPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = 0
                self.avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = 0
                self.stdevPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = 0

        # The chi Squared test for the dataset's dicrete attribtues
        chiSquareTestsScores = StatisticOperations.calculateChiSquareTestValues(self._filterOperatorAssignmentAttributes(discreteColumns,oa),[generatedAttribute],dataset)
        if len(chiSquareTestsScores) > 0:
            self.maxChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = np.max(chiSquareTestsScores)
            self.minChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = np.min(chiSquareTestsScores)
            self.avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = np.mean(chiSquareTestsScores)
            self.stdevChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = np.std(chiSquareTestsScores)
        else:
            self.maxChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = 0
            self.minChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = 0
            self.avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = 0
            self.stdevChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = 0

        # The Chi Square test for ALL the dataset's attirbutes (Numeric attributes will be discretized)
        discreteColumns.extend(numericColumns)
        AllAttributesChiSquareTestsScores = StatisticOperations.calculateChiSquareTestValues(self._filterOperatorAssignmentAttributes(discreteColumns,oa),[generatedAttribute],dataset)
        if len(AllAttributesChiSquareTestsScores) > 0:
            self.maxChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = np.max(AllAttributesChiSquareTestsScores)
            self.minChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = np.min(AllAttributesChiSquareTestsScores)
            self.avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = np.mean(AllAttributesChiSquareTestsScores)
            self.stdevChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = np.std(AllAttributesChiSquareTestsScores)
        else:
            self.maxChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = 0
            self.minChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = 0
            self.avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = 0
            self.stdevChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes =  0

    # Receives a list of attirbutes and an OperatorAssignment object. The function filters out the source and target
    # attribtues of the OperatorAssignmetn from the given list.
    def _filterOperatorAssignmentAttributes(self, attributesList: List[pd.Series], operatorAssignment: OperatorAssignment) -> List[pd.Series]:
        listToReturn: List[pd.Series] = []
        for ci in attributesList:
            if any(ci.name == col.name for col in operatorAssignment.getSources()):
                continue

            if operatorAssignment.getTargets() != None and any(ci.name == col.name for col in operatorAssignment.getTargets()):
                continue

            listToReturn.append(ci)
        return listToReturn

    #region process_operator_no_evaluation

    # Analyzes the characteristics of the OperatorAssignment object - the characteristics of the feature
    # that make up the object. Here we do not process the analyzed attribute itself.
    def _ProcessOperatorAssignment(self, dataset: Dataset, oa: OperatorAssignment):
        # numOfSources
        if oa.getSources() != None:
            self.numOfSources = len(oa.getSources())
        else:
            self.numOfSources = 0

        # numOfNumericSources + numOfDiscreteSources + numOfDateSources
        self.numOfNumericSources = 0
        if oa.getSources() != None:
            for ci in oa.getSources():
                if self._is_column_numeric(ci):
                    self.numOfNumericSources += 1

                if self._is_column_discrete(ci):
                    self.numOfDiscreteSources += 1

                if self._is_column_date(ci):
                    self.numOfDateSources += 1
         # operatorType
        self.operatorTypeIdentifier = self._getOperatorTypeID(oa.getOperator().getType())

         # operatorName
        self.operatorIdentifier = self._GetOperatorIdentifier(oa.getOperator())

         # isOutputDiscrete
        if oa.getSecondaryOperator() != None:
            if ((oa.getSecondaryOperator().getOutputType().equals(Operator.outputType.Discrete))):
                self.isOutputDiscrete = 1
            else:
                self.isOutputDiscrete = 0
        else:
            if oa.getOperator().getOutputType() == Operator.outputType.Discrete:
                self.isOutputDiscrete = 1
            else:
                self.isOutputDiscrete = 0

        self.discretizerInUse = self._getDiscretizerID(oa.getSecondaryOperator())

        self.normalizerInUse = self._getNormalizerID(oa.getSecondaryOperator())

        self.numOfDiscreteValues = self._getNumOfNewAttributeDiscreteValues(oa)

    def _getNumOfNewAttributeDiscreteValues(self, oa: OperatorAssignment) -> int:
        # TODO: the current code assumes that the only way for a generated attribute to have discrete values is to be generated using the Discretizer unary operator. This will have to be modified as we expand the system. Moreover, we will have to see if this can remain in the part of the code that calculates the meta-features without generating the attributes values

        if oa.getSecondaryOperator() is not None:
            return oa.getSecondaryOperator().getNumOfBins()
        else:
            if oa.getOperator().getOutputType() != Operator.outputType.Discrete:
                return -1
            else:
                # currently the only operators which return a discrete value are the Unary.
                return (oa.getOperator()).getNumOfBins()

    # Returns an integer that represents the type of the operator in use
    def _getOperatorTypeID(self, operatorType: Operator.operatorType) -> int:
        if operatorType == Operator.operatorType.Unary:
            return 1

        if operatorType == Operator.operatorType.Binary:
            return 2

        if operatorType == Operator.operatorType.GroupByThen:
            return 3

        if operatorType == Operator.operatorType.TimeBasedGroupByThen:
            return 4

        raise Exception("Unrecognized operator type")

    def _GetOperatorIdentifier(self, operator: Operator) -> int:
        if operator == None:
            return 0

        op_name = operator.getName()
        if "_" in op_name:
            op_name = op_name[0: op_name.indexOf("_")]

        op_mapping = {
            "EqualRangeDiscretizerUnaryOperator": 1,
            "DayOfWeekUnaryOperator": 2,
            "HourOfDayUnaryOperator": 3,
            "IsWeekendUnaryOperator": 4,
            "StandardScoreUnaryOperator": 5,
            "AddBinaryOperator": 6,
            "DivisionBinaryOperator": 7,
            "MultiplyBinaryOperator": 8,
            "SubtractBinaryOperator": 9,
            "GroupByThenAvg": 10,
            "GroupByThenCount": 11,
            "GroupByThenMax": 12,
            "GroupByThenMin": 13,
            "GroupByThenStdev": 14,
            "TimeBasedGroupByThenCountAndAvg": 15,
            "TimeBasedGroupByThenCountAndCount": 16,
            "TimeBasedGroupByThenCountAndMax": 17,
            "TimeBasedGroupByThenCountAndMin": 18,
            "TimeBasedGroupByThenCountAndStdev": 19
        }
        if op_name not in op_mapping.keys():
            raise Exception("Unidentified operator in use")

        return op_mapping[op_name]

    def _getDiscretizerID(self, uo: UnaryOperator) -> int:
        if uo == None:
            return 0

        try:
            return {
                "EqualRangeDiscretizerUnaryOperator": 1,
                "DayOfWeekUnaryOperator": 2,
                "HourOfDayUnaryOperator": 3,
                "IsWeekendUnaryOperator": 4
            }[uo.getName()]
        except:
            # we can get here because even if the opertaor is not null, it may be a normalizer and not a discretizer
            return 0

    def _getNormalizerID(self, uo: UnaryOperator) -> int:
        if uo == None:
            return 0

        if uo.getName() == "StandardScoreUnaryOperator":
             return 1
        else:
             return 0

    #endregion

    def _processSourceAndTargetAttributes(self, dataset: Dataset,  oa: OperatorAssignment):
        # start by computing statistics on the discrete source attributes
        sourceAttributesValuesList: List[float] = []
        for sourceAttribute in oa.getSources():
            if self._is_column_discrete(sourceAttribute):
                sourceAttributesValuesList.append(sourceAttribute.nunique())

        if len(sourceAttributesValuesList) == 0:
            self.maxNumOfDiscreteSourceAttribtueValues = 0
            self.minNumOfDiscreteSourceAttribtueValues = 0
            self.avgNumOfDiscreteSourceAttribtueValues = 0
            self.stdevNumOfDiscreteSourceAttribtueValues = 0
        else:
            sourceAttributesValuesNd = np.array(sourceAttributesValuesList)
            self.maxNumOfDiscreteSourceAttribtueValues = sourceAttributesValuesNd.max()
            self.minNumOfDiscreteSourceAttribtueValues = sourceAttributesValuesNd.min()
            self.avgNumOfDiscreteSourceAttribtueValues = sourceAttributesValuesNd.mean()
            self.stdevNumOfDiscreteSourceAttribtueValues = sourceAttributesValuesNd.std()

        # Statistics on numeric target attribute (we currently support a single attribute)
        if oa.getTargets() == None or not self._is_column_numeric(oa.getTargets()[0]):
            self.maxValueOfNumericTargetAttribute = 0
            self.minValueOfNumericTargetAttribute = 0
            self.avgValueOfNumericTargetAttribute = 0
            self.stdevValueOfNumericTargetAttribute = 0
        else:
            numericTargetAttributeValues = oa.getTargets()[0]
            self.maxValueOfNumericTargetAttribute = numericTargetAttributeValues.max()
            self.minValueOfNumericTargetAttribute = numericTargetAttributeValues.min()
            self.avgValueOfNumericTargetAttribute = numericTargetAttributeValues.mean()
            self.stdevValueOfNumericTargetAttribute = numericTargetAttributeValues.std()

        if not self._is_column_numeric(oa.getSources()[0]):
            self.maxValueOfNumericSourceAttribute = 0
            self.minValueOfNumericSourceAttribute = 0
            self.avgValueOfNumericSourceAttribute = 0
            self.stdevValueOfNumericSourceAttribute = 0
        else:
            numericSourceAttributeValues = oa.getSources()[0]
            self.maxValueOfNumericSourceAttribute = numericSourceAttributeValues.max()
            self.minValueOfNumericSourceAttribute = numericSourceAttributeValues.min()
            self.avgValueOfNumericSourceAttribute = numericSourceAttributeValues.mean()
            self.stdevValueOfNumericSourceAttribute = numericSourceAttributeValues.mean()

    def _performStatisticalTestsOnSourceAndTargetAttributes(self, dataset: Dataset, oa: OperatorAssignment):
        # Chi Square test for discrete source attributes
        self.chiSquareTestValueForSourceAttributes = 0
        if len(oa.getSources()) == 2:
            if self._is_column_discrete(oa.getSources()[0]) and self._is_column_discrete(oa.getSources()[1]):
                dc1 = oa.getSources()[0]
                dc2 = oa.getSources()[1]

                tempVal, p, dof, expected = chi2_contingency(StatisticOperations.generateDiscreteAttributesCategoryIntersection(dc1,dc2))
                if (not np.isnan(tempVal) and not np.isinf(tempVal)):
                    self.chiSquareTestValueForSourceAttributes = tempVal
                else:
                    self.chiSquareTestValueForSourceAttributes = -1

        # Paired T-Test for numeric source and target
        self.pairedTTestValueForSourceAndTargetAttirbutes = 0
        if len(oa.getSources()) == 1 and self._is_column_numeric(oa.getSources()[0]) and oa.getTargets() != None and len(oa.getTargets()) == 1:
            (statistic, pvalue) = ttest_rel(oa.getSources()[0], oa.getTargets()[0])
            self.pairedTTestValueForSourceAndTargetAttirbutes = pvalue

        # The chiSquare Test scores of all source and target attribtues (numeric atts are discretized, other non-discrete types are ignored)
        if len(oa.getSources()) == 1 and oa.getTargets() == None:
            self.maxChiSquareTsetForSourceAndTargetAttributes = 0
            self.minChiSquareTsetForSourceAndTargetAttributes = 0
            self.avgChiSquareTsetForSourceAndTargetAttributes = 0
            self.stdevChiSquareTsetForSourceAndTargetAttributes = 0
        else:
            columnsToAnalyze = []
            for ci in oa.getSources():
                if self._is_column_discrete(ci):
                    columnsToAnalyze.append(ci)
                else:
                    if self._is_column_numeric(ci):
                        columnsToAnalyze.append(StatisticOperations.discretizeNumericColumn(dataset, ci, None))

            if len(columnsToAnalyze) > 1:
                chiSquareTestValues = []
                for i in range(len(columnsToAnalyze) - 1):
                    for j in range(i+1, len(columnsToAnalyze)):
                        chiSquareTestVal, p, dof, expected = chi2_contingency(StatisticOperations.generateDiscreteAttributesCategoryIntersection(
                                columnsToAnalyze[i], columnsToAnalyze[j]))
                        if not np.isnan(chiSquareTestVal) and not np.isinf(chiSquareTestVal):
                            chiSquareTestValues.append(chiSquareTestVal)

                if len(chiSquareTestValues) > 0:
                    chiSquareTestNd = np.array(chiSquareTestValues)
                    self.maxChiSquareTsetForSourceAndTargetAttributes = chiSquareTestNd.max()
                    self.minChiSquareTsetForSourceAndTargetAttributes = chiSquareTestNd.max()
                    self.avgChiSquareTsetForSourceAndTargetAttributes = chiSquareTestNd.mean()
                    self.stdevChiSquareTsetForSourceAndTargetAttributes = chiSquareTestNd.std()
                else:
                    self.maxChiSquareTsetForSourceAndTargetAttributes = 0
                    self.minChiSquareTsetForSourceAndTargetAttributes = 0
                    self.avgChiSquareTsetForSourceAndTargetAttributes = 0
                    self.stdevChiSquareTsetForSourceAndTargetAttributes = 0

    def _performStatisticalTestOnOperatorAssignmentAndDatasetAtributes(self, dataset: Dataset, oa: OperatorAssignment):
        # first we put all the OA attributes (sources and targets) in one list. Numeric atts are discretized, other non-discretes are ignored
        columnsToAnalyze: List[pd.Series] = []
        for ci in oa.getSources():
            if self._is_column_discrete(ci):
                columnsToAnalyze.append(ci)
            else:
                if self._is_column_numeric(ci):
                    columnsToAnalyze.append(StatisticOperations.discretizeNumericColumn(dataset, ci, None))

        if oa.getTargets() != None:
            for ci in oa.getTargets():
                if self._is_column_discrete(ci):
                    columnsToAnalyze.append(ci)
                else:
                    if self._is_column_numeric(ci):
                        columnsToAnalyze.append(StatisticOperations.discretizeNumericColumn(dataset, ci, None))

        # For each attribute in the list we created, we iterate over all the attributes in the dataset (all those that are not in the OA)
        chiSquareTestValues: List[float] = []
        for ci in columnsToAnalyze:
            for _, datasetCI in dataset.getAllColumns(False).items():
                # if datasetCI is in the OA then skip
                if self._is_series_in_list(oa.getSources(), datasetCI) or self._is_series_in_list(oa.getTargets(), datasetCI):
                    continue

                chiSquareTestValue = 0
                if self._is_column_date(datasetCI) or pd.api.types.is_string_dtype(datasetCI):
                    continue
                if self._is_column_discrete(datasetCI):
                    chiSquareTestValue, p, dof, expected = chi2_contingency(StatisticOperations.generateDiscreteAttributesCategoryIntersection(ci,datasetCI))

                if self._is_column_numeric(datasetCI):
                    tempCI = StatisticOperations.discretizeNumericColumn(dataset, datasetCI,None)
                    chiSquareTestValue, p, dof, expected = chi2_contingency(StatisticOperations.generateDiscreteAttributesCategoryIntersection(ci,tempCI))

                if not np.isnan(chiSquareTestValue) and not np.isinf(chiSquareTestValue):
                    chiSquareTestValues.append(chiSquareTestValue)

        # now we calculate the max/min/avg/stdev
        if len(chiSquareTestValues) > 0:
            chiSquareTestValuesNd = np.array(chiSquareTestValues)
            self.maxChiSquareTestvalueForSourceDatasetAttributes = chiSquareTestValuesNd.max()
            self.minChiSquareTestvalueForSourceDatasetAttributes = chiSquareTestValuesNd.min()
            self.avgChiSquareTestvalueForSourceDatasetAttributes = chiSquareTestValuesNd.mean()
            self.stdevChiSquareTestvalueForSourceDatasetAttributes = np.std(chiSquareTestValuesNd)
        else:
            self.maxChiSquareTestvalueForSourceDatasetAttributes = 0
            self.minChiSquareTestvalueForSourceDatasetAttributes = 0
            self.avgChiSquareTestvalueForSourceDatasetAttributes = 0
            self.stdevChiSquareTestvalueForSourceDatasetAttributes = 0






