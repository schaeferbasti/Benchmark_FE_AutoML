import time
from builtins import staticmethod
from typing import List, Dict
from itertools import combinations

from src.feature_engineering.ExploreKit.method.Evaluation.AucWrapperEvaluator import AucWrapperEvaluator
from src.feature_engineering.ExploreKit.method.Data.Dataset import Dataset
from src.feature_engineering.ExploreKit.method.Evaluation.ClassificationResults import ClassificationResults
from src.feature_engineering.ExploreKit.method.Evaluation.FilterEvaluator import FilterEvaluator
from src.feature_engineering.ExploreKit.method.Evaluation.InformationGainFilterEvaluator import InformationGainFilterEvaluator
from src.feature_engineering.ExploreKit.method.Evaluation.OperationAssignmentAncestorsSingleton import OperationAssignmentAncestorsSingleton
from src.feature_engineering.ExploreKit.method.Evaluation.OperatorAssignment import OperatorAssignment
from src.feature_engineering.ExploreKit.method.Operators.BinaryOperators.DivsionBinaryOperator import DivisionBinaryOperator
from src.feature_engineering.ExploreKit.method.Operators.BinaryOperators.MultiplyBinaryOperator import MultiplyBinaryOperator
from src.feature_engineering.ExploreKit.method.Operators.BinaryOperators.SubtractBinaryOperator import SubtractBinaryOperator
from src.feature_engineering.ExploreKit.method.Operators.GroupByThenOperators.GroupByThenAvg import GroupByThenAvg
from src.feature_engineering.ExploreKit.method.Operators.GroupByThenOperators.GroupByThenCount import GroupByThenCount
from src.feature_engineering.ExploreKit.method.Operators.GroupByThenOperators.GroupByThenMax import GroupByThenMax
from src.feature_engineering.ExploreKit.method.Operators.GroupByThenOperators.GroupByThenMin import GroupByThenMin
from src.feature_engineering.ExploreKit.method.Operators.GroupByThenOperators.GroupByThenStdev import GroupByThenStdev
from src.feature_engineering.ExploreKit.method.Utils import Parallel
from src.feature_engineering.ExploreKit.method.Utils.Logger import Logger
from src.feature_engineering.ExploreKit.method.Operators.BinaryOperators.AddBinaryOperator import AddBinaryOperator
from src.feature_engineering.ExploreKit.method.Operators.UnaryOperators.EqualRangeDiscretizerUnaryOperator import EqualRangeDiscretizerUnaryOperator
from src.feature_engineering.ExploreKit.method.Operators.Operator import Operator, operatorType
from src.feature_engineering.ExploreKit.method.Operators.UnaryOperators.UnaryOperator import UnaryOperator
from src.feature_engineering.ExploreKit.method.Properties import Properties

import pandas as pd

class OperatorsAssignmentsManager:

    # a new copy of the provided operator
    @staticmethod
    def getOperator(operator: Operator)-> Operator:
        if operator.getType() == operatorType.Unary:
            return OperatorsAssignmentsManager.getUnaryOperator(operator.getName())
        return OperatorsAssignmentsManager.getNonUnaryOperator(operator.getName())

    # Returns a list of unary operators from the configuration file
    @staticmethod
    def getUnaryOperatorsList():
        operatorNames = Properties.unaryOperators.split(",")
        unaryOperatorsList = []
        for unaryOperator in operatorNames:
            #TODO: temporal try catch - remove when use all operators
            try:
                uo = OperatorsAssignmentsManager.getUnaryOperator(unaryOperator)
                unaryOperatorsList.append(uo)
            except:
                pass
        return unaryOperatorsList

    # Returns an unary operator by name
    @staticmethod
    def getUnaryOperator(operatorName: str) -> UnaryOperator:
        if operatorName == "EqualRangeDiscretizerUnaryOperator":
            bins = [0] * int(Properties.equalRangeDiscretizerBinsNumber)
            erd = EqualRangeDiscretizerUnaryOperator(bins)
            return erd
        # elif operatorName == "StandardScoreUnaryOperator":
        #     ssuo = StandardScoreUnaryOperator()
        #     return ssuo
        # elif operatorName == "DayOfWeekUnaryOperator":
        #     dowuo = DayOfWeekUnaryOperator()
        #     return dowuo
        # elif operatorName == "HourOfDayUnaryOperator":
        #     hoduo = HourOfDayUnaryOperator()
        #     return hoduo
        # elif operatorName == "IsWeekendUnaryOperator":
        #     iwuo = IsWeekendUnaryOperator()
        #     return iwuo
        else:
            raise Exception("unindentified unary operator: " + operatorName)


    # Returns a list of nonUnary operators from the configuration file (i.e. all other operator types)
    @staticmethod
    def getNonUnaryOperatorsList():
        operatorNames = Properties.nonUnaryOperators.split(',')
        operatorsList = []
        for unaryOperator in operatorNames:
            try:
                operator = OperatorsAssignmentsManager.getNonUnaryOperator(unaryOperator)
                operatorsList.append(operator)
            except:
                pass

        return operatorsList


    # Returns a non-unary operator by name
    @staticmethod
    def getNonUnaryOperator(operatorName: str):
        timeSpan = 0
        # if operatorName.startswith("TimeBasedGroupByThen"):
        #     timeSpan = float(operatorName.split("_")[1])
        #     operatorName = operatorName.split("_")[0]

        # switch (operatorName) {
        # GroupByThenOperators
        if operatorName == "GroupByThenAvg":
            gbtAvg = GroupByThenAvg()
            return gbtAvg
        elif operatorName == "GroupByThenMax":
            gbtMmax = GroupByThenMax()
            return gbtMmax
        elif operatorName == "GroupByThenMin":
            gbtMin = GroupByThenMin()
            return gbtMin
        elif operatorName == "GroupByThenCount":
            gbtCount = GroupByThenCount()
            return gbtCount
        elif operatorName == "GroupByThenStdev":
            gbtStdev = GroupByThenStdev()
            return gbtStdev

        # BinaryOperators
        if operatorName == "AddBinaryOperator":
            abo = AddBinaryOperator()
            return abo
        elif operatorName == "SubtractBinaryOperator":
            sbo = SubtractBinaryOperator()
            return sbo
        elif operatorName == "MultiplyBinaryOperator":
            mbo = MultiplyBinaryOperator()
            return mbo
        elif operatorName == "DivisionBinaryOperator":
            dbo = DivisionBinaryOperator()
            return dbo
        else:
            raise Exception("unindentified unary operator: " + operatorName)

    # Activates the applyOperatorsAndPerformInitialEvaluation function, but only for Unary Operators
    # @param mustIncluseAttributes Attributes which must be in either the source or the target of every generated feature
    @staticmethod
    def applyUnaryOperators(dataset: Dataset, mustIncluseAttributes, filterEvaluator: FilterEvaluator,
                subFoldTrainingDatasets: List[Dataset], currentScores: List[ClassificationResults]) -> List[OperatorAssignment]:

        unaryOperatorsList = OperatorsAssignmentsManager.getUnaryOperatorsList()
        return OperatorsAssignmentsManager.applyOperatorsAndPerformInitialEvaluation(dataset, unaryOperatorsList,
                                                                                     mustIncluseAttributes, 1,
                                                                                     filterEvaluator, None,
                                                                                     subFoldTrainingDatasets,
                                                                                     currentScores, False)

    # Activates the applyOperatorsAndPerformInitialEvaluation function, for all operator types by Unary
    # @param mustIncluseAttributes Attributes which must be in either the source or the target of every generated feature
    @staticmethod
    def applyNonUnaryOperators(dataset: Dataset, mustIncluseAttributes: List, preRankerEvaluator,
                filterEvaluator: FilterEvaluator, subFoldTrainingDatasets:List[Dataset],  currentScores: List[ClassificationResults]) -> List[OperatorAssignment]:
        nonUnaryOperatorsList = OperatorsAssignmentsManager.getNonUnaryOperatorsList()
        return OperatorsAssignmentsManager.applyOperatorsAndPerformInitialEvaluation(dataset, nonUnaryOperatorsList,mustIncluseAttributes,
                Properties.maxNumOfAttsInOperatorSource, filterEvaluator, preRankerEvaluator, subFoldTrainingDatasets, currentScores, True)

    # Receives a dataset with a set of attributes and a list of operators and generates all possible source/target/operator/secondary operator assignments
     # @param dataset The dataset with the attributes that need to be analyzed
     # @param attributesToInclude A list of attributes that must be included in either the source or target of every generated assignment. If left empty, there are no restrictions
     # @param operators A list of all the operators whose assignment will be considered
     # @param maxCombinationSize the maximal number of attributes that can be a in the source of each operator. Smaller number (down to 1) are also considered
    @staticmethod
    def getOperatorAssignments(dataset: Dataset, attributesToInclude: list, operators: List[Operator], maxCombinationSize: int):
        areNonUniaryOperatorsBeingUsed = False
        if len(operators) > 0 and not operators[0].getType() == operatorType.Unary:
            areNonUniaryOperatorsBeingUsed = True

        if attributesToInclude == None: attributesToInclude = []
        operatorsAssignments = []
        for i in range(maxCombinationSize, 0, -1): # (int i=maxCombinationSize; i>0; i--) {
            # List<List<ColumnInfo>>
            sourceAttributeCombinations = OperatorsAssignmentsManager.getAttributeCombinations(dataset.getAllColumns(False), i)

            # for each of the candidate source attributes combinations
            for listOfAttrs in sourceAttributeCombinations:
                sources: List[pd.Series] = [dataset.df[colName] for colName in listOfAttrs]
                # if a distinct dolumn(s) exists, we need to make sure that at least one column (or one of its ancestors) satisfies the constraint
                # ignore - distinct value
                # if dataset.getDistinctValueColumns() != None and len(dataset.getDistinctValueColumns()) > 0:
                #     if areNonUniaryOperatorsBeingUsed and not isDistinctValueCompliantAttributeExists(dataset.getDistinctValueCompliantColumns(), sources)) {
                #         continue;
                #     }
                # }

                # first check if any of the required atts (if there are any) are included
                if len(attributesToInclude) > 0:
                    tempList = sources.copy()
                    tempList = [item for item in tempList if item in attributesToInclude] #TODO: change to item.name
                    if len(tempList) == 0:
                        continue

                # Now we check all the operators on the source attributes alone.
                for operator in operators:
                    if operator.isApplicable(dataset, sources, None):
                        os = OperatorAssignment(sources, None, OperatorsAssignmentsManager.getOperator(operator), None)
                        operatorsAssignments.append(os)

                    # now we pair the source attributes with a target attribute and check again
                    for targetColumnName, targetColumn in dataset.getAllColumns(False).items():
                        # if (sources.contains(targetColumn)) { continue; }
                        if OperatorsAssignmentsManager.overlapExistsBetweenSourceAndTargetAttributes(sources,targetColumn): continue
                        tempList = []
                        tempList.append(targetColumn)
                        if operator.isApplicable(dataset, sources, tempList):
                            os = OperatorAssignment(sources, tempList, OperatorsAssignmentsManager.getOperator(operator), None)
                            operatorsAssignments.append(os)

        # Finally, we go over all the operator assignments. For every assignment that is not performed on
        # an unary operator, we check if any of the unary operators can be applied on it.
        additionalAssignments = []
        for os in operatorsAssignments:
            if os.getOperator().getType() != operatorType.Unary:
                for operator in OperatorsAssignmentsManager.getUnaryOperatorsList():
                    if operator.getType() == operatorType.Unary:
                        tempOperator = operator
                        if tempOperator.requiredInputType() == os.getOperator().getOutputType():
                            additionalAssignment = OperatorAssignment(os.getSources(), os.getTargets(), os.getOperator(), tempOperator)
                            additionalAssignments.append(additionalAssignment)


        operatorsAssignments.extend(additionalAssignments)
        return operatorsAssignments

    @staticmethod
    def overlapExistsBetweenSourceAndTargetAttributes(sourceAtts, targetAtt) -> bool:
        # the simplest case - the same attribute appears both in the source and the target
        sourceAttsNames = [series.name for series in sourceAtts]
        if targetAtt.name in sourceAttsNames:
            return True

        oaAncestors = OperationAssignmentAncestorsSingleton()
        # Now we need to check that the source atts and the target att has no shared columns (including after the application of an operator)
        sourceAttsAndAncestorsNames = []
        for sourceAtt in sourceAttsNames:
            sourceAttsAndAncestorsNames.append(sourceAtt)
            if oaAncestors.getSources(sourceAtt)[0]:
                for ancestorAtt in oaAncestors.getSources(sourceAtt)[1]:
                    # if (!sourceAttsAndAncestors.contains(ancestorAtt)) {
                    if ancestorAtt.name not in sourceAttsAndAncestorsNames:
                        sourceAttsAndAncestorsNames.append(ancestorAtt.name)

            if oaAncestors.getTargets(sourceAtt)[0]:
                for ancestorAtt in oaAncestors.getTargets(sourceAtt)[1]:
                    # if (!sourceAttsAndAncestors.contains(ancestorAtt)) {
                    if ancestorAtt.name not in sourceAttsAndAncestorsNames:
                        sourceAttsAndAncestorsNames.append(ancestorAtt.name)

        # do the same for the target att (because we only have one we don't need the external loop)
        targetAttsAndAncestorsNames = []
        targetAttsAndAncestorsNames.append(targetAtt.name)
        if oaAncestors.getSources(targetAtt.name)[0]:
            for ancestorAtt in oaAncestors.getSources(targetAtt.name)[1]:
                # if (!targetAttsAndAncestors.contains(ancestorAtt)) {
                if ancestorAtt.name not in targetAttsAndAncestorsNames:
                    targetAttsAndAncestorsNames.append(ancestorAtt.name)

        if oaAncestors.getTargets(targetAtt.name)[0]:
            for ancestorAtt in oaAncestors.getTargets(targetAtt.name)[1]:
                # if (!targetAttsAndAncestors.contains(ancestorAtt)) {
                if ancestorAtt.name not in targetAttsAndAncestorsNames:
                    targetAttsAndAncestorsNames.append(ancestorAtt.name)

        # boolean overlap =  !Collections.disjoint(sourceAttsAndAncestors, targetAttsAndAncestors);
        if len(set(sourceAttsAndAncestorsNames).intersection(set(targetAttsAndAncestorsNames))) > 0:
            overlap = True
        else:
            overlap = False

        #Todo: what is it mean
        if overlap and len(targetAttsAndAncestorsNames) > 1:
            x=5

        return overlap

    # Returns lists of column-combinations
    @staticmethod
    def getAttributeCombinations(attributes: pd.DataFrame, numOfAttributesInCombination: int) -> List[tuple]:
        attributeCombinations = list(combinations(attributes.columns.values, numOfAttributesInCombination))
        # attributeCombinations = []
        # gen = CombinationGenerator(attributes.shape[1], numOfAttributesInCombination)
        # while gen.hasMore():
        #     indices = gen.getNext()
        #     tempColumns = []
        #     for index in indices:
        #         tempColumns.append(attributes[index])
        #     attributeCombinations.append(tempColumns)

        return attributeCombinations


     # Receives a dataset and a list of OperatorAssignment objects, generates/gets them from file and
     # adds them to the dataset
    @staticmethod
    def GenerateAndAddColumnToDataset(self, dataset:Dataset, oaList:List[OperatorAssignment]):
        for oa in oaList:
            ci = OperatorsAssignmentsManager.generateColumn(dataset, oa, True)
            dataset.addColumn(ci)

     # Receives a a dataset and a list of operators, finds all possible combinations, generates and writes the attributes to file
     # and returns the assignments list
     # @param dataset The full dataset. The new attribute generated for it is the one to be saved to file
     # @param operators The operators for which assignments will be generated
     # @param mustIncluseAttributes The attributes that must be present in EITHER the source or the target. Empty lists or null mean there's no restriction
     # @param maxNumOfSourceAttributes The maximal number of attributes that can be in the source (if the operator permits). Smaller number down to 1 (including) will also be generated
     # @param filterEvaluator The filter evaluator that will be used to compute the initial ranking of the attriubte. The calculation is carried out on the sibfolds
     # @param subFoldTrainingDatasets The training set sub-folds. Used in order to calculate the score, as the test set cannot be used for this purpose here.
    @staticmethod
    def applyOperatorsAndPerformInitialEvaluation(dataset: Dataset, operators: List[Operator], mustIncluseAttributes,
                maxNumOfSourceAttributes: int, filterEvaluator: FilterEvaluator, preRankerEvaluator,
                subFoldTrainingDatasets:List[Dataset], currentScores:List[ClassificationResults], reduceNumberOfAttributes:bool) -> List[OperatorAssignment]:

        # in case the number of initial attributes is very high, we need narrow the search space
        if (reduceNumberOfAttributes and (mustIncluseAttributes == None or len(mustIncluseAttributes) == 0)):
            # It is important to break the condition in two, because in advanced interations we always have a "must include" attribute
            if dataset.getAllColumns(False).shape[1] > 60:
                initialSelectionAttEvaluator = InformationGainFilterEvaluator()
                mustIncluseAttributes = OperatorsAssignmentsManager.getTopRankingDiscreteAttributesByFilterScore(dataset, initialSelectionAttEvaluator, 10)

        operatorAssignments = OperatorsAssignmentsManager.getOperatorAssignments(dataset, mustIncluseAttributes, operators, maxNumOfSourceAttributes)
        if preRankerEvaluator != None:
            preRankedAttributesToGenerate = Properties.preRankedAttributesToGenerate
            operatorAssignments = OperatorsAssignmentsManager.getTopRankingOperatorAssignmentsWithoutGenerating( subFoldTrainingDatasets, operatorAssignments, preRankerEvaluator, preRankedAttributesToGenerate )

        # Create all the new features, save them to file and evaluate them using the filter evaluator
        OperatorsAssignmentsManager.generateAttributeAndCalculateFilterEvaluatorScore(dataset, filterEvaluator, subFoldTrainingDatasets, currentScores, operatorAssignments)

        # /*
        # // The single thread version
        # for (OperatorAssignment os: operatorAssignments) {
        #     ColumnInfo ci = generateColumn(dataset, os, true);
        #     //if the filter evaluator is not null, we'll conduct the initial evaluation of the new attribute
        #     if (filterEvaluator != null) {
        #         os.setFilterEvaluatorScore(EvaluateAttributeUsingTrainingSubFolds(subFoldTrainingDatasets, filterEvaluator, os));
        #     }
        # }*/
        return operatorAssignments

    # Generates/retrieves the attribute specified, adds it to a replica of the dataset and calculates the score
    # based on the provided wrapper method
    def applyOperatorAndPerformWrapperEvaluation(self, datasets: List[Dataset], operatorAssignment: OperatorAssignment,
                wrapperEvaluator: AucWrapperEvaluator, currentScores:  List[ClassificationResults], completeDataset:  Dataset):
        score = 0.0
        for i in range(len(datasets)):
            dataset = datasets[i]
            currentScore = None
            if currentScores != None:
                currentScore = currentScores[i]

            datasetReplica = dataset.replicateDataset()
            ci = OperatorsAssignmentsManager.generateColumn(datasetReplica, operatorAssignment, True)
            iterationScore = wrapperEvaluator.produceScore(datasetReplica, currentScore, completeDataset, operatorAssignment, ci)
            score += iterationScore
        return score/len(datasets)

    # Ranks all current DISCRETE columns in the dataset using a filter evaluator and returns the top X ranking columns (ties are broken randomly)
    @staticmethod
    def getTopRankingDiscreteAttributesByFilterScore(dataset: Dataset, filterEvaluator: FilterEvaluator, numOfAttributesToReturn: int) -> list:
        IGScoresPerColumnIndex: Dict[float, list] = {}
        for colName in dataset.getAllColumns(False).columns:
            ci = dataset.df[colName]
            if dataset.targetClass == ci.name:
                continue

            # if the attribute is string or date, not much we can do about that
            if not pd.api.types.is_integer_dtype(ci):
                continue

            indicedList = []
            indicedList.append(colName)
            replicatedDataset = dataset.emptyReplica()

            columnsToAnalyze = []
            columnsToAnalyze.append(colName)
            filterEvaluator.initFilterEvaluator(ci)
            score = filterEvaluator.produceScore(replicatedDataset, None, dataset, None, ci)
            if score not in IGScoresPerColumnIndex:
                IGScoresPerColumnIndex[score] = []
            IGScoresPerColumnIndex[score].append(colName)

        columnsToReturn = []

        for score, cols in IGScoresPerColumnIndex.items():
            for col in cols:
                columnsToReturn.append(col)
                if len(columnsToReturn) >= numOfAttributesToReturn:
                    return columnsToReturn
        return columnsToReturn

    @staticmethod
    def getTopRankingOperatorAssignmentsWithoutGenerating( subFoldTrainingDatasets: List[Dataset], operatorAssignments:List[OperatorAssignment],
                             preRankerEvaluator, numOfOperatorAssignmentsToGet:int) -> List[OperatorAssignment]:
        replicatedSubFoldsList = []
        for subFoldDataset in subFoldTrainingDatasets:
            replicatedSubFoldsList.append(subFoldDataset.replicateDataset())

        numOfThread = Properties.numOfThreads

        def produceScoreOfOperationAssigment(oa):
            try:
                finalScore = 0.0
                for dataset in replicatedSubFoldsList:
                    datasetEmptyReplica = dataset.emptyReplica()
                    try:
                        finalScore += preRankerEvaluator.produceScore(datasetEmptyReplica, None, dataset, oa, None)
                    except:
                        finalScore += preRankerEvaluator.produceScore(datasetEmptyReplica, None, dataset, oa, None)

                finalScore = (finalScore / len(replicatedSubFoldsList))
                oa.setPreRankerEvaluatorScore(finalScore)
            except Exception as ex:
                Logger.Error("getTopRankingOperatorAssignmentsWithoutGenerating -> error while evaluating attribute: " + oa.getName(),
                             ex)

        if numOfThread > 1:
            Parallel.ParallelForEach(produceScoreOfOperationAssigment, [[oa] for oa in operatorAssignments])
        else:
            for oa in operatorAssignments:
                produceScoreOfOperationAssigment(oa)

        from ..Search.PreRankerScoreRanker import PreRankerScoreRanker
        ranker = PreRankerScoreRanker()
        oaListRanked = ranker.rankAndFilter(operatorAssignments, None, None, None)
        if numOfOperatorAssignmentsToGet < len(oaListRanked):
            oaListRanked =oaListRanked[: numOfOperatorAssignmentsToGet]
        return oaListRanked

    @staticmethod
    def generateAttributeAndCalculateFilterEvaluatorScore(dataset: Dataset, filterEvaluator: FilterEvaluator,
                 subFoldTrainingDatasets: List[Dataset] , currentScores:  List[ClassificationResults],
                  operatorAssignments: List[OperatorAssignment]):
        # //System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", "1");
        Logger.Info("generateAttributeAndCalculateFilterEvaluatorScore -> num of attributes to evaluate: " + str(len(operatorAssignments)))
        counter = 0
        numOfThread = Properties.numOfThreads
        def evaluateScore(oa):
            try:
                # attributeGenerationLock.lock();
                replicatedDataset = dataset.replicateDataset()
                # counter += 1
                # if (counter % 1000) == 0:
                #     date = Date()
                #     Logger.Info("generateAttributeAndCalculateFilterEvaluatorScore -> analyzed " + counter + " attributes : " + date.toString())

                # // attributeGenerationLock.unlock();

                ci = OperatorsAssignmentsManager.generateColumn(replicatedDataset, oa, True)
                 # if the filter evaluator is not null, we'll conduct the initial evaluation of the new attribute
                if (ci is not None) and (filterEvaluator is not None):
                 # filterEvaluationLock.lock();
                    cloneEvaluator = filterEvaluator.getCopy()
                    replicatedSubFoldsList = []
                    for subFoldDataset in subFoldTrainingDatasets:
                        replicatedSubFoldsList.append(subFoldDataset.replicateDataset())

                    # filterEvaluationLock.unlock();
                    filterEvaluatorScore = OperatorsAssignmentsManager.EvaluateAttributeUsingTrainingSubFolds(
                        replicatedSubFoldsList, cloneEvaluator, oa, currentScores)
                    # oa.setFilterEvaluatorScore(filterEvaluatorScore)
                return filterEvaluatorScore

            except Exception as ex:
                Logger.Error(
                        "generateAttributeAndCalculateFilterEvaluatorScore -> error when generating and evaluating attribute: " + oa.getName(), ex)
                return None

        if (numOfThread > 1):
            filterEvaluatorScores = Parallel.ParallelForEach(evaluateScore, [[oa] for oa in operatorAssignments])
            for i, oa in enumerate(operatorAssignments):
                oa.setFilterEvaluatorScore(filterEvaluatorScores[i])

        else:
            for oa in operatorAssignments:
                oa.setFilterEvaluatorScore(evaluateScore(oa))

    # Creates the new attribute. Also writes it to a file.
    # @param finalAttribute indicates if this is the version that is generated from the COMPLETE training set.
    # This is the only version that needs to be written or read from the file system
    @staticmethod
    def generateColumn(dataset: Dataset, os:  OperatorAssignment, finalAttribute: bool):
        writeToFile = False
        try:
            ci = None
            # No writing to files
            # if finalAttribute and writeToFile:
            #     ci = OperatorsAssignmentsManager.readColumnInfoFromFile(dataset.name, os.getName())
            if ci == None:
                operator = None
                try:
                    operator = OperatorsAssignmentsManager.getOperator(os.getOperator())

                except Exception as ex:
                    Logger.Info("Sleeping, try again")
                    time.sleep(0.1)
                    operator = OperatorsAssignmentsManager.getOperator(os.getOperator())

                operator.processTrainingSet(dataset, os.getSources(), os.getTargets())

                try:
                    ci = operator.generate(dataset, os.getSources(), os.getTargets())

                except:
                    x=5

                if (ci is not None) and (os is not None) and (os.getSecondaryOperator() is not None):
                    replica = dataset.emptyReplica()
                    replica.addColumn(ci)
                    uOperator = os.getSecondaryOperator()
                    tempList = []
                    tempList.append(ci)
                    try:
                        uOperator.processTrainingSet(replica, tempList, None)
                        ci2 = uOperator.generate(replica, tempList, None, True)
                        ci = ci2

                    except Exception as ex:
                        pass

                if finalAttribute and writeToFile:
                    # write the column to file, so we don't have to calculate it again
                    OperatorsAssignmentsManager.writeColumnInfoToFile(dataset.name, os.getName(), ci)

            return ci

        except Exception as ex:
            operator = OperatorsAssignmentsManager.getOperator(os.getOperator())
            operator.processTrainingSet(dataset, os.getSources(), os.getTargets())
            Logger.Error("Error while generating column: " + str(ex), ex)
            raise Exception("Failure to generate column")

    # Evaluates a set of datasets using a leave-one-out evaluation
    @staticmethod
    def EvaluateAttributeUsingTrainingSubFolds(datasets: List[Dataset], filterEvaluator:  FilterEvaluator,
                            operatorAssignment:OperatorAssignment, currentScores: List[ClassificationResults] ) -> float:
        finalScore = 0

        for i, dataset in enumerate(datasets):
            currentScore = None
            if currentScores != None:
                currentScore = currentScores[i]

            ci = OperatorsAssignmentsManager.generateColumn(dataset, operatorAssignment, False)
            if ci == None:
                return float('-inf')

            tempList = []
            tempList.append(ci)
            filterEvaluator.initFilterEvaluator(tempList)
            datasetEmptyReplica = dataset.emptyReplica()

            try:
                finalScore += filterEvaluator.produceScore(datasetEmptyReplica, currentScore, dataset, operatorAssignment, ci)
            except:
                finalScore += filterEvaluator.produceScore(datasetEmptyReplica, currentScore, dataset, operatorAssignment, ci)

        return finalScore / len(datasets)

    # Used to recalculate the scores of existing attributes when a new search iteration begins.
    @staticmethod
    def recalculateFilterEvaluatorScores(dataset: Dataset, candidateAttributes: List[OperatorAssignment], subFoldTrainingDatasets: List[Dataset],
            filterEvaluator: FilterEvaluator, currentScores: List[ClassificationResults]):
        # If the filter is not of a type that requires recalculcation (like IG) then terminate
        if not filterEvaluator.needToRecalculateScoreAtEachIteration():
            return
        OperatorsAssignmentsManager.generateAttributeAndCalculateFilterEvaluatorScore(dataset, filterEvaluator, subFoldTrainingDatasets, currentScores, candidateAttributes)

    # Read column from a file
    def readColumnInfoFromFile(self, datasetName: str, operatorAssignmentName: str):
        # fileName = getHashedName(datasetName + operatorAssignmentName) + ".ser";
        # String filePath = properties.get("operatorAssignmentFilesLocation") + fileName;
        #
        # File file = new File(filePath);
        # if (!file.exists()) {
        #     return null;
        # }
        # FileInputStream streamIn = new FileInputStream(filePath);
        # ObjectInputStream objectinputstream = new ObjectInputStream(streamIn);
        # try {
        #     return (ColumnInfo) objectinputstream.readObject();
        # }
        # catch (Exception ex) {
        #     LOGGER.error("Error reading ColumnInfo from file " + ex.getMessage());
        # }
        return None

    # Writes a ColumnInfo object to file
    def writeColumnInfoToFile(datasetName: str, operatorAssignmentName: str,  ci):
        # String fileName = getHashedName(datasetName + operatorAssignmentName) + ".ser";
        # FileOutputStream fout = new FileOutputStream(properties.getProperty("operatorAssignmentFilesLocation") + fileName, true);
        # ObjectOutputStream oos = new ObjectOutputStream(fout);
        # oos.writeObject(ci);
        pass