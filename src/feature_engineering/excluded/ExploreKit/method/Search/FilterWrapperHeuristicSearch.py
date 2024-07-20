
from typing import List, Tuple, Optional

from ..Utils import Parallel
from ..Evaluation.AucWrapperEvaluator import AucWrapperEvaluator
from ..Evaluation.ClassificationResults import ClassificationResults
from ..Utils.Date import Date
from ..Evaluation.FilterEvaluator import FilterEvaluator
from ..Evaluation.FilterPreRankerEvaluator import FilterPreRankerEvaluator
from ..Evaluation.OperatorAssignment import OperatorAssignment
from ..Evaluation.OperatorsAssignmentsManager import OperatorsAssignmentsManager
from ..Properties import Properties
from ..Evaluation.MLFilterEvaluator import MLFilterEvaluator
from ..Data.Dataset import Dataset
from ..Utils.Logger import Logger
from ..Search.Search import Search


class FilterWrapperHeuristicSearch(Search):


    def __init__(self, maxIterations: int):
        self.maxIteration = maxIterations
        self.experimentStartDate: Date = None
        chosenOperatorAssignment = None
        topRankingAssignment = None
        evaluatedAttsCounter:int = 0
        super().__init__()

    def run(self, originalDataset: Dataset, runInfo: str, name: str):
        Logger.Info('Initializing evaluators')
        filterEvaluator = MLFilterEvaluator(originalDataset, name)

        preRankerEvaluator = None
        if bool(Properties.usePreRanker):
            preRankerEvaluator = FilterPreRankerEvaluator(originalDataset)

        if Properties.wrapperApproach == 'AucWrapperEvaluator':
            wrapperEvaluator = AucWrapperEvaluator()
        else:
            Logger.Error('Missing wrapper approach')
            raise Exception('Missing wrapper approach')

        experimentStartDate = Date()
        Logger.Info("Experiment Start Date/Time: " + str(self.experimentStartDate) + " for dataset " + originalDataset.name)

        # The first step is to evaluate the initial attributes, so we get a reference point to how well we did
        wrapperEvaluator.EvaluationAndWriteResultsToFile(originalDataset, "", 0, runInfo, True, 0, -1, -1)

        # now we create the replica of the original dataset, to which we can add columns
        dataset = originalDataset.replicateDataset()

        # Get the training set sub-folds, used to evaluate the various candidate attributes
        originalDatasetTrainingFolds = originalDataset.GenerateTrainingSetSubFolds()
        subFoldTrainingDatasets = dataset.GenerateTrainingSetSubFolds()

        date = Date()

        # We now apply the wrapper on the training subfolds in order to get the baseline score. This is the score a candidate attribute needs to "beat"
        currentScore = wrapperEvaluator.produceAverageScore(subFoldTrainingDatasets, None, None, None, None)
        Logger.Info(f"Initial score: {str(currentScore)} : {date}")

        # The probabilities assigned to each instance using the ORIGINAL dataset (training folds only)
        Logger.Info(f"Producing initial classification results: {date}")
        currentClassificationProbs = wrapperEvaluator.produceClassificationResults(originalDatasetTrainingFolds)
        date = Date()
        Logger.Info(f"  .....done {date}")

        # Apply the unary operators (discretizers, normalizers) on all the original features. The attributes generated
        # here are different than the ones generated at later stages because they are included in the dataset that is
        # used to generate attributes in the iterative search phase
        Logger.Info(f"Starting to apply unary operators: {date}")
        oam = OperatorsAssignmentsManager()
        candidateAttributes = oam.applyUnaryOperators(dataset, None, filterEvaluator, subFoldTrainingDatasets, currentClassificationProbs)
        date = Date()
        Logger.Info("  .....done " + str(date))

        # Now we add the new attributes to the dataset (they are added even though they may not be included in the
        # final dataset beacuse they are essential to the full generation of additional features
        Logger.Info("Starting to generate and add columns to dataset: " + str(date))
        oam.GenerateAndAddColumnToDataset(dataset, candidateAttributes)
        date = Date()
        Logger.Info("  .....done " + str(date))

        # The initial dataset has been populated with the discretized/normalized features. Time to begin the search
        iterationsCounter = 1
        columnsAddedInthePreviousIteration = None

        self.performIterativeSearch(originalDataset, runInfo, preRankerEvaluator, filterEvaluator, wrapperEvaluator, dataset, originalDatasetTrainingFolds, subFoldTrainingDatasets, currentClassificationProbs, oam, candidateAttributes, iterationsCounter, columnsAddedInthePreviousIteration)
        return dataset, candidateAttributes
    '''
    Performs the iterative search - the selection of the candidate features and the generation of the additional candidates that are added to the pool
    in the next round.
    @param originalDataset The dataset with the original attributes set
    @param runInfo
    @param preRankerEvaluator
    @param filterEvaluator The type of FilterEvaluator chosen for the expriments
    @param wrapperEvaluator The type of wrapper evaluator chosen for the experiments
    @param dataset The dataset with the "augmented" attributes set (to this object we add the selected attributes)
    @param originalDatasetTrainingFolds The training folds and the test fold (the original partitioning of the data)
    @param subFoldTrainingDatasets Only the training folds (a subset of the previous parameter)
    @param currentClassificationProbs The probabilities assigned to each instance by the classifier of belonging to each of the classes
    @param oam Manages the applying of the various operators on the attributes
    @param candidateAttributes The attributes that are being ocnsidered for adding to the dataset
    @param iterationsCounter
    @param columnsAddedInthePreviousIteration The attriubtes that were already added to the dataset
    '''
    def performIterativeSearch(self, originalDataset: Dataset, runInfo: str,  preRankerEvaluator:FilterPreRankerEvaluator, filterEvaluator:  FilterEvaluator,  wrapperEvaluator:AucWrapperEvaluator,
                             dataset: Dataset, originalDatasetTrainingFolds: List[Dataset], subFoldTrainingDatasets: List[Dataset], currentClassificationProbs:List[ClassificationResults],
                             oam: OperatorsAssignmentsManager, candidateAttributes: List[OperatorAssignment], iterationsCounter:  int, columnsAddedInthePreviousIteration):
        totalNumberOfWrapperEvaluations = 0
        rankerFilter = self.getRankerFilter(Properties.rankerApproach)

        #TODO: make sure not exceeding property "maxNumOfWrapperEvaluationsPerIteration"
        def evaluateOperationAssignment(oa: OperatorAssignment) -> Tuple[float, Optional[OperatorAssignment]]:
            try:
                if oa.getFilterEvaluatorScore() != float('-inf')  and oa.getFilterEvaluatorScore() > 0.001:
                    score = OperatorsAssignmentsManager.applyOperatorAndPerformWrapperEvaluation(
                        originalDatasetTrainingFolds, oa, wrapperEvaluator, localCurrentClassificationProbs, None)
                    oa.setWrapperEvaluatorScore(score)
                    return (score, oa)
                    # wrapperResultsLock.lock();
                    # evaluatedAttsCounter ++;

                    # we want to keep tabs on the OA with the best observed wrapper performance
                    # if topRankingAssignment == None or topRankingAssignment.getWrapperEvaluatorScore() < score:
                    #     Logger.Info("found new top ranking assignment")
                    #     topRankingAssignment = oa

                    # if isStoppingCriteriaMet(filterEvaluator, wrapperEvaluator, oa, score, topRankingAssignment):
                    #     chosenOperatorAssignment = oa

                    # if (evaluatedAttsCounter % 100) == 0:
                    #     currentDate = Date()
                    #     Logger.Info(
                    #         f"performIterativeSearch ->                     Evaluated: {evaluatedAttsCounter} attributes: {str(currentDate)}")
                    #
                    # wrapperResultsLock.unlock();

            except Exception as ex:
                Logger.Error(f"Exception occurred {ex}", ex)
            return (0.0, None)

        while iterationsCounter <= self.maxIteration:
            filterEvaluator.recalculateDatasetBasedFeatures(originalDataset)
            date = Date()
            Logger.Info(f"performIterativeSearch -> Starting search iteration {int(iterationsCounter)}{str(date)}")

            # recalculte the filter evaluator score of the existing attributes
            OperatorsAssignmentsManager.recalculateFilterEvaluatorScores(dataset,candidateAttributes,subFoldTrainingDatasets,filterEvaluator,currentClassificationProbs)

            # now we generate all the candidate features
            date = Date()
            Logger.Info(f"performIterativeSearch ->            Starting feature generation:  {str(date)}")
            candidateAttributes.addAll(oam.applyNonUnaryOperators(dataset, columnsAddedInthePreviousIteration,preRankerEvaluator, filterEvaluator, subFoldTrainingDatasets, currentClassificationProbs))
            date = Date()
            Logger.Info(f"performIterativeSearch ->            Finished feature generation: {str(date)}")

            # Sort the candidates by their initial (filter) score and test them using the wrapper evaluator
            candidateAttributes = rankerFilter.rankAndFilter(candidateAttributes,columnsAddedInthePreviousIteration,subFoldTrainingDatasets,currentClassificationProbs)

            Logger.Info(f"performIterativeSearch ->            Starting wrapper evaluation : {str(date)}")
            evaluatedAttsCounter = 0
            chosenOperatorAssignment = None
            topRankingAssignment = None

            # ReentrantLock wrapperResultsLock = new ReentrantLock();
            numOfThreads = Properties.numOfThreads

            localCurrentClassificationProbs = currentClassificationProbs
            # for i in range(len(candidateAttributes), numOfThreads):
            #     if chosenOperatorAssignment != None:
            #         break
                # oaList = candidateAttributes[i, i + min(numOfThreads, len(candidateAttributes)-i)]
                # oaList.parallelStream().forEach(oa -> {
            evaluatedCandidateAttrs = Parallel.ParallelForEach(evaluateOperationAssignment, [[oa] for oa in candidateAttributes])
            from operator import itemgetter
            tempTopRank = max(evaluatedCandidateAttrs, key=itemgetter(0))
            # if self.isStoppingCriteriaMet(tempTopRank[0]):

            totalNumberOfWrapperEvaluations += len(evaluatedCandidateAttrs)
            Logger.Info(f"performIterativeSearch ->            Finished wrapper evaluation : {str(date)}")

            # remove the chosen attribute from the list of "candidates"
            candidateAttributes.remove(chosenOperatorAssignment)

            # The final step - add the new attribute to the datasets
            # start with the dataset used in the following search iterations
            columnsAddedInthePreviousIteration = OperatorsAssignmentsManager.addAddtibuteToDataset(dataset, chosenOperatorAssignment, True, currentClassificationProbs)

            # continue with the final dataset
            OperatorsAssignmentsManager.addAddtibuteToDataset(originalDataset, chosenOperatorAssignment, False, currentClassificationProbs);

            # finally, we need to recalculate the baseline score used for the attribute selection (using the updated final dataset)
            currentClassificationProbs = wrapperEvaluator.produceClassificationResults(originalDatasetTrainingFolds)

            expDescription = ''
            expDescription += f"Evaluation results for iteration {str(iterationsCounter)}\n"
            expDescription += f"Added attribute: {chosenOperatorAssignment.getName()}\n"
            wrapperEvaluator.EvaluationAndWriteResultsToFile(originalDataset, chosenOperatorAssignment.getName(), iterationsCounter, runInfo, False, evaluatedAttsCounter, chosenOperatorAssignment.getFilterEvaluatorScore() ,chosenOperatorAssignment.getWrapperEvaluatorScore())
            iterationsCounter += 1

        # some cleanup, if required
        filterEvaluator.deleteBackgroundClassificationModel(originalDataset)

        # After the search process is over, write the total amount of time spent and the number of wrapper evaluations that were conducted
        self.writeFinalStatisticsToResultsFile(dataset.name, runInfo, self.experimentStartDate, totalNumberOfWrapperEvaluations)

    # Determines whether to terminate the wrapper evaluation of the candidates. If returns "true", it also
    # sets the value of the chosenOperatorAssignment parameter that contains the attribute that will be added
    # to the dataset
    # def isStoppingCriteriaMet(self, filterEvaluator: FilterEvaluator, wrapperEvaluator: WrapperEvaluator,
    #                currentAssignment: OperatorAssignment, score: float, topRankingAssignment: OperatorAssignment):
    def isStoppingCriteriaMet(self, score: float):
        return score > 0.01

    def writeFinalStatisticsToResultsFile(self, datasetName: str, runInfo: str,
                                          experimentStartTime: Date , totalNumOfWrapperEvaluations: int):
        filename = Properties.resultsFilePath + datasetName + runInfo + ".csv"
        with open(filename, 'a') as fw:
            experimentEndTime = Date()
            diff = (experimentEndTime -experimentStartTime)
            diffSeconds = diff.seconds % 60
            diffMinutes = diff.seconds / 60  % 60
            diffHours = diff.seconds / (60 * 60)

            fw.write("\n")
            fw.write("Total Run Time: " + "\n")
            fw.write(f"Total Run Time: {'{:0>8}'.format(str(diff))} \n")
            fw.write(f"Number of hours: {diffHours}\n")
            fw.write(f"Number of minutes: {diffMinutes}\n")
            fw.write(f"Number of seconds: {diffSeconds}\n")
            fw.write(f"Total number of evaluated attribtues: {totalNumOfWrapperEvaluations}")
            fw.close()
