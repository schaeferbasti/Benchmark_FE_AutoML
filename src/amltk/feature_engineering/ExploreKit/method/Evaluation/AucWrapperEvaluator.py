import os
from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score

from src.amltk.feature_engineering.ExploreKit.method.Evaluation.ClassificationItem import ClassificationItem
from src.amltk.feature_engineering.ExploreKit.method.Evaluation.ClassificationResults import ClassificationResults
from src.amltk.feature_engineering.ExploreKit.method.Evaluation.Classifier import Classifier
from src.amltk.feature_engineering.ExploreKit.method.Data.Dataset import Dataset
from src.amltk.feature_engineering.ExploreKit.method.Evaluation.EvaluationInfo import EvaluationInfo
from src.amltk.feature_engineering.ExploreKit.method.Utils.Logger import Logger
from src.amltk.feature_engineering.ExploreKit.method.Utils.Date import Date
from src.amltk.feature_engineering.ExploreKit.method.Evaluation.OperatorAssignment import OperatorAssignment
from src.amltk.feature_engineering.ExploreKit.method.Properties import Properties


class AucWrapperEvaluator:

     # Gets the ClassificationResults items for each of the analyzed datasets (contains the class probabilites and true
     # class for each instance)

    def produceClassificationResults(self, datasets: list) -> list:
        classificationResultsPerFold = []
        for dataset in datasets:
            date = Date()
            Logger.Info("Starting to run classifier " + str(date))
            trainSet = dataset.generateSet(True)
            testSet = dataset.generateSet(False)
            evaluationResults = self.runClassifier(Properties.classifier, trainSet, testSet)
            date = Date()
            Logger.Info("Starting to process classification results " + str(date))
            classificationResults = self.getClassificationResults(evaluationResults, dataset, testSet)
            date = Date()
            Logger.Info("Done " + str(date))
            classificationResultsPerFold.append(classificationResults)

        return classificationResultsPerFold

    def runClassifier(self, classifierName: str, trainingSet, testSet) -> EvaluationInfo:
        try:
            classifier = Classifier(classifierName)
            classifier.buildClassifier(trainingSet)

            # evaluation = new Evaluation(trainingSet);
            # evaluation.evaluateModel(classifier, testSet)
            evaluationInfo = classifier.evaluateClassifier(testSet)

            return evaluationInfo

        except Exception as ex:
            Logger.Error("problem running classifier " + str(ex))

        return None

    # Obtains the classification probabilities assigned to each instance and returns them as a ClassificationResults object
    def getClassificationResults(self, evaluation, dataset: Dataset, testSet):
        date = Date()

        # used for validation - by making sure that that the true classes of the instances match we avoid "mix ups"
        actualTargetColumn = dataset.df[dataset.targetClass]

        classificationItems = []
        counter = 0
        actualValues = testSet[dataset.targetClass].values
        predDistribution = evaluation.getScoreDistribution()
        for i in range(predDistribution.shape[0]):
            # if ((counter%10000) == 0) {
            #     if ((int) prediction.actual() != (Integer) actualTargetColumn.getValue(dataset.getIndicesOfTestInstances().get(counter))) {
            #         if (dataset.getTestDataMatrixWithDistinctVals() == null || dataset.getTestDataMatrixWithDistinctVals().length == 0) {
            #             throw new Exception("the target class values do not match");
            #         }
            #     }
            # }
            counter += 1
            ci = ClassificationItem(actualValues[i],predDistribution[i])
            classificationItems.append(ci)

        # Now generate all the statistics we may want to use
        auc = self.CalculateAUC(evaluation, dataset, testSet)

        logloss = self.CalculateLogLoss(evaluation, dataset)

        # We calcualte the TPR/FPR rate. We do it ourselves because we want all the values
        tprFprValues = self.calculateTprFprRate(evaluation, dataset, testSet)

        # The TRR/FPR values enable us to calculate the precision/recall values.
        recallPrecisionValues = self.calculateRecallPrecisionValues(dataset, tprFprValues,
                float(Properties.precisionRecallIntervals))

        # Next, we calculate the F-Measure at the selected points
        fMeasureValuesPerRecall = {}
        fMeasurePrecisionValues = Properties.FMeausrePoints
        for recallVal in fMeasurePrecisionValues:
            recall = float(recallVal)
            precision = recallPrecisionValues[recall]
            F1Measure = (2*precision*recall)/(precision+recall)
            fMeasureValuesPerRecall[recall] = F1Measure

        classificationResults = ClassificationResults(classificationItems, auc, logloss, tprFprValues, recallPrecisionValues, fMeasureValuesPerRecall)

        return classificationResults


    def CalculateAUC(self, evaluation, dataset: Dataset, testSet) -> float:
        return roc_auc_score(testSet[dataset.targetClass], evaluation.scoreDistPerInstance[:, 1])

    def CalculateLogLoss(self, evaluation, dataset):
        probs = evaluation.getScoreDistribution()
        probs = np.max(probs, axis=1)
        probs = np.maximum(np.minimum(probs, 1 - 1E-15), 1E-15)
        probs = np.log(probs)
        logLoss = np.sum(probs) / probs.shape[0]
        return logLoss

    def getClassificationItemList(self, testSet, evaluation):
         assert testSet.shape[0] == evaluation.getScoreDistribution().shape[0]
         # classificationItems = []
         probs = evaluation.getScoreDistribution()
         classes = evaluation.getEvaluationStats().classes_
         classLabel = 'class' if 'class' in testSet.columns else testSet.columns[-1]
         classificationItems = [ClassificationItem(classIndex, dict(zip(classes, prob))) for classIndex, prob in zip(testSet[classLabel].values, probs)]
         # for i in range(testSet.shape[0]):
         #     classificationItems.append(ClassificationItem(testSet.iloc[i], dict(zip(classes, probs[i]))))
         return classificationItems

    # Used to calculate all the TPR-FPR values of the provided evaluation
    def calculateTprFprRate(self, evaluation, dataset, testSet) -> dict:
        date = Date()
        Logger.Info("Starting TPR/FPR calculations : " + str(date))

        # trpFprRates = {}

        # we convert the results into a format that's more comfortable to work with
        classificationItems = self.getClassificationItemList(testSet, evaluation)
        # for (Prediction prediction: evaluation.predictions()) {
        #     ClassificationItem ci = new ClassificationItem((int)prediction.actual(),((NominalPrediction)prediction).distribution());
        #     classificationItems.add(ci);
        # }

        # now we need to know what is the minority class and the number of samples for each class
        minorityClassIndex = dataset.getMinorityClassIndex()
        numOfNonMinorityClassItems = 0 #all non-minority class samples are counted together (multi-class cases)
        for cls in dataset.getNumOfRowsPerClassInTestSet().keys():
            if cls != minorityClassIndex:
                numOfNonMinorityClassItems += dataset.getNumOfRowsPerClassInTestSet()[cls]

        # sort all samples by their probability of belonging to the minority class
        classificationItems.sort(reverse=True, key=lambda x:x.getProbabilitiesOfClass(minorityClassIndex))
        # Collections.sort(classificationItems, new ClassificationItemsComparator(minorityClassIndex));
        # Collections.reverse(classificationItems);

        tprFprValues = {}
        tprFprValues[0.0] = 0.0
        minoritySamplesCounter = 0
        majoritySamplesCounter = 0
        currentProb = 2
        for ci in classificationItems:
            currentSampleProb = ci.getProbabilitiesOfClass(minorityClassIndex)
            # if the probability is different, time to update the TPR/FPR statistics
            if currentSampleProb != currentProb:
                tpr =  minoritySamplesCounter/dataset.getNumOfRowsPerClassInTestSet()[minorityClassIndex]
                fpr = majoritySamplesCounter/numOfNonMinorityClassItems
                tprFprValues[tpr] = fpr
                currentProb = currentSampleProb

            if ci.getTrueClass() == minorityClassIndex:
                minoritySamplesCounter += 1
            else:
                majoritySamplesCounter += 1

        tprFprValues[1.0] = 1.0
        tprFprValues[1.0001] = 1.0
        date = Date()
        Logger.Info("Done : " + str(date))
        return tprFprValues

    # Used to calculate the recall/precision values from the TPR/FPR values. We use the recall values as the basis for
    # our calculation because they are monotonic and becuase it enables the averaging of different fold values
    def calculateRecallPrecisionValues(self, dataset: Dataset, tprFprValues: dict, recallInterval: float):
        # start by getting the number of samples in the minority class and in other classes
        minorityClassIndex = dataset.getMinorityClassIndex()
        numOfMinorityClassItems = dataset.getNumOfRowsPerClassInTestSet()[minorityClassIndex]
        numOfNonMinorityClassItems = 0 # all non-minority class samples are counted together (multi-class cases)
        for idx, value in dataset.getNumOfRowsPerClassInTestSet().items():
            if idx != minorityClassIndex:
                numOfNonMinorityClassItems += value

        recallPrecisionValues = {}
        for i in np.arange(0, 1+1e-5, recallInterval):
            recallKey = self.getClosestRecallValue(tprFprValues, i)  # the recall is the TPR
            #TODO: ask Gilad about recallKey and tprFprValue of 0.0
            try:
                precision = (recallKey*numOfMinorityClassItems)/((recallKey*numOfMinorityClassItems) + (tprFprValues[recallKey]*numOfNonMinorityClassItems))
            except ZeroDivisionError:
                precision = 0
            # if np.isnan(precision):
            #     precision = 0
            recallPrecisionValues[round(i, 2)] = precision

        return recallPrecisionValues


    # Returns the ACTUAL recall value that is closest to the requested value. It is important to note that there are
    # no limitations in this function, so in end-cases the function may return strange results.
    def getClosestRecallValue(self, tprFprValues: dict, recallVal: float) -> float:
        for key in tprFprValues.keys():
            if key >= recallVal:
                return key
        return 0

    # This procedure should only be called when an attribute (or attributes) have been selected. It will
    # several statistics into the output file, statistics whose calculation requires additional time to the
    # one spent by Weka itself.
    # @param newFile used to determine whether the text needs to be appended or override any existing text
    def EvaluationAndWriteResultsToFile(self,dataset: Dataset, addedAttribute: str, iteration: int, runInfo: str,
                newFile: bool, evaluatedAttsCounter: int, filterEvaluatorScore: float, wrapperEvaluationScore: float):

        evaluation = self.runClassifier(Properties.classifier,dataset.generateSet(True), dataset.generateSet(False))

        # We calcualte the TPR/FPR rate. We do it ourselves because we want all the values
        tprFprValues = self.calculateTprFprRate(evaluation, dataset)

        # The TRR/FPR values enable us to calculate the precision/recall values.
        recallPrecisionValues = self.calculateRecallPrecisionValues(dataset,
                                        tprFprValues, Properties.precisionRecallIntervals)


        # Next, we calculate the F-Measure at the selected points
        fMeasureValuesPerRecall = {}
        fMeasurePrecisionValues = Properties.FMeausrePoints
        for recallVal in fMeasurePrecisionValues:
            precision = recallPrecisionValues[recallVal]
            F1Measure = (2*precision*recallVal)/(precision+recallVal)
            fMeasureValuesPerRecall[recallVal], = F1Measure

        # now we can write everything to file
        sb = ''

        # If it's a new file, we need to create a header for the file
        if newFile:
            sb += "Iteration,Added_Attribute,LogLoss,AUC,"
            for recallVal in fMeasureValuesPerRecall.keys():
                sb += f"F1_Measure_At_Recall_{recallVal},"

            for recallVal in recallPrecisionValues.keys():
                sb += f"Precision_At_Recall_Val_{recallVal},"

            sb += "Chosen_Attribute_Filter_Score,Chosen_Attribute_Wrapper_Score,Num_Of_Evaluated_Attributes_In_Iteration"
            sb += "Iteration_Completion_time"
            sb += os.linesep

        sb += str(iteration) + ","
        sb += f'"{addedAttribute}",'

        # The LogLoss
        sb += str(self.CalculateLogLoss(evaluation, dataset))+ ","

        # The AUC
        sb += str(roc_auc_score(evaluation.actualPred,
                                evaluation.scoreDistPerInstance[:, dataset.getMinorityClassIndex()])) + ','
        # evaluation.areaUnderROC(dataset.getMinorityClassIndex())).concat(","));

        # The F1 measure
        for recallVal in fMeasureValuesPerRecall.keys():
            sb += str(fMeasureValuesPerRecall[recallVal]) + ","

        # Recall/Precision values
        for recallVal in recallPrecisionValues.keys():
            sb += str(recallPrecisionValues[recallVal]) + ","

        sb += str(filterEvaluatorScore) + ","
        sb += str(wrapperEvaluationScore) + ","
        sb += str(evaluatedAttsCounter) + ","

        date = Date()
        sb += date.__str__()

        try:
            filename= Properties.resultsFilePath + dataset.name + runInfo + ".csv"
            if newFile:
                fw = open(filename, "w")
            else:
                fw = open(filename, "a")
            fw.write(sb + "\n")
            fw.close()

        except Exception as ex:
            Logger.Error("IOException: " + ex)

    # Calculates the score for each of the datasets in the list (subfolds) and returns the average score
    def produceAverageScore(self, analyzedDatasets: List[Dataset], classificationResults: list,
                            completeDataset: Dataset, oa: OperatorAssignment, candidateAttribute) -> float:
        score = 0.0
        for i, dataset in enumerate(analyzedDatasets):
            # the ClassificationResult can be null for the initial run.
            classificationResult = None
            if classificationResults != None:
                classificationResult = classificationResults[i]

            score += self.produceScore(dataset, classificationResult, completeDataset, oa, candidateAttribute)
        return score/len(analyzedDatasets)

    def produceScore(self, analyzedDatasets: Dataset, currentScore: ClassificationResults, completeDataset: Dataset,
                      oa: OperatorAssignment, candidateAttribute) -> float:
        if candidateAttribute != None:
            analyzedDatasets.addColumn(candidateAttribute)

        evaluationResults = self.runClassifier(Properties.classifier,
                                    analyzedDatasets.generateSet(True), analyzedDatasets.generateSet(False))

        # in order to deal with multi-class datasets we calculate an average of all AUC scores (we may need to make this weighted)
        auc = self.CalculateAUC(evaluationResults, analyzedDatasets, evaluationResults.actualPred)

        if currentScore != None:
            return auc - currentScore.getAuc()
        else:
            return auc
