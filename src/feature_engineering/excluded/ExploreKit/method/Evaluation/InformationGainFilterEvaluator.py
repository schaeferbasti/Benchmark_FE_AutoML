from typing import Dict

import pandas as pd

from src.feature_engineering.excluded.ExploreKit.method.Evaluation.ClassificationResults import ClassificationResults
from src.feature_engineering.excluded.ExploreKit.method.Data.Dataset import Dataset
from src.feature_engineering.excluded.ExploreKit.method.Evaluation.FilterEvaluator import FilterEvaluator
from src.feature_engineering.excluded.ExploreKit.method.Evaluation.OperatorAssignment import OperatorAssignment

import math

class InformationGainFilterEvaluator(FilterEvaluator):

    def __init__(self):
        super().__init__()

    def produceScore(self, analyzedDatasets: Dataset, currentScore: ClassificationResults, completeDataset: Dataset, oa: OperatorAssignment, candidateAttribute):
        if candidateAttribute != None:
            analyzedDatasets = pd.concat([analyzedDatasets, candidateAttribute], axis=1)


        # if any of the analyzed attribute is not discrete, it needs to be discretized
        bins = [0]*10
        FilterEvaluator.discretizeColumns(self, analyzedDatasets, bins)
        # Todo: distinct value. make sure to ignore
        # if (analyzedDatasets.getDistinctValueColumns() != None) and (analyzedDatasets.getDistinctValueColumns().size() > 0):
        #     return self.produceScoreWithDistinctValues(analyzedDatasets, currentScore, oa, candidateAttribute)

        valuesPerKey = {}
        targetColumn = analyzedDatasets.getTargetClassColumn()

        # In filter evaluators we evaluate the test set, the same as we do in wrappers. The only difference here is that we
        # train and test on the test set directly, while in the wrappers we train a model on the training set and then apply on the test set
        # for i in range(analyzedDatasets.getNumOfTestDatasetRows()):
        for j in analyzedDatasets.getIndicesOfTestInstances():
            # sourceValues: list = [self.analyzedColumns[c][j] for c in self.analyzedColumns.columns]
            sourceValues = [col[j] for col in self.analyzedColumns]  #[self.analyzedColumns.columns]
            targetValue = targetColumn[j]
            key = hash(tuple(sourceValues)) #hash(sourceValues.tobytes())
            if key not in valuesPerKey:
                # valuesPerKey[key] = np.zeros(analyzedDatasets.getTargetClassColumn().getColumn().getNumOfPossibleValues())
                valuesPerKey[key] = dict.fromkeys(analyzedDatasets.classes, 0)
            valuesPerKey[key][targetValue] += 1

        return self.calculateIG(analyzedDatasets, valuesPerKey)

    # def produceScoreWithDistinctValues(self, dataset:Dataset , currentScore:ClassificationResults, oa:OperatorAssignment, candidateAttribute:ColumnInfo):
    #     pass

    def calculateIG(self, dataset: Dataset, valuesPerKey: Dict[int, Dict[str, int]]):
        IG = 0.0
        for val in valuesPerKey.values():
            numOfInstances = sum(val.values())
            tempIG = 0
            for value in val.values():
                if value != 0:
                    tempIG += -((value / numOfInstances) * math.log10(value / numOfInstances))

            IG += (numOfInstances/dataset.getNumOfTrainingDatasetRows()) * tempIG
        return IG

