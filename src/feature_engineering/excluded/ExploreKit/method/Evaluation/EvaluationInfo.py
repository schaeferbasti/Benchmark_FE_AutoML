import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score


class EvaluationInfo:

    # evaluationStats: Classifier
    # scoreDistributions: 2d array of test predictions
    def __init__(self, evaluationStats, scoreDistributions: np.ndarray, actualPred: pd.Series):
        self.classifier = evaluationStats
        self.scoreDistPerInstance: np.ndarray = scoreDistributions
        self.predictions: np.ndarray = np.max(scoreDistributions, axis=1)
        self.actualPred: pd.Series = actualPred

    def getEvaluationStats(self):
        return self.classifier

    def getScoreDistribution(self) -> np.ndarray:
        return self.scoreDistPerInstance

    def getPredictions(self) -> np.ndarray:
        return self.predictions

    def get_roc_auc_score(self) -> float:
        return roc_auc_score(self.actualPred, self.scoreDistPerInstance[:, 1])

    def get_accuracy_score(self) -> float:
        y_pred = self.classifier.classes_.take(np.argmax(self.scoreDistPerInstance, axis=1), axis=0)
        return accuracy_score(self.actualPred, y_pred)
