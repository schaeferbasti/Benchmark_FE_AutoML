from typing import Optional

import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder

from Evaluation.EvaluationInfo import EvaluationInfo
from Utils.Logger import Logger
from Properties import Properties

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class Classifier2:

    def __init__(self, classifier: str):
        if classifier == 'RandomForest':
            # self.cls = RandomForestClassifier(n_estimators=2000, random_state=Properties.randomSeed)
            self.cls = RandomForestClassifier(min_samples_leaf=50, n_estimators=150, bootstrap=True, oob_score=True,
                                   n_jobs=-1, random_state=42)
            self.classLabel = ''
        else:
            msg = f'Unknown classifier: {classifier}'
            Logger.Error(msg)
            raise Exception(msg)

        self.pipe: Optional[Pipeline] = None
        self.classes: np.array = []

    def buildClassifier(self, trainingSet: pd.DataFrame):
        self.classLabel = 'class' if 'class' in trainingSet.columns else trainingSet.columns[-1]
        X = trainingSet.drop(labels=[self.classLabel], axis=1)

        # X = self._saveValuesOfCategoricalColumns(X)
        # X = pd.get_dummies(X)
        y = trainingSet[self.classLabel]

        # self.classes = np.sort(y.unique())

        discreteColumns = X.select_dtypes(include='int').columns
        col_trans = make_column_transformer(
            (OneHotEncoder(), discreteColumns),
            remainder="passthrough"
        )
        self.pipe = make_pipeline(col_trans, self.cls)
        self.pipe.fit(X, y)

        # self.classes = self.pipe.classes_

    def evaluateClassifier(self, testSet: pd.DataFrame) -> EvaluationInfo:
        y = testSet[self.classLabel]
        X = testSet.drop(labels=[self.classLabel], axis=1)

        # Returns ndarray of shape (n_samples, n_classes)
        scoresDist = self.pipe.predict_proba(X)

        return EvaluationInfo(self.cls, scoresDist, y)

