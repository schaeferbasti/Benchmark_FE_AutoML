from src.feature_engineering.ExploreKit.method.Evaluation.EvaluationInfo import EvaluationInfo
from src.feature_engineering.ExploreKit.method.Utils.Logger import Logger
from src.feature_engineering.ExploreKit.method.Properties import Properties

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pandas.api.types import CategoricalDtype

class Classifier:

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

        self.categoricalColumnsMap: dict

    def buildClassifier(self, trainingSet: pd.DataFrame):
        self.classLabel = 'class' if 'class' in trainingSet.columns else trainingSet.columns[-1]
        X = trainingSet.drop(labels=[self.classLabel], axis=1)

        X = self._saveValuesOfCategoricalColumns(X)
        X = pd.get_dummies(X)

        y = trainingSet[self.classLabel]

        from sklearn.preprocessing import OneHotEncoder
        discreteColumns = trainingSet.select_dtypes(include='int').columns
        encoder = OneHotEncoder()
        X2 = encoder.fit_transform(trainingSet[discreteColumns])

        self.cls.fit(X, y)

    def evaluateClassifier(self, testSet: pd.DataFrame) -> EvaluationInfo:
        X = testSet.drop(labels=[self.classLabel], axis=1)

        X = self._getDataframeWithCategoricalColumns(X)

        X = pd.get_dummies(X)

        # Returns ndarray of shape (n_samples, n_classes)
        scoresDist = self.cls.predict_proba(X)

        return EvaluationInfo(self.cls, scoresDist, testSet[self.classLabel])

    # Returns 2 lists, first is the the true/actual values and the second one is the predictions
    def predictClassifier(self, testSet: pd.DataFrame):
        X = testSet.drop(labels=['class'], axis=1)

        X = self._getDataframeWithCategoricalColumns(X)

        X = pd.get_dummies(X)

        # Returns ndarray of shape (n_samples, n_classes)
        preds = self.cls.predict(X)

        return testSet['class'].values, preds

    # Save categorical columns for one-hot encoding in test
    def _saveValuesOfCategoricalColumns(self, df: pd.DataFrame) -> pd.DataFrame:
        discreteColumns = df.select_dtypes(include='int').columns
        self.categoricalColumnsMap = {col: df[col].unique() for col in discreteColumns}
        convertIntToCategoryMap = {colName: 'category' for colName in discreteColumns}
        df = df.astype(convertIntToCategoryMap)
        return df

    # Set df's categorical columns to Categorical type to remember missing categories in test
    def _getDataframeWithCategoricalColumns(self, df: pd.DataFrame):
        for columnsName, categories in self.categoricalColumnsMap.items():
            df[columnsName] = df[columnsName].astype(CategoricalDtype(categories))
        return df
