import random
import pandas as pd

from sklearn.metrics import roc_auc_score, accuracy_score

from ..Data.Dataset import Dataset
from ..Evaluation.Classifier import Classifier
from ..Properties import Properties
from ..Evaluation.Classifier2 import Classifier2
from ..Utils.Loader import Loader

def split_data(data: pd.DataFrame, test_size):
    random.seed(42)
    size = data.shape[0]
    indicesOfTrainingFolds = random.sample(list(range(size)), int(size * (1 - test_size)))
    indicesOfTestFolds = list(set(list(range(size))) - set(indicesOfTrainingFolds))

    train_set = data.iloc[indicesOfTrainingFolds, :]
    test_set = data.iloc[indicesOfTestFolds, :]
    return train_set, test_set

def test_classifier2(data: pd.DataFrame):
    classifier = Classifier2(Properties.classifier)

    train_set, test_set = split_data(data, test_size=0.333)

    classifier.buildClassifier(train_set)
    eval_info = classifier.evaluateClassifier(test_set)

    acc = eval_info.get_accuracy_score()
    auc = eval_info.get_roc_auc_score()
    print(f"The accuracy of the model is {round(acc, 3) * 100} %")
    print(f'Test ROC AUC  Score: {auc}')
def main3(data: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import make_column_transformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, roc_auc_score

    y = data.pop('21')
    X = data #.drop('id', axis=1)

    # size = X.shape[0]
    # indicesOfTrainingFolds = random.sample(list(range(size)), int(size * 0.666))
    # indicesOfTestFolds = list(set(list(range(size))) - set(indicesOfTrainingFolds))
    # X_train, X_test, y_train, y_test = X.iloc[indicesOfTrainingFolds, :], X.iloc[indicesOfTestFolds],y.iloc[indicesOfTrainingFolds], y.iloc[indicesOfTestFolds]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=50)
    X_train = X_train.fillna('na')
    X_test = X_test.fillna('na')
    print(f'{X_train.shape}, {X_test.shape}')
    features_to_encode = X_train.columns[X_train.dtypes == int].tolist()

    col_trans = make_column_transformer(
        (OneHotEncoder(), features_to_encode),
        remainder="passthrough"
    )

    rf_classifier = RandomForestClassifier(min_samples_leaf=50, n_estimators=150, bootstrap=True, oob_score=True,
                                            n_jobs=-1, random_state=42)
    pipe = make_pipeline(col_trans, rf_classifier)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred), 3) * 100} %")
    probs = pipe.predict_proba(X_test)[:, 1]
    print(f'Test ROC AUC  Score: {roc_auc_score(y_test, probs)}')
def main2(df: pd.DataFrame):
    classifier = Classifier(Properties.classifier)
    size = df.shape[0]

    indicesOfTrainingFolds = random.sample(list(range(size)), int(size * 0.666))
    indicesOfTestFolds = list(set(list(range(size))) - set(indicesOfTrainingFolds))

    classifier.buildClassifier(df.iloc[indicesOfTrainingFolds,:])
    test_set = df.iloc[indicesOfTestFolds, :]
    evaluation_info = classifier.evaluateClassifier(test_set)
    score = roc_auc_score(test_set[df.columns[-1]].values,
                          evaluation_info.getScoreDistribution()[:, 1])
    print(score)
    print(f"The accuracy of the model is {round(accuracy_score(test_set[df.columns[-1]].values, evaluation_info.getPredictions()), 3) * 100} %")
    probs = evaluation_info.getScoreDistribution()[:,1] # pipe.predict_proba(X_test)[:, 1]
    print(f'Test ROC AUC  Score: {roc_auc_score(test_set[df.columns[-1]].values, probs)}')

def test_classifier(dataset: Dataset):
    df = dataset.df

    classifier = Classifier(Properties.classifier)
    classifier.buildClassifier(df.iloc[dataset.getIndicesOfTrainingInstances(), :])
    test_set = df.iloc[dataset.getIndicesOfTestInstances(), :]
    eval_info = classifier.evaluateClassifier(test_set)

    acc = eval_info.get_accuracy_score()
    auc = eval_info.get_roc_auc_score()
    print(f"The accuracy of the model is {round(acc, 3) * 100} %")
    print(f'Test ROC AUC  Score: {auc}')


if __name__ == '__main__':
    baseFolder = '/home/itay/Documents/java/ExploreKit/AutomaticFeatureGeneration-master/ML_Background/Datasets/'
    german_credit_dataset_path = baseFolder + "german_credit.arff"
    dataset = Loader().readArff(german_credit_dataset_path, 42, None, None, 1.0)
    df = dataset.df
    # main2(data)
    # main3(data.copy())
    test_classifier2(df.copy())
    test_classifier(dataset.replicateDataset())





