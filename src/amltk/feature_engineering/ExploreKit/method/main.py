import os
from pathlib import Path

from sklearn.model_selection import KFold
import pandas as pd
from scipy.io import arff

from src.amltk.datasets.Datasets import get_dataset
from src.amltk.feature_engineering.ExploreKit.method.Search.FilterWrapperHeuristicSearch import FilterWrapperHeuristicSearch
from src.amltk.feature_engineering.ExploreKit.method.Utils.Loader import Loader


def getFolds(df: pd.DataFrame, k: int) -> list:
    #TODO: make it Stratified-KFold
    cv = KFold(n_splits=k, shuffle=True, random_state=20)
    cv.get_n_splits()
    folds = []
    for train_index, test_index in cv.split(df):
        folds.append(test_index)
    return folds


def main(df, name):
    datasets = []
    classAttributeIndices = {}
    """
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    datasets.append("ML_Background/Datasets/german_credit.arff")

    loader = Loader()
    randomSeed = 42
    for i in range(1):
        for datasetPath in datasets:
            abs_file_path = os.path.join(script_dir, datasetPath)
            if datasetPath not in classAttributeIndices.keys():
                dataset = loader.readArff(abs_file_path, randomSeed, None, None, 0.66)
            else:
                dataset = loader.readArff(abs_file_path, randomSeed, None, classAttributeIndices[datasetPath], 0.66)
    """
    exp = FilterWrapperHeuristicSearch(15)
    dataset, candidate_attributes = exp.run(df, "", name)
    return dataset, candidate_attributes


if __name__ == '__main__':
    train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=16)
    df_train = pd.concat([train_x, train_y], axis=1)
    df_test = pd.concat([test_x, test_y], axis=1)
    df = pd.concat([df_train, df_test], axis=0)
    main(df, name)
