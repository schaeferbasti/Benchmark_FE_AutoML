import random

import pandas as pd

from Properties import Properties
from RL.Env.ExploreKitEnv import ExploreKitEnv
from Utils.Loader import Loader

if __name__ == '__main__':
    baseFolder = '/home/itay/Documents/java/ExploreKit/AutomaticFeatureGeneration-master/ML_Background/Datasets/'
    german_credit_dataset_path = baseFolder + "german_credit.arff"

    loader = Loader()
    randomSeed = 42
    dataset = loader.readArff(german_credit_dataset_path, randomSeed, None, None, 0.66)

    discrete_columns = dataset.getAllColumnsOfType(pd.api.types.is_integer_dtype, False)
    numeric_columns = dataset.getAllColumnsOfType(pd.api.types.is_float_dtype, False)

    features = random.sample(discrete_columns, 2)
    features.extend(random.sample(numeric_columns, 2))

    env = ExploreKitEnv(dataset, features, Properties.classifier.split(',')[0])
