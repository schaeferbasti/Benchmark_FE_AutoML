import random

import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from Env.Environment import EnvGym
from Env.ExploreKitEnv import ExploreKitEnv
from PytorchDQN import DQN

from ..Properties import Properties
from ..Utils.Loader import Loader


def get_dataset():
    baseFolder = '/home/itay/Documents/java/ExploreKit/AutomaticFeatureGeneration-master/ML_Background/Datasets/'
    german_credit_dataset_path = baseFolder + "german_credit.arff"

    loader = Loader()
    randomSeed = 42
    dataset = loader.readArff(german_credit_dataset_path, randomSeed, None, None, 0.66)

    discrete_columns = dataset.getAllColumnsOfType(pd.api.types.is_integer_dtype, False)
    numeric_columns = dataset.getAllColumnsOfType(pd.api.types.is_float_dtype, False)

    features = random.sample(discrete_columns, 2)
    features.extend(random.sample(numeric_columns, 2))
    return dataset, features

def main():
    dataset, features = get_dataset()
    env = EnvGym()
    # env = ExploreKitEnv(dataset, features, Properties.classifier.split(',')[0])
    model = DQN(env)
    model.train()

if __name__ == '__main__':
    main()