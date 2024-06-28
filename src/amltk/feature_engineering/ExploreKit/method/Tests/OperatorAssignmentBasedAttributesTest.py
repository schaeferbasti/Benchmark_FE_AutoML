import numpy as np
import pandas as pd

from Evaluation.OperatorAssignment import OperatorAssignment
from Evaluation.OperatorAssignmentBasedAttributes import OperatorAssignmentBasedAttributes
from Operators.BinaryOperators.AddBinaryOperator import AddBinaryOperator
from Operators.BinaryOperators.BinaryOperator import BinaryOperator

from Utils.Loader import Loader

def test_without_generating_feature(dataset):
    binary_operator = AddBinaryOperator()
    # source = [dataset.df.iloc[:, 1]]
    # target = [dataset.df.iloc[:, 4]]
    # df = pd.DataFrame(np.arange(0, 100, size=(100, 2)), columns=list('xy'), dtype=float)
    source = [pd.Series(np.arange(1000), dtype=float)]
    target = [pd.Series(np.arange(1000), dtype=float)]

    if not binary_operator.isApplicable(dataset, source, target):
        print("Bad operator")
        exit(0)
    oa = OperatorAssignment(source, target, binary_operator, None)

    oaba = OperatorAssignmentBasedAttributes()
    features = oaba.getOperatorAssignmentBasedMetaFeatures(dataset, oa)
    print(features)


def test_with_generating_feature(dataset):
    binary_operator = BinaryOperator()

    # if not binary_operator.isApplicable(dataset, source, target):
    #     print("Bad operator")
    #     exit(0)
    oa = OperatorAssignment([], [], binary_operator, None)

    oaba = OperatorAssignmentBasedAttributes()
    features = oaba.getGeneratedAttributeValuesMetaFeatures(dataset, oa, dataset.df.iloc[:, 0])
    print(features)


if __name__ == '__main__':
    baseFolder = '/home/itay/Documents/java/ExploreKit/AutomaticFeatureGeneration-master/ML_Background/Datasets/'
    german_credit_dataset_path = baseFolder + "german_credit.arff"

    loader = Loader()
    randomSeed = 42
    dataset = loader.readArff(german_credit_dataset_path, randomSeed, None, None, 0.66)

    # test_without_generating_feature(dataset)

    test_with_generating_feature(dataset)