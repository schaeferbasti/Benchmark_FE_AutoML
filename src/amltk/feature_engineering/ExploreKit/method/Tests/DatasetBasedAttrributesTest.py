from Utils.Loader import Loader
from Evaluation.DatasetBasedAttributes import DatasetBasedAttributes
from Properties import Properties

if __name__=='__main__':
    datasets = []
    classAttributeIndices = {}
    baseFolder = '/home/itay/Documents/java/ExploreKit/AutomaticFeatureGeneration-master/ML_Background/Datasets/'
    # datasets.add("/global/home/users/giladk/Datasets/heart.arff");
    # datasets.add("/global/home/users/giladk/Datasets/cancer.arff");
    # datasets.add("/global/home/users/giladk/Datasets/contraceptive.arff");
    # datasets.add("/global/home/users/giladk/Datasets/credit.arff");
    datasets.append("german_credit.arff")
    # datasets.add("/global/home/users/giladk/Datasets/diabetes_old.arff");
    # datasets.add("/global/home/users/giladk/Datasets/Diabetic_Retinopathy_Debrecen.arff");
    # datasets.add("/global/home/users/giladk/Datasets/horse-colic.arff");
    # datasets.add("/global/home/users/giladk/Datasets/Indian_Liver_Patient_Dataset.arff");
    # datasets.add("/global/home/users/giladk/Datasets/seismic-bumps.arff");
    # datasets.add("/global/home/users/giladk/Datasets/cardiography_new.arff");

    loader = Loader()
    randomSeed = 42
    dataset = loader.readArff(baseFolder + datasets[0], randomSeed, None, None, 0.66)

    dba = DatasetBasedAttributes()
    dba.getDatasetBasedFeatures(dataset, Properties.classifier)