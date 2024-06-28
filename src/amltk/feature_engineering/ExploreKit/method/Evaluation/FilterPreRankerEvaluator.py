from src.amltk.feature_engineering.ExploreKit.method.Evaluation.AttributeInfo import AttributeInfo
from src.amltk.feature_engineering.ExploreKit.method.Evaluation.ClassificationItem import ClassificationItem
from src.amltk.feature_engineering.ExploreKit.method.Evaluation.ClassificationResults import ClassificationResults
from src.amltk.feature_engineering.ExploreKit.method.Evaluation.Classifier import Classifier
from src.amltk.feature_engineering.ExploreKit.method.Data.Dataset import Dataset
from src.amltk.feature_engineering.ExploreKit.method.Evaluation.DatasetBasedAttributes import DatasetBasedAttributes
from src.amltk.feature_engineering.ExploreKit.method.Utils.Logger import Logger
from src.amltk.feature_engineering.ExploreKit.method.Evaluation import MLAttributeManager
# from src.amltk.feature_engineering.ExploreKit.method.Evaluation.MLAttributeManager import MLAttributeManager
from src.amltk.feature_engineering.ExploreKit.method.Operators.Operator import outputType
from src.amltk.feature_engineering.ExploreKit.method.Evaluation.OperatorAssignment import OperatorAssignment
from src.amltk.feature_engineering.ExploreKit.method.Evaluation.OperatorAssignmentBasedAttributes import OperatorAssignmentBasedAttributes
from src.amltk.feature_engineering.ExploreKit.method.Properties import Properties


class FilterPreRankerEvaluator:

    def __init__(self, dataset: Dataset):
        classifier, datasetAttributes = self.initializeBackgroundModel(dataset)
        self.classifier: Classifier = classifier
        self.datasetAttributes: dict = datasetAttributes

    def initializeBackgroundModel(self, dataset: Dataset):
        Logger.Info("Initializing background model for pre-ranking process")
        mlam = MLAttributeManager.MLAttributeManager()
        classifier = mlam.getBackgroundClassificationModel(dataset, False)

        dba = DatasetBasedAttributes()
        datasetAttributes = dba.getDatasetBasedFeatures(dataset, Properties.classifier)
        return classifier, datasetAttributes

    def produceScore(self, analyzedDatasets: Dataset,  currentScore: ClassificationResults,
                     completeDataset: Dataset,  oa:OperatorAssignment,  candidateAttribute) -> float:
        try:
            mlam = MLAttributeManager.MLAttributeManager()
            if self.classifier == None:
                Logger.Error("Classifier is not initialized")
                raise Exception("Classifier is not initialized")

            # we need to generate the features for this candidate attribute and then run the (previously) calculated classification model
            oaba = OperatorAssignmentBasedAttributes()
            oaAttributes: dict = {} #oaba.getOperatorAssignmentBasedMetaFeatures(analyzedDatasets, oa)


            candidateAttributes = {k:v for k,v in self.datasetAttributes.items()}
            for attributeInfo in oaAttributes.values():
                candidateAttributes[len(candidateAttributes)] = attributeInfo


            # We need to add the type of the classifier we're using
            classifierAttribute = AttributeInfo("Classifier", outputType.Discrete, Properties.classifier, len(Properties.classifiersForMLAttributesGeneration.split(",")))
            candidateAttributes[len(candidateAttributes)] = classifierAttribute

            # In order to have attributes of the same set size, we need to add the class attribute. We don't know the true value, so we set it to negative
            classAttrubute = AttributeInfo("classAttribute", outputType.Discrete, 0, 2)
            candidateAttributes[len(candidateAttributes)] = classAttrubute


            # finally, we need to set the index of the target class
            testInstances = mlam.generateValuesMatrix(candidateAttributes)
            # testInstances.setClassIndex(classAtributeKey);

            # evaluation = Evaluation(testInstances);
            # evaluation.evaluateModel(classifier, testInstances);


            # we have a single prediction, so it's easy to process
            evaluationInfo = self.classifier.evaluateClassifier(testInstances)
            prediction = evaluationInfo.predictions[0] #.predictions().get(0);
            ci = ClassificationItem(prediction.actual(), prediction.distribution())
            return ci.getProbabilities()[analyzedDatasets.getMinorityClassIndex()]

        except Exception as ex:
            Logger.Warn("oa working on " + oa.getName())

            Logger.Error("FilterPreRankerEvaluator.produceScore -> Error in ML score generation : " + str(ex))
            return -1