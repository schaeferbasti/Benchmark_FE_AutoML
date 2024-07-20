from src.feature_engineering.ExploreKit.method.Evaluation.DatasetBasedAttributes import DatasetBasedAttributes
from src.feature_engineering.ExploreKit.method.Evaluation.FilterEvaluator import FilterEvaluator
from src.feature_engineering.ExploreKit.method.Evaluation.MLAttributeManager import MLAttributeManager
from src.feature_engineering.ExploreKit.method.Data.Dataset import Dataset
from src.feature_engineering.ExploreKit.method.Utils.Logger import Logger
from src.feature_engineering.ExploreKit.method.Properties import Properties


class MLFilterEvaluator(FilterEvaluator):

    analyzedColumns = []
    datasetAttributes = {}

    def __init__(self, dataset: Dataset, name: str):
        super().__init__()
        self.initializeBackgroundModel(dataset, name)

    # Used to create or load the data used by the background model - all the datasets that are evaluated "offline" to create
    # the meta-features classifier.
    def initializeBackgroundModel(self, dataset: Dataset, name: str):
        Logger.Info('Initializing background model for dataset ' + name)
        mlam = MLAttributeManager()
        self.classifier = mlam.getBackgroundClassificationModel(dataset, name, True)

        dba = DatasetBasedAttributes()
        self.datasetAttributes = dba.getDatasetBasedFeatures(dataset, Properties.classifier)

    # def produceScore(self, analyzedDatasets: Dataset, currentScore: ClassificationResults, completeDataset: Dataset, oa: OperatorAssignment, candidateAttribute: pd.Series) -> float:
    #     try:
    #         mlam = MLAttributesManager()
    #         if classifier is None:
    #             LOGGER.error("Classifier is not initialized");
    #             throw new Exception("Classifier is not initialized");
    #         }
    #
    #         //we need to generate the features for this candidate attribute and then run the (previously) calculated classification model
    #         OperatorAssignmentBasedAttributes oaba = new OperatorAssignmentBasedAttributes();
    #         TreeMap<Integer,AttributeInfo> oaAttributes = oaba.getOperatorAssignmentBasedMetaFeatures(analyzedDatasets, oa, properties);
    #         TreeMap<Integer, AttributeInfo> candidateAttributeValuesDependentMetaFeatures = oaba.getGeneratedAttributeValuesMetaFeatures(analyzedDatasets, oa, candidateAttribute, properties);
    #
    #
    #         TreeMap<Integer, AttributeInfo> candidateAttributes = new TreeMap<>(datasetAttributes);
    #         for( AttributeInfo attributeInfo : oaAttributes.values()){
    #             candidateAttributes.put(candidateAttributes.size(), attributeInfo);
    #         }
    #
    #         //We need to add the type of the classifier we're using
    #         AttributeInfo classifierAttribute = new AttributeInfo("Classifier", Column.columnType.Discrete, mlam.getClassifierIndex(properties.getProperty("classifier")), properties.getProperty("classifiersForMLAttributesGeneration").split(",").length);
    #         candidateAttributes.put(candidateAttributes.size(), classifierAttribute);
    #
    #         //In order to have attributes of the same set size, we need to add the class attribute. We don't know the true value, so we set it to negative
    #         AttributeInfo classAttrubute = new AttributeInfo("classAttribute", Column.columnType.Discrete, 0, 2);
    #         int classAtributeKey = candidateAttributes.size();
    #         candidateAttributes.put(classAtributeKey, classAttrubute);
    #
    #         for( AttributeInfo attributeInfo : candidateAttributeValuesDependentMetaFeatures.values()){
    #             candidateAttributes.put(candidateAttributes.size(), attributeInfo);
    #         }
    #
    #         //finally, we need to set the index of the target class
    #         Instances testInstances = mlam.generateValuesMatrix(candidateAttributes);
    #         testInstances.setClassIndex(classAtributeKey);
    #
    #
    #         evaluation = new Evaluation(testInstances);
    #         evaluation.evaluateModel(classifier, testInstances);
    #
    #         //we have a single prediction, so it's easy to process
    #         Prediction prediction = evaluation.predictions().get(0);
    #         ClassificationItem ci = new ClassificationItem((int) prediction.actual(), ((NominalPrediction) prediction).distribution());
    #         return ci.getProbabilities()[analyzedDatasets.getMinorityClassIndex()];
    #     }
    #     catch (Exception ex) {
    #         LOGGER.error("MLFilterEvaluator.produceScore -> Error in ML score generation : " + ex.getMessage());
    #         return -1;
    #     }
    # }


    def recalculateDatasetBasedFeatures(self,  analyzedDatasets: Dataset):
        dba = DatasetBasedAttributes()
        self.datasetAttributes = dba.getDatasetBasedFeatures(analyzedDatasets, Properties.classifier)

    def needToRecalculateScoreAtEachIteration(self) -> bool:
        return True

    def getCopy(self):
        #TODO
        # mlf = MLFilterEvaluator()
        # mlf.setClassifier(this.classifier);
        # mlf.setDatasetAttributes(this.datasetAttributes);
        # mlf.setEvaluation(this.evaluation);
        return None

    def setClassifier(self, classifier):
        self.classifier = classifier

    def setDatasetAttributes(self, datasetAttributes):
        self.datasetAttributes = datasetAttributes

    def setEvaluation(self, evaluation):
        self.evaluation = evaluation
