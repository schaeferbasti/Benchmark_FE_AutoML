
class Properties:
    numOfFolds=4
    randomSeed=10
    testFoldDesignation='last'
    equalRangeDiscretizerBinsNumber=10
    precisionRecallIntervals=0.05
    FMeausrePoints=[0.3,0.4,0.5,0.6]
    filterApproach='MLFilterEvaluator'
    wrapperApproach='AucWrapperEvaluator'
    rankerApproach='FilterScoreWithExclusionsRanker'
    classifier='RandomForest'
    unaryOperators='StandardScoreUnaryOperator,EqualRangeDiscretizerUnaryOperator,HourOfDayUnaryOperator,DayOfWeekUnaryOperator,IsWeekendUnaryOperator'
    nonUnaryOperators='AddBinaryOperator' #,\
    # DivisionBinaryOperator,MultiplyBinaryOperator,SubtractBinaryOperator,GroupByThenAvg,GroupByThenStdev,GroupByThenCount,GroupByThenMax,GroupByThenMin,\
    # TimeBasedGroupByThenCountAndAvg_180,TimeBasedGroupByThenCountAndCount_180,TimeBasedGroupByThenCountAndMax_180,TimeBasedGroupByThenCountAndMin_180,\
    # TimeBasedGroupByThenCountAndStdev_180,TimeBasedGroupByThenCountAndAvg_1440,TimeBasedGroupByThenCountAndCount_1440,TimeBasedGroupByThenCountAndMax_1440,\
    # TimeBasedGroupByThenCountAndMin_1440,TimeBasedGroupByThenCountAndStdev_1440'
    resultsFilePath='C:/workspace/data/results/'
    maxNumOfAttsInOperatorSource=2
    writeAttributesToFile=False
    numOfThreads=1
    maxNumOfWrapperEvaluationsPerIteration=15000
    maxNumberOfDiscreteValuesForInclusionInSet=1000
    classifiersForMLAttributesGeneration='RandomForest'
    #/home/itay/Documents/java/ExploreKit/AutomaticFeatureGeneration-master/ML_Background
    # /home/itay/Documents/EKpy/ML_Background
    DatasetInstancesFilesLocation='ML_Background/Candidate_attributes/'
    backgroundClassifierLocation='ML_Background/Background_classifiers_and_arffs/'
    originalBackgroundDatasetsLocation='/home/itay/Documents/EKpy/ML_Background/DatasetsForMetaModel/'
    preRankedAttributesToGenerate=50000
    usePreRanker=False
    # mySQLUrl=jdbc:mysql://localhost:3306/explorekit
    # mySQLUsername=java
    # mySQLPassword=password