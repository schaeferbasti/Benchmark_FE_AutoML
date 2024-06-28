import os

from src.amltk.feature_engineering.ExploreKit.method.Evaluation.AucWrapperEvaluator import AucWrapperEvaluator
from src.amltk.feature_engineering.ExploreKit.method.Data.Dataset import Dataset
from src.amltk.feature_engineering.ExploreKit.method.Utils.Date import Date
from src.amltk.feature_engineering.ExploreKit.method.Evaluation.FilterEvaluator import FilterEvaluator
from src.amltk.feature_engineering.ExploreKit.method.Evaluation.InformationGainFilterEvaluator import InformationGainFilterEvaluator
from src.amltk.feature_engineering.ExploreKit.method.Utils.Logger import Logger
from src.amltk.feature_engineering.ExploreKit.method.Evaluation.MLFilterEvaluator import MLFilterEvaluator
from src.amltk.feature_engineering.ExploreKit.method.Properties import Properties


class Search:
    # private static Logger LOGGER = Logger.getLogger(Search.class.getName());


    # Begins the generation and evaluation of the candidate attributes
    def run(self, dataset: Dataset, runInfo: str):
        raise NotImplementedError('Search is an abstract class')

    # Used to run Weka on the dataset and produce all relevant statistics.
    # IMPORTANT: we currently assume that the target class is discrete
    def evaluateDataset(self, dataset: Dataset):
        raise NotImplementedError('Search is an abstract class')

    # Returns the requested wrapper (initialized)
    def getWrapper(self, wrapperName: str) ->  AucWrapperEvaluator:
        if "AucWrapperEvaluator" == wrapperName:
            return AucWrapperEvaluator()

        raise Exception("Unidentified wrapper")

    # Returns the requested filter (initialized)
    def getFilter(self, filterName: str, dataset: Dataset) -> FilterEvaluator:
        Logger.Info("Getting filter evaluator - " + filterName)
        # switch
        try:
            return {
            "InformationGainFilterEvaluator": InformationGainFilterEvaluator(),
            "MLFilterEvaluator": MLFilterEvaluator(dataset)
            }[filterName]
        except:
            raise Exception("Unidentified evaluator")

    # Returns the requested ranker filter
    def getRankerFilter(self, rankerFilterName: str):
        # switch
        try:
            return {
                # "FilterScoreRanker": FilterScoreRanker(),
                # "WrapperScoreRanker": WrapperScoreRanker(),
                # "FilterScoreWithExclusionsRanker": FilterScoreWithExclusionsRanker()
            }
        except:
            raise Exception("Unidentified rankerFilter")


    def writeFinalStatisticsToResultsFile(self, datasetName: str, runInfo: str, experimentStartTime: Date, totalNumOfWrapperEvaluations: int):
        filename= Properties.resultsFilePath + datasetName + runInfo + ".csv"
        fw = open(filename, 'a')

        experimentEndTime = Date()
        diff = experimentEndTime - experimentStartTime

        newLine = os.linesep
        fw.write(newLine)
        fw.write(f"Total Run Time:{newLine}")
        fw.write(f"Full time: {'{:0>8}'.format(str(diff))}{newLine}")
        fw.write(f"Number of minutes: {divmod(diff.seconds, 60)[0]}{newLine}")
        fw.write(f"Number of seconds: {diff.seconds}{newLine}")
        fw.write(f"Total number of evaluated attribtues: {str(totalNumOfWrapperEvaluations)}")
        fw.close()
