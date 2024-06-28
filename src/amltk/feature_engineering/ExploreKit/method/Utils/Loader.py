from random import Random

from scipy.io import arff
import pandas as pd

from src.amltk.feature_engineering.ExploreKit.method.Utils.ArffManager import ArffManager
from src.amltk.feature_engineering.ExploreKit.method.Utils.Logger import Logger
from src.amltk.feature_engineering.ExploreKit.method.Data.Dataset import Dataset
from src.amltk.feature_engineering.ExploreKit.method.Properties import Properties
from src.amltk.feature_engineering.ExploreKit.method.Data.Fold import Fold


class Loader:

    def readArffAsDataframe(self, filePath: str) -> pd.DataFrame:
        # data = arff.loadarff(filePath)
        # df = pd.DataFrame(data[0])
        df = ArffManager.LoadArff(filePath)
        return df

    def readArff(self, filePath: str, randomSeed: int, distinctValIndices: list, classAttIndex: str, trainingSetPercentageOfDataset: float) -> Dataset:

        try:
            data = arff.loadarff(filePath)
            df = pd.DataFrame(data[0])
            df = self._setDataframeStringsToDiscrete(df)

            Logger.Info(f'num of attributes: {len(df.keys())}')
            Logger.Info(f'num of instances: {len(df.values)}')


            if (classAttIndex == None) or (classAttIndex == ''):
                targetClassName = df.keys()[-1]
            else:
                targetClassName = classAttIndex
            # df[targetClassName] = df[targetClassName].str.decode("utf-8")

            if distinctValIndices == None:
                folds = self.GenerateFolds(df[targetClassName], randomSeed, trainingSetPercentageOfDataset)
            else:
                raise Exception("No support for distinct values")

            distinctValColumnInfos = []
            # if distinctValIndices != None:
            #     for distinctColumnIndex in distinctValIndices:
            #         distinctValColumnInfos.append(df[distinctColumnIndex])

            # Fially, we can create the Dataset object
            return Dataset(df, folds, targetClassName, data[1].name, randomSeed, Properties.maxNumberOfDiscreteValuesForInclusionInSet)

        except Exception as ex:
            Logger.Error(f'Exception in readArff. message: {ex}')
            raise

    def GenerateFolds(self, targetColumnInfo, randomSeed: int, trainingSetPercentage: float) -> list: #List<Fold>

        # Next, we need to get the number of classes (we assume the target class is discrete)
        numOfClasses = targetColumnInfo.unique().shape[0]

        # Store the indices of the instances, partitioned by their class
        # itemIndicesByClass = [[] for i in range(numOfClasses)]
        itemIndicesByClass = {key:[] for key in targetColumnInfo.unique()}

        for i in range(targetColumnInfo.shape[0]):
            instanceClass = targetColumnInfo[i]
            itemIndicesByClass[instanceClass].append(i)

        # Now we calculate the number of instances from each class we want  to assign to fold
        numOfFolds = Properties.numOfFolds
        maxNumOfInstancesPerTrainingClassPerFold = {} # np.arr numOfClasses];
        maxNumOfInstancesPerTestClassPerFold = {} # new double[numOfClasses];
        for key in itemIndicesByClass.keys():
            # If the training set overall size (in percentages) is predefined, use it. Otherwise, just create equal folds
            if trainingSetPercentage == -1:
                maxNumOfInstancesPerTrainingClassPerFold[key] = len(itemIndicesByClass[key])/numOfFolds
                maxNumOfInstancesPerTestClassPerFold[key] = len(itemIndicesByClass[key])/numOfFolds

            else:
                # The total number of instances, multipllied by the training percentage and then divided by the number of the TRAINING folds
                maxNumOfInstancesPerTrainingClassPerFold[key] = (len(itemIndicesByClass[key]) * trainingSetPercentage /(numOfFolds-1))
                maxNumOfInstancesPerTestClassPerFold[key] = (len(itemIndicesByClass[key]) - maxNumOfInstancesPerTrainingClassPerFold[key])

        # We're using a fixed seed so we can reproduce our results
        # int randomSeed = Integer.parseInt(properties.getProperty("randomSeed"))
        rnd = Random(randomSeed)

        # Now create the Fold objects and start filling them
        folds = [] #new ArrayList<>(numOfClasses);
        for i in range(numOfFolds):
            isTestFold = self.designateFoldAsTestSet(numOfFolds, i, Properties.testFoldDesignation)
            fold = Fold(targetColumnInfo.unique(), isTestFold)
            folds.append(fold)


        for i in range(targetColumnInfo.shape[0]):
            instanceClass = targetColumnInfo[i]

            foundAssignment = False
            exploredIndices = []
            while not foundAssignment:
                # We now randomly sample a fold and see whether the instance can be assigned to it. If not, sample again
                foldIdx = rnd.randrange(numOfFolds)
                if str(foldIdx) not in exploredIndices:
                    exploredIndices.append(str(foldIdx))

                # Now see if the instance can be assigned to the fold
                fold = folds[foldIdx]
                if not fold.isTestFold:
                    if fold.getNumOfInstancesPerClass(instanceClass) < maxNumOfInstancesPerTrainingClassPerFold[instanceClass] or len(exploredIndices) == numOfFolds:
                        fold.addInstance(i, instanceClass)
                        foundAssignment = True

                else:
                    if fold.getNumOfInstancesPerClass(instanceClass) < maxNumOfInstancesPerTestClassPerFold[instanceClass] or len(exploredIndices) == numOfFolds:
                        fold.addInstance(i, instanceClass)
                        foundAssignment = True

        return folds

    def getFolds(df: pd.DataFrame, targetClassname: str, k: int) -> list:
        numOfClasses = df[targetClassname].nunique()
        return [Fold(numOfClasses, False) for i in range(k)]

    def designateFoldAsTestSet(self, numOfFolds: int, currentFoldIdx: int, designationMethod: str):
         if designationMethod == 'last':
             if currentFoldIdx == (numOfFolds - 1):
                return True
             else:
                return False
         else:
             raise Exception("unknown test fold selection method")

    def _setDataframeStringsToDiscrete(self, df: pd.DataFrame) -> pd.DataFrame:
        categoricalColumns = df.select_dtypes(include='object').columns
        for colName in categoricalColumns:
            categoryToIntMap = {cat: i for i, cat in enumerate(sorted(df[colName].unique()))}
            df[colName].replace(categoryToIntMap, inplace=True)
        return df