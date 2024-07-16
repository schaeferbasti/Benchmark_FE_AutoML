from typing import Union, Any

import pandas as pd


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class OperationAssignmentAncestorsSingleton(metaclass=Singleton):
    def __init__(self):
        self.ancestors = {}

    def addAssignment(self, colName, source, target=None):
        if colName not in self.ancestors:
            if target is not None:
                self.ancestors[colName] = {'source': source, 'target': target}
            else:
                self.ancestors[colName] = {'source': source}

    # If bool is True, then there's a source, else, there's no source
    def getSources(self, colName) -> (bool, Union[pd.Series, None]):
        if colName not in self.ancestors:
            return (False, None)

        return (True, self.ancestors[colName]['source'])

    # If bool is True, then there's a target, else there's no target
    def getTargets(self, colName) -> (bool, Union[pd.Series, None]):
        if colName not in self.ancestors:
            return (False, None)

        if 'target' not in self.ancestors[colName]:
            return (False, None)

        return (True, self.ancestors[colName]['target'])
