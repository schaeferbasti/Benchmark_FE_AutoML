from typing import Optional

from src.amltk.feature_engineering.ExploreKit.method.Operators.Operator import Operator
from src.amltk.feature_engineering.ExploreKit.method.Operators.UnaryOperators.UnaryOperator import UnaryOperator


class OperatorAssignment:

    def __init__(self, sourceColumns: list, targetColumns: list, operator: Operator, secondaryOperator: Optional[UnaryOperator]):
        self.sourceColumns = sourceColumns
        self.targetColumns = targetColumns
        self.operator = operator
        # a discretizer/normalizer that will be applied on the product of the previous operator
        # this operator is to be applied AFTER the main operator is complete (serves as a discretizer/normalizer)
        self.secondaryOperator = secondaryOperator

        self.filterEvaluatorScore: float = 0
        self.wrapperEvaluatorScore: float = 0
        self.preRankerEvaluatorScore: float = 0

    def getName(self) -> str:
        sb = ''
        sb += '{Sources:['
        sb += ','.join([sCI.name for sCI in self.sourceColumns])
            # sb += sCI.name
            # sb += ','
        sb += '];'
        sb += 'Targets:['
        if self.targetColumns != None:
            sb += ','.join([sCI.name for sCI in self.targetColumns])
            # for tCI in self.targetColumns:
                # sb += tCI.name
                # sb += ','
        sb += '];'
        sb += self.operator.getName()
        if self.secondaryOperator != None:
            sb += ','
            sb += self.secondaryOperator.getName()

        sb += '}'
        return sb

    def getOperator(self):
        return self.operator

    def getSources(self):
        return self.sourceColumns

    def getTargets(self):
        return self.targetColumns

    def getSecondaryOperator(self):
        return self.secondaryOperator

    def getFilterEvaluatorScore(self):
        return self.filterEvaluatorScore

    def setFilterEvaluatorScore(self, score: float):
        self.filterEvaluatorScore = score

    def getWrapperEvaluatorScore(self):
        return self.wrapperEvaluatorScore

    def setWrapperEvaluatorScore(self, score: float):
        self.wrapperEvaluatorScore = score

    def getPreRankerEvaluatorScore(self):
        return self.preRankerEvaluatorScore

    def setPreRankerEvaluatorScore(self, score: float):
        self.preRankerEvaluatorScore = score

    def __str__(self):
        return self.getName()

