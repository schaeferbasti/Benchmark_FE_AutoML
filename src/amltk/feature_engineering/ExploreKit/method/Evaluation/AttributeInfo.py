

class AttributeInfo:

    def __init__(self,attName: str, attType, attValue, numOfValues: int):
        self.attributeName = attName
        self.attributeType = attType
        self.value = attValue
        self.numOfDiscreteValues = numOfValues

    def __repr__(self):
        return f"Name: {self.attributeName}, Type: {self.attributeType}, Value: {self.value}, num_discrete_values: {self.numOfDiscreteValues}"
