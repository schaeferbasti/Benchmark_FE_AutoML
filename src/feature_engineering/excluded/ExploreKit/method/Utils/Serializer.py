import pickle

class Serializer:
    @staticmethod
    def Serialize(filePath: str, obj):
        with open(filePath, 'wb') as file:
            pickle.dump(obj, file)

    # Load from file
    @staticmethod
    def Deserialize(filePath: str):
        with open(filePath, 'rb') as file:
            obj = pickle.load(file)
            return obj

