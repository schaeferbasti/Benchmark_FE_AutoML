import arff
import pandas as pd
from pandas import CategoricalDtype


class ArffManager:
    @staticmethod
    def SaveArff(arffFilename: str, df: pd.DataFrame, relation: str):
        attributes = [
            (str(j), 'NUMERIC') if df[j].dtypes in ['int64', 'float64'] else (j, df[j].unique().astype(str).tolist()) for j
            in df]

        arff_dic = {
            'attributes': attributes,
            'data': df.values,
            'relation': relation,
            'description': ''
        }
        with open(arffFilename, "w", encoding="utf8") as f:
            arff.dump(arff_dic, f)

    @staticmethod
    def LoadArff(arffFilename: str) -> pd.DataFrame:
        with open(arffFilename, 'r') as f:
            data = arff.load(f)
            df = ArffManager._get_dataframe_by_attrs(data['data'], data['attributes'])
            return df

    @staticmethod
    def _get_dataframe_by_attrs(data, attributes):
        df = pd.DataFrame(data, columns=[attr_name for attr_name, _ in attributes])
        for attr_name, attr_type in attributes:
            if type(attr_type) == str:
                if attr_type.upper() in ['NUMERIC', 'REAL']:
                    df[attr_name] = df[attr_name].astype(float)
            elif type(attr_type) == list:
                df[attr_name] = df[attr_name].astype(CategoricalDtype(attr_type))
            else:
                raise Exception(f'Unknown attribute type while loading arff: "{attr_type}"')
        return df
