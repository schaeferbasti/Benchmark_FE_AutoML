
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

import arff

def get_dataframe_by_attrs(data, attributes):
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

def test_load():
    # raw_data = loadarff('../ML_Background/Datasets/diabetes_old.arff')
    # raw_data = loadarff('../ML_Background/Datasets/german_credit.arff')

    data = arff.load(open('../ML_Background/Datasets/german_credit.arff','r'))
    df = get_dataframe_by_attrs(data['data'], data['attributes'])
    print(df)
    return df

def test_save(df: pd.DataFrame):
    attributes = [
        (j, 'NUMERIC') if df[j].dtypes in ['int64', 'float64'] else (j, df[j].unique().astype(str).tolist()) for j in df]

    arff_dic = {
        'attributes': attributes,
        'data': df.values,
        'relation': 'myRel',
        'description': ''
    }
    with open("myfile.arff", "w", encoding="utf8") as f:
        arff.dump(arff_dic, f)

def main():
    df = test_load()
    test_save(df)

if __name__ == '__main__':
    main()