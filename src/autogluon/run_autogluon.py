import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split

from src.datasets.Splits import get_splits

datasets = ["abalone"]
methods = ["autofeat"]

for name in datasets:
    for method in methods:
        data = TabularDataset(f'src/datasets/feature_engineered_datasets/{name}_{method}.csv')
        X = data.drop('target', axis=1)
        y = data['target']

        train_x, train_y, test_x, test_y = train_test_split(X, y, test_size=0.1, random_state=42)
        train_x, train_y, test_x, test_y = get_splits(train_x, train_y, test_x, test_y)
        train_data = pd.concat([train_x, train_y], axis=1)
        test_data = pd.concat([test_x, test_y], axis=1)
        train_data = TabularDataset(train_data)
        test_data = TabularDataset(test_data)

        label = train_y.name
        predictor = TabularPredictor(label=label).fit(train_data)

        y_pred = predictor.predict(test_data.drop(columns=[label]))
        y_pred.head()
        predictor.evaluate(test_data, silent=True)
