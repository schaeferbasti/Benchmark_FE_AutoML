import json
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from src.amltk.datasets.Datasets import get_dataset


class Dataset:

    TASK_MAP = {
        'R': 'regression',
        'C': 'classification'
    }

    def __init__(self, name):
        name = "pima"
        project_path = os.path.abspath(os.path.dirname(__file__))
        data_path = f"{project_path[:project_path.find('DIFER')]}DIFER/method/data"
        path = Path(data_path)
        # """
        self.data = pd.read_csv(
            path / f"{name}.csv",
            header=None
        )
        self.data.columns = self.data.columns.astype(str)
        with open(path / f"{name}.json", 'r') as f:
            self.meta = json.load(f)
        # """
        """
        x_train, y_train, x_valid, y_valid, task_hint, name = get_dataset(16)
        df_train = pd.concat([x_train, y_train], axis=1)
        df_valid = pd.concat([x_valid, y_valid], axis=1)
        df = pd.concat([df_train, df_valid], axis=0)

        index_list = list(range(df.shape[1]))
        index_list = [str(e) for e in index_list]
        print("Index list:", index_list)
        df.columns = index_list

        if task_hint == 'regression':
            task = "R"
        elif task_hint == 'classification':
            task = "C"
        metadata = {}
        for i, col in enumerate(df.columns):
            if pd.api.types.is_numeric_dtype(df[col]):
                metadata[str(i)] = "num"
            else:
                metadata[str(i)] = "cat"
        metadata["task"] = task

        self.dataset_name = name
        self.data = df
        print(metadata)
        self.meta = metadata
        """
        self._x = None
        self._y = None
        self._label_encoder = LabelEncoder()

    @property
    def instances(self):
        if self._x is None:
            self._x = self.data.iloc[:, :-1]
        return self._x

    @property
    def labels(self):
        if self._y is None:
            self._y = self.data.iloc[:, -1]
            if self.task == Dataset.TASK_MAP['C']:
                self._y = self._label_encoder.fit_transform(self._y)
        return self._y

    @property
    def task(self):
        return Dataset.TASK_MAP[self.meta['task']]

    @property
    def time_budget(self):
        return self.meta.get('time_budget', 24 * 3600)

    @property
    def features(self):
        return self.instances.columns


if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score

    dataset = Dataset('spectf')
    x = dataset.data.iloc[:, :-1]
    y = dataset.data.iloc[:, -1]
    y = LabelEncoder().fit_transform(y)
    s = cross_val_score(
        RandomForestClassifier(n_estimators=10, random_state=0),
        x, y,
        scoring='f1_micro',
        cv=5
    ).mean()
    print(dataset.task)
    print(s)
