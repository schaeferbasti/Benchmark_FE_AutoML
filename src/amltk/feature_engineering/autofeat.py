from autofeat.autofeat import AutoFeatRegressor, AutoFeatClassifier
from src.amltk.datasets.get_datasets import preprocess_dataframe


def get_autofeat_features(train_x, train_y, test_x, task_hint):
    train_x = preprocess_dataframe(train_x)
    #test_x = preprocess_dataframe(test_x)
    #train_y = preprocess_dataframe(train_y)
    if task_hint == 'regression':
        autofeat_regression = AutoFeatRegressor(
            verbose=1,
            # categorical_cols=cat_cols,
            feateng_steps=1,
            featsel_runs=0,
            always_return_numpy=False,
            transformations=("1/", "exp", "log", "abs", "sqrt", "^2", "^3")
        )
        train_x = autofeat_regression.fit_transform(train_x, train_y)
    elif task_hint == "classification":
        autofeat_classification = AutoFeatClassifier(
            verbose=1,
            # categorical_cols=cat_cols,
            feateng_steps=1,
            featsel_runs=0,
            always_return_numpy=False,
            transformations=("1/", "exp", "log", "abs", "sqrt", "^2", "^3"))
        train_x = autofeat_classification.fit_transform(train_x, train_y)
    print(train_x)
    return train_x, test_x
