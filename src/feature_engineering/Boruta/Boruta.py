# https://github.com/numb3r33/fgcnn
from boruta import BorutaPy

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_boruta_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators="auto", verbose=2)

    train_x_np = train_x.values
    train_y_np = train_y.values
    test_x_np = test_x.values

    feat_selector.fit(train_x_np, train_y_np.ravel())

    # Transform the training and testing data
    train_x_selected = feat_selector.transform(train_x_np)
    test_x_selected = feat_selector.transform(test_x_np)

    # Convert the transformed data back to DataFrame
    train_x_selected_df = pd.DataFrame(train_x_selected, columns=train_x.columns[feat_selector.support_])
    test_x_selected_df = pd.DataFrame(test_x_selected, columns=test_x.columns[feat_selector.support_])

    print(train_x_selected_df.columns)
    print(test_x_selected_df.columns)
    return train_x_selected_df, test_x_selected_df
