# https://github.com/giladkatz/ExploreKit
# https://github.com/itayh1/ExploreKitPy

import pandas as pd


from src.amltk.feature_engineering.ExploreKit.method.main import main


def get_xxx_features(train_x, train_y, test_x, test_y) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    # df_train = pd.concat([train_x, train_y], axis=1)
    # df_test = pd.concat([test_x, test_y], axis=1)
    # dataset = pd.concat([df_train, df_test], axis=0)
    # string = "xxx"
    main()
    return train_x, test_x
