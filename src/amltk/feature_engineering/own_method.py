import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, normalize, binarize, QuantileTransformer


def preprocess_data(test_x, train_x):
    # Preprocessing
    # Numerize categorical columns
    print("Preprocess categorical values, NaN and negative values")
    cat_columns = train_x.select_dtypes(['category']).columns
    train_x[cat_columns] = train_x[cat_columns].apply(lambda x: pd.factorize(x, use_na_sentinel=True)[0])
    cat_columns = test_x.select_dtypes(['category']).columns
    test_x[cat_columns] = test_x[cat_columns].apply(lambda x: pd.factorize(x, use_na_sentinel=True)[0])
    # Replace NaN and negative values by mean
    imp1 = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp1.fit(train_x)
    train_x = imp1.transform(train_x)
    imp2 = SimpleImputer(missing_values=-1, strategy='mean')
    imp2.fit(train_x)
    train_x = imp2.transform(train_x)
    imp3 = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp3.fit(test_x)
    test_x = imp3.transform(test_x)
    imp4 = SimpleImputer(missing_values=-1, strategy='mean')
    imp4.fit(test_x)
    test_x = imp4.transform(test_x)
    return test_x, train_x


def get_sklearn_features(train_x, test_x, train_y, test_y) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    # ****** COMBINE TWO FEATURES (following the Expand & Reduce Strategy) ******
    # 1. Feature Generation
    #   a. Create Polynomial Features (PolynomialFeatures)
    #   b. Dimensionality Reduction (PCA, TruncatedSVD)
    #   c. Custom Feature Engineering (FunctionTransformer, TransformerMixin)
    # 2. Feature Selection
    #   a. SelectKBest
    #   b. SelectPercentile
    """ Only do transform for test data, no fit_transform --> No learning from test data """

    columns_train_x = train_x.columns
    columns_test_x = test_x.columns

    test_x, train_x = preprocess_data(test_x, train_x)

    # Generate Polynomial features
    print("Generate new features")
    pf = PolynomialFeatures(degree=2, interaction_only=True)
    train_x = pf.fit_transform(train_x)
    test_x = pf.transform(test_x)

    # Normalize
    train_x = normalize(train_x, axis=0)
    test_x = normalize(test_x, axis=0)

    # Binarize
    train_x = binarize(train_x)
    test_x = binarize(test_x)

    # Quantile Transformer
    qt = QuantileTransformer(random_state=0)
    train_x = qt.fit_transform(train_x)
    test_x = qt.transform(test_x)

    # Select Best Features
    print("Select best features")
    train_x = SelectKBest(chi2, k=38).fit_transform(train_x, train_y)
    test_x = SelectKBest(chi2, k=38).fit_transform(test_x, test_y)

    # Transform to DataFrame again
    train_x = pd.DataFrame(train_x, columns=columns_train_x)
    test_x = pd.DataFrame(test_x, columns=columns_test_x)

    return train_x, test_x
