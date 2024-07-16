# https://github.com/Bonidia/BioAutoML

import pandas as pd

import lightgbm as lgb

from catboost import CatBoostClassifier
from hyperopt import hp, Trials, fmin, tpe, STATUS_OK
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer


global global_train_y


def get_bioautoml_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    estimations = 50
    train_x, test_x = feature_engineering(estimations, train_x, train_y, test_x)
    return train_x, test_x


def feature_engineering(estimations, train, train_labels, test):
    """Automated Feature Engineering - Bayesian Optimization"""

    global df_x, labels_y

    print('Automated Feature Engineering - Bayesian Optimization')

    df_x = train
    labels_y = train_labels
    df_test = test

    param = {'NAC': [0, 1], 'DNC': [0, 1],
             'TNC': [0, 1], 'kGap_di': [0, 1], 'kGap_tri': [0, 1],
             'ORF': [0, 1], 'Fickett': [0, 1],
             'Shannon': [0, 1], 'FourierBinary': [0, 1],
             'FourierComplex': [0, 1], 'Tsallis': [0, 1],
             'Classifier': [0, 1, 2]}

    space = {'NAC': hp.choice('NAC', [0, 1]),
             'DNC': hp.choice('DNC', [0, 1]),
             'TNC': hp.choice('TNC', [0, 1]),
             'kGap_di': hp.choice('kGap_di', [0, 1]),
             'kGap_tri': hp.choice('kGap_tri', [0, 1]),
             'ORF': hp.choice('ORF', [0, 1]),
             'Fickett': hp.choice('Fickett', [0, 1]),
             'Shannon': hp.choice('Shannon', [0, 1]),
             'FourierBinary': hp.choice('FourierBinary', [0, 1]),
             'FourierComplex': hp.choice('FourierComplex', [0, 1]),
             'Tsallis': hp.choice('Tsallis', [0, 1]),
             'Classifier': hp.choice('Classifier', [0, 1, 2])}

    trials = Trials()
    best_tuning = fmin(fn=objective_rf,
                       space=space,
                       algo=tpe.suggest,
                       max_evals=estimations,
                       trials=trials)

    index = range(len(df_x.columns.tolist()))


    classifier = param['Classifier'][best_tuning['Classifier']]

    btrain = df_x.iloc[:, index]
    btest = df_test.iloc[:, index]

    return btrain, btest

def objective_rf(space):
    fasta_label_train = 2
    n_cpu = 1

    """Automated Feature Engineering - Objective Function - Bayesian Optimization"""

    index = range(len(df_x.columns.tolist()))

    x = df_x.iloc[:, index]

    if int(space['Classifier']) == 0:
        if fasta_label_train > 2:
            model = AdaBoostClassifier(random_state=63)
        else:
            model = CatBoostClassifier(n_estimators=500,
                                       thread_count=n_cpu, nan_mode='Max',
                                       logging_level='Silent', random_state=63)
    elif int(space['Classifier']) == 1:
        model = RandomForestClassifier(n_estimators=500, n_jobs=n_cpu, random_state=63)
    else:
        model = lgb.LGBMClassifier(n_estimators=500, n_jobs=n_cpu, random_state=63)

    if fasta_label_train > 2:
        score = make_scorer(f1_score, average='weighted')
    else:
        score = make_scorer(balanced_accuracy_score)

    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    metric = cross_val_score(model,
                             x,
                             labels_y,
                             cv=kfold,
                             scoring=score,
                             n_jobs=n_cpu).mean()

    return {'loss': -metric, 'status': STATUS_OK}
