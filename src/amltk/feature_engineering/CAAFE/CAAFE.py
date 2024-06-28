# https://github.com/automl/CAAFE

import pandas as pd
import torch
from functools import partial

from caafe import CAAFEClassifier
from caafe.caafe import generate_features
from sklearn.metrics import accuracy_score

from remoteinference.models.models import LlamaCPPLLM
from remoteinference.util.config import ServerConfig

from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier


def get_xxx_features(train_x, train_y, test_x, test_y) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    # clf_no_feat_eng = RandomForestClassifier()
    # clf_no_feat_eng = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)
    # clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)

    # clf_no_feat_eng.fit(train_x, train_y)
    # pred = clf_no_feat_eng.predict(test_x)
    # acc = accuracy_score(pred, test_y)
    # print(f'Accuracy before CAAFE {acc}')

    # caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng, llm_model="gpt-4", iterations=2)

    df_train = pd.concat([train_x, train_y], axis=1)
    df_test = pd.concat([test_x, test_y], axis=1)
    ds = pd.concat([df_train, df_test], axis=0)
    ds = ds.values.tolist()
    # target_column_name = train_y.name
    # dataset_description = None
    # caafe_clf.fit_pandas(df_train, target_column_name=target_column_name, dataset_description=dataset_description)

    cfg = ServerConfig(server_address="localhost", server_port=8080)
    model = LlamaCPPLLM(cfg)
    full_code, prompt, messages = generate_features(ds, df_train, model=model, display_method="print")

    print(full_code)
    print(prompt)
    print(messages)

    return train_x, test_x


