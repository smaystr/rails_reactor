import sys
import json

from pathlib import Path
from sklearn.externals import joblib
import torch
import pandas as pd

from HA8 import my_config, my_utils, preprocessing
from HA8.app.settings import MODELS_SUPPORTED
from HA8.dnn import PriceRegressorDNN

# This is needed for pickle to work properly (it requires that class definition must be
# importable and live in the same module as when the object was stored)
sys.modules["preprocessing"] = preprocessing


def load_model(model_name):

    if not model_name:
        raise NameError("Param 'model' should be provided")

    model_name = preprocess_string(model_name)

    if model_name not in MODELS_SUPPORTED:
        raise NameError(
            f"Invalid model name {model_name}, "
            f"the following models supported {MODELS_SUPPORTED}"
        )

    if model_name == "decision_tree":
        model_path = my_config.DT_PATH
    elif model_name == "lgbm":
        model_path = my_config.LGB_PATH
    else:
        model_path = my_config.DNN_PATH

    model_path = Path("..") / model_path

    if model_name in ["decision_tree", "lgbm"]:
        model = joblib.load(model_path)
    else:
        model = PriceRegressorDNN(
            my_config.DNN_DEFAULT_FEATURES_NUM,
            my_config.DNN_DEFAULT_HIDDEN_UNITS_DIM,
            torch.nn.ReLU(),
        )
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    return model


def preprocess_string(string_value):
    return string_value.strip().lower()


def check_features(features_dict):
    features_trained = joblib.load(Path("..") / my_config.FEATURES_PATH)

    if features_dict:
        features_dict = json.loads(features_dict)
        keys = list(features_dict.keys())
    else:
        keys = []
    if set(keys) != set(features_trained):
        raise AttributeError(
            f"Following features should be specified {features_trained} , got {keys}"
        )
    return features_dict


def predict(model, features_dict):

    features_df = pd.DataFrame.from_dict(features_dict)

    data_prepared = my_utils.prepare_data(features_df, is_train=False)

    if hasattr(model, "__call__"):
        data_prepared = torch.from_numpy(data_prepared).float()
        preds = model(data_prepared).tolist()
    else:
        preds = model.predict(data_prepared).tolist()

    return preds
