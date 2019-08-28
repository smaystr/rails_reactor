from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from pathlib import Path

import numpy as np
from HA8 import my_config


class BoolTranformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array(X).astype(np.float)


class MissingValuesImputer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        cat_features,
        num_features,
        bool_features,
        categorical_imputation_dict=None,
        num_features_imputation_dict=None,
        bool_imputation_dict=None,
        default_cat_value="Uknown",
        default_bool_value=False,
        default_num_value=0,
    ):

        self.cat_features = cat_features
        self.num_features = num_features
        self.bool_features = bool_features

        dict_initializer = (
            lambda given_dict: given_dict if type(given_dict) is dict else dict()
        )
        self.cat_feature_imputer = dict_initializer(categorical_imputation_dict)
        self.num_feature_imputer = dict_initializer(num_features_imputation_dict)
        self.bool_feature_imputer = dict_initializer(bool_imputation_dict)
        self.default_cat_value = default_cat_value
        self.default_num_value = default_num_value
        self.default_bool_value = default_bool_value

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for category in self.cat_features:
            X_transformed[category].fillna(
                self.cat_feature_imputer.get(category, self.default_cat_value),
                inplace=True,
            )
        for num_feature in self.num_features:
            X_transformed[num_feature].fillna(
                self.num_feature_imputer.get(num_feature, self.default_num_value),
                inplace=True,
            )
        for bool_feature in self.bool_features:
            X_transformed[bool_feature].fillna(
                self.bool_feature_imputer.get(bool_feature, self.default_bool_value),
                inplace=True,
            )
        return X_transformed


def preprocess_data(data, is_train):
    if not is_train:
        transformer_path = Path("..") / my_config.COLUMN_TRANSFORMER_PATH
        if transformer_path.exists():
            column_transformer = joblib.load(transformer_path)
        else:
            raise Exception("Preprocessors should be fitted first")
    else:
        column_transformer = ColumnTransformer(
            [
                (
                    "one_hot_encoder",
                    OneHotEncoder(
                        categories="auto", handle_unknown="ignore", sparse=False
                    ),
                    my_config.CAT_FEATURES,
                ),
                ("scaler", StandardScaler(), my_config.NUM_FEATURES),
                ("bool_encoder", BoolTranformer(), my_config.BOOL_FEATURES),
            ],
            remainder="passthrough",
            verbose=2,
        )

    imputer = MissingValuesImputer(
        my_config.CAT_FEATURES,
        my_config.NUM_FEATURES,
        my_config.BOOL_FEATURES,
        num_features_imputation_dict={"years_elapsed": 1000},
    )

    preprocessing_pipeline = Pipeline(
        [("features_imputer", imputer), ("column_transformer", column_transformer)]
    )

    if is_train:
        data_preprocessed = preprocessing_pipeline.fit_transform(data)
        # This is needed to restore population stats when serving model
        # For consistent encoding and scaling
        print(f"Serializing transformer {column_transformer}")
        joblib.dump(column_transformer, my_config.COLUMN_TRANSFORMER_PATH)
        return data_preprocessed
    else:
        return preprocessing_pipeline.transform(data)
