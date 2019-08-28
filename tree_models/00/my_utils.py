import datetime
from string import punctuation

import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import make_scorer, mean_squared_error

import numpy as np
import torch
import pathlib

from HA8 import my_config
from HA8.preprocessing import preprocess_data


def load_data():
    engine = create_engine(my_config.DB_PATH)
    data = pd.read_sql_query(f"select * from {my_config.TABLE_NAME}", con=engine)
    return data


def count_upper(string):
    return sum(map(lambda x: x.isupper(), string)) if string else 0


def count_punctuations(string):
    return sum(map(lambda x: x in punctuation, string)) if string else 0


def preprocess_columns(data):
    data_without_cols = data.drop(my_config.COLS_TO_REMOVE, axis=1)
    data_without_cols.set_index("id", inplace=True)
    return data_without_cols


def prepare_data(data, is_train=True):

    data_with_features = generate_features(data)

    # removing extreme targets here
    if is_train:
        inliers_ind = get_inliers_ind(data_with_features, my_config.TARGET_COLUMN)
        not_nan_ind = get_nan_ind(data_with_features, my_config.TARGET_COLUMN)
        inliers_ind = list(not_nan_ind & inliers_ind)
        data_with_features = data_with_features.loc[inliers_ind]
        data_with_features, target = (
            data_with_features.drop([my_config.TARGET_COLUMN], axis=1),
            data_with_features[[my_config.TARGET_COLUMN]].values,
        )

    data_preprocessed = preprocess_data(data_with_features, is_train=is_train)
    if is_train:
        return data_preprocessed, target

    return data_preprocessed


def generate_features(data):

    data["number_of_images_attached"] = data["image_urls"].str.len()
    data["len_of_description"] = data["description"].str.len()
    data["num_of_uppercase_letters_in_description"] = data["description"].apply(
        count_upper
    )
    data["num_of_punctuations_in_description"] = data["description"].apply(count_upper)

    data.drop(["image_urls", "description"], axis=1, inplace=True)

    data["construction_year"] = data.construction_period.str.extract("([\d]{4})")

    data["years_elapsed"] = datetime.datetime.today().year - data[
        "construction_year"
    ].astype(np.float)

    data.drop(["construction_year", "construction_period"], axis=1, inplace=True)

    data["is_bargain"] = data.tags.apply(lambda row: "Торг" in row)
    data["is_used"] = data.tags.apply(lambda row: "Вторичное жилье" in row)
    data["is_not_used"] = data.tags.apply(lambda row: "Первичное жилье" in row)
    data["in_installments"] = data.tags.apply(lambda row: "Рассрочка/Кредит" in row)

    data.drop("tags", axis=1, inplace=True)
    return data


def get_inliers_ind(data, feature_name, is_positive=True):

    """
    Generate inliers indecies using IQR

    :param data: pd.Dataframe
    :param feature_name: feature
    :param is_positive:
    :return:
    """

    q1 = data[feature_name].quantile(q=0.25)
    q3 = data[feature_name].quantile(q=0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    if is_positive:
        lower_bound = max(0, lower_bound)
    mask = (data[feature_name] > upper_bound) | (data[feature_name] < lower_bound)
    indecies = mask[mask == False].index
    return set(indecies)


def get_nan_ind(data, feature_name):
    return set(data[data[feature_name].isna() == False].index)


def torch_rmse(Y_pred, Y_true):
    return torch.sqrt(torch.mean((torch.pow(Y_pred - Y_true, 2))))


def torch_r2_score(Y_pred, Y_true):
    mse = torch.mean((torch.pow(Y_pred - Y_true, 2)))
    return 1 - mse / torch.var(Y_true)


rmse_scorer = make_scorer(
    lambda y, preds: np.sqrt(mean_squared_error(y, preds)), greater_is_better=False
)


def write_stats_to_file(model, stats_dict):

    markdown_path = pathlib.Path(my_config.REPORT_PATH)

    with markdown_path.open("a") as file:
        file.write(f"\n\n## Report for model {model.__class__.__name__}")

    for key, val in stats_dict.items():
        report_text = f"\n\n**{key}** {val:.5f}"
        with markdown_path.open("a") as file:
            file.write(report_text)
