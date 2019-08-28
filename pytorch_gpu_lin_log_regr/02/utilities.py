import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_dataset(
        train_csv,
        test_csv,
        target,
        categorial_features=None,
        boolean_features=None
) -> tuple:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    headers = train_df.columns
    if boolean_features is None:
        boolean_features = []
    if categorial_features is None:
        categorial_features = []
    numerical_features = [feature for feature in headers if (feature not in categorial_features and feature not in boolean_features)]
    numerical_features.remove(target)

    train_df = pd.get_dummies(train_df, columns=categorial_features)
    test_df = pd.get_dummies(test_df, columns=categorial_features)

    df = pd.concat([train_df, test_df], sort=True)

    df = scale_features(
        df=df,
        num_cols=numerical_features
    )

    X, y = extract_X_y(
        df=df,
        target=target
    )
    return X, y


def extract_X_y(
        df,
        target
):
    """
    Get numpy arrays from pandas data frame
    :type df: pd.DataFrame
    :type target: str
    """
    return np.array(df.loc[:, df.columns != target]), np.array(df.loc[:, df.columns == target])


def scale_features(
        df,
        num_cols
) -> pd.DataFrame:
    """
    Scale the numerical features and return the pandas data frame with that modifications
    :type df: pd.DataFrame
    :type num_cols: list
    """
    scaled_features = df[num_cols]
    scaled_features = StandardScaler() \
        .fit_transform(scaled_features)
    df[num_cols] = scaled_features
    return df
