import numpy as np
from collections import Counter


def detect_outliers(dataset, n, features):
    """
    Detecting outliers function.
    :param dataset:
    :param n: outliers threshold
    :param features:
    :return: list of the indices corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(dataset[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(dataset[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = dataset[(dataset[col] < Q1 - outlier_step) | (dataset[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than n outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = [k for k, v in outlier_indices.items() if v > n]

    return multiple_outliers


def normalize_data(features):
    """
    Normalize features.
    Normalizes input features X. Returns a normalized version of X where
    the mean value of each feature is 0 and deviation is close to 1.
    :param features: set of features.
    :return: normalized set of features.
    """

    features_normalized = np.copy(features).astype(float)
    features_mean = np.mean(features, 0)
    features_deviation = np.std(features, 0)

    if features.shape[0] > 1:
        features_normalized -= features_mean

    # Normalize each feature values so that all features are close to [-1:1].
    # Also prevent division by zero error.
    if features_deviation[features_deviation == 0]:
        features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized
