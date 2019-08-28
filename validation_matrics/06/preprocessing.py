import numpy as np
import pandas as pd

class DataPreparator:

    def __init__(self, percent_cat_unique=.05):

        self.percent_cat_unique = percent_cat_unique
        self.mean = None
        self.std = None

    def is_category(self, column):

        return len(np.unique(column)) <= 5

    def is_numerical_type(self, column):

        try:
            column.astype(np.float64)
        except Exception as e:
            return False
        return True

    def get_column_uniqueness(self, column):

        return len(np.unique(column))/len(column)

    def one_hot_encode(self, column):

        unique_vals, category_ind = np.unique(column, return_inverse=True)
        matrix = np.zeros((len(column), len(unique_vals)))
        matrix[np.arange(len(column)), category_ind] = 1
        return matrix

    def fit_transform(self, X):

        num_columns_mask = np.array([self.is_numerical_type(column) for column in X.T])
        non_num_columns_mask = ~num_columns_mask

        X[..., non_num_columns_mask] = X[..., non_num_columns_mask].astype(np.str)
        categorical_columns = X[..., non_num_columns_mask].T

        categorical_ratios = np.array([self.get_column_uniqueness(column) for column in categorical_columns])

        mask = categorical_ratios <= self.percent_cat_unique

        final_cat = categorical_columns[mask]

        numerical_columns = np.nan_to_num(X[..., num_columns_mask].T.astype(np.float))
        num_columns_cat = np.array([self.is_category(column) for column in numerical_columns])
        final_num_columns = numerical_columns[~num_columns_cat]
        numerical_categorical = numerical_columns[num_columns_cat]

        all_categorical = np.concatenate([final_cat, numerical_categorical], axis = 0)

        all_categorical = np.concatenate([self.one_hot_encode(column) for column in all_categorical], axis=1)

        final_num_columns = final_num_columns.T

        self.mean = final_num_columns.mean(axis=0)
        self.std = final_num_columns.std(axis=0)

        final_num_columns = (final_num_columns-self.mean)/self.std
        return np.concatenate([final_num_columns, all_categorical], axis=1)


def remove_col(data, col_ind):
        return np.delete(data.T,col_ind,axis=0).T



def extract_target_and_time_columns(data,
                                    time_column_name,
                                    target_column_name,
                                    cv_type):

        data_columns = data.columns
        data_np = data.values

        target_ind = np.argwhere(data_columns == target_column_name).flatten()
        target_val = data_np[..., target_ind]

        data_np = remove_col(data_np, target_ind)

        time_column_value = None
        if cv_type == 3:
            time_column_ind = np.argwhere(data_columns == time_column_name).flatten()
            time_column_value = pd.to_datetime(data_np[...,time_column_ind]).values
            data_np = remove_col(data_np, time_column_ind)

        return data_np, target_val.astype(np.float64), time_column_value
