import time
import argparse
import pathlib
import pandas as pd
import numpy as np
import csv

from linear_models import MyLogisticRegression, MyLinearRegression
from model_selection import RandomSerach, GridSearch, cross_val_score
from preprocessing import extract_target_and_time_columns, DataPreparator
import config

class Experiment:
    def __init__(self, csv_path, target_column_name, task_type,  param_search_type,
                 validation_type = 1, validation_split_size = 0.2, num_folds_validation=5,
                 verbose=True,
                 num_param_iterations = 10,
                 time_column_name = None,
                 output_namespace = "model"):

        self.csv_path = self.validate_path(csv_path)
        self.target_column_name = target_column_name
        self.task_type = task_type
        self.model_constructor = MyLinearRegression if task_type else MyLogisticRegression
        self.param_search_type= param_search_type
        self.param_search_constructor = GridSearch if param_search_type else RandomSerach
        self.num_param_iterations = num_param_iterations
        self.validation_type = validation_type
        self.validation_split_size = validation_split_size
        self.num_folds_validation = num_folds_validation
        self.time_column_name = time_column_name
        self.verbose = verbose
        self.output_namespace = output_namespace
        self.metrics = config.REGRESSION_METRICS if task_type else config.CLASSIFICATION_METRICS
        self.validate_params()

    @staticmethod
    def validate_path(csv_path):
        path = pathlib.Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Path {csv_path} doesn not exist")
        return path

    def get_column_names(self):
        with open(self.csv_path, 'r') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            return fieldnames

    def validate_params(self):

        column_names = self.get_column_names()

        if self.validation_type == 3 and \
            (self.time_column_name is None
             or self.time_column_name in column_names):

            raise AttributeError(f"Column for time_based validation {self.time_column_name}"
                                 f" should be in specified dataset")
        if self.target_column_name not in column_names:

            raise AttributeError(f"Target column {self.target_column_name} "
                                 f"should be in specified dataset")

    def get_feature_IR(self, trained_model,
                       num_features = 5):
        # First taking the absolute value of the weight coefs,
        # sorting indexes according to the importance,
        # and taking top N afterwards

        return np.argsort(
            np.abs(
                trained_model.weights.flatten()
            )
        )[::-1][:num_features]

    def select_param_grid(self):
        if not self.task_type and self.param_search_type:
            return config.CLASS_GRID
        if not self.task_type and not self.param_search_type:
            return config.CLASS_RAND
        if self.task_type and self.param_search_type:
            return config.REG_GRID
        return config.REG_RAND

    def generate_report(self, best_model , scores, time_elapsed):

        file_info = pathlib.Path(f"{self.output_namespace}.info")
        file_out = pathlib.Path(f"{self.output_namespace}.out")

        metric_dict = {metric.__name__: value for metric, value in zip(self.metrics, scores)}

        report_info = f"Metrics : {metric_dict}\nTraining time on {self.num_folds_validation} folds " \
                      f"{time_elapsed}s.\nWeight indecies associated with the most important features " \
                      f"{self.get_feature_IR(best_model)} "

        task_type = "regression" if self.task_type else "classification"
        report_out = f"Task type: {task_type}\nBest hyperparams {best_model.get_params()}\n" \
                     f"Model weights {best_model.weights} "

        file_info.write_text(report_info)
        file_out.write_text(report_out)

    def run_experiment(self):

        data = pd.read_csv(self.csv_path)
        data_np, target_value , time_col_value = extract_target_and_time_columns(data,
                                                                                 self.time_column_name,
                                                                                 self.target_column_name,
                                                                                 cv_type=self.validation_type)

        data_preparator = DataPreparator(config.CARDINALITY_THRESHOLD)
        data_preprocessed = data_preparator.fit_transform(data_np)
        param_grid = self.select_param_grid()

        param_search = self.param_search_constructor(self.model_constructor(), param_grid, self.num_folds_validation,
                                                    cv_type = self.validation_type,
                                                    num_iterations = self.num_param_iterations)

        best_model = param_search.fit(data_preprocessed, target_value,
                                      time_column_value = time_col_value)

        print(f"Best model is {best_model}\n")
        print("Starting to evaluate different metrics")

        time_start = time.time()

        scores = cross_val_score(best_model, data_preprocessed, target_value, 1,
                                 num_folds=self.num_folds_validation,
                                 metrics = self.metrics)

        time_elapsed = time.time() - time_start

        best_model.fit(data_preprocessed, target_value)

        self.generate_report(best_model, scores, time_elapsed)



def parse_args():

    p = argparse.ArgumentParser(description=
        """
        Argument parsed for an automated model training
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("csv_path",type=str,
                   help="Absolute path where csv file is stored")

    p.add_argument("target_column_name",type=str,
                   help="Name of the column which will be used for prediction")


    p.add_argument("task_type", type=int, choices=[0,1],
                   help="0 for classification , 1 for regression")

    p.add_argument("profile_path" ,type=str,
                   help="output path for profile report")

    p.add_argument("--validation_type",type=int, choices=[0,1,2,3], default=1,
                   help="0 for train_test_split\n ,"
                        "1 for k-fold validation\n "
                        "2 for LOO validation\n"
                        "3 for time_based_validation (time_column_value should be specified)"
                    )

    p.add_argument("--time_column_value",type=str, default= None,
                   help="Name of the column which will be used for time_based_validation")

    p.add_argument("--train_test_split_size",type=float, default=.2,
                   help="Name of the column which will be used for time_based_validation")

    p.add_argument("--num_folds_validation",type=int, default=5,
                   help="Name of the column which will be used for time_based_validation")


    p.add_argument("--param_search_type", type=int, choices=[0,1], default=0,
                   help="0 for RandomSearch , 1 for GridSearch ;"
                        "Parameters for tuning are prespecified in a config file "
                        "with reasonable values")

    p.add_argument("--num_rounds_for_rand_search", type=int, default=10,
                   help="Number of iterations to run RandomSearch"
                   )

    p.add_argument("-v", "--verbosity", type=int, choices=[0,1], default=1,
                   help="turn on/off verbosity")

    return p.parse_args()


def main():
    args = parse_args()
    model_setting = Experiment(args.csv_path,
                                  args.target_column_name,
                                  args.task_type,
                                  args.param_search_type,
                                  args.validation_type,
                                  args.train_test_split_size,
                                  args.num_folds_validation,
                                  args.verbosity,
                                  args.num_rounds_for_rand_search,
                                  args.time_column_value,
                                  args.profile_path
                                  )

    model_setting.run_experiment()

if __name__ == '__main__':
    main()

