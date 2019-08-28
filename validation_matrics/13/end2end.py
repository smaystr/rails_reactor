import random
import numpy as np

from preprocessing import parse_args, get_data
from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from magic import Validation

if __name__ == '__main__':
    random.seed(42)
    args = parse_args()
    data, column_names = get_data(args)
    target = args.target

    if target == 'charges':
        d = 2
    elif target == 'target':
        d = 1
    else:
        raise AttributeError(f"Please give me heart disease or Medical Cost Personal dataset")

    target_column = 0
    try:
        target_column = np.where(column_names == target)[0][0]
    except Exception:
        print('Probably you have two equal columns')

    if args.task == 'classification':
        model = LogisticRegression
    elif args.task == 'regression':
        model = LinearRegression
    else:
        raise AttributeError(f"Incorect name of task - {args.task}")

    validation = Validation(data, target_column, model, args.output, args.split,
                            args.validation_size, args.time_series, args.hyperparameter_fit, d)
    validation.fit()
