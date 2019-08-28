import numpy as np
import argparse
import logistic, linear
import validation
import model_selection
from pathlib import Path
import preprocessing
import metrics

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Dataset path', type=str)
    parser.add_argument('--target_variable', help='Name of the target variable', type=str)
    parser.add_argument('--task', help='Task: regression/classification', type=str, choices=('regression', 'classification'))
    parser.add_argument('--validation_type', help='Validation type: train_test_split/KFold/LeaveOneOut', type=str, 
                        choices=('train_test_split', 'KFold', 'LeaveOneOut'))
    parser.add_argument('--test_size', help='Size of the test set', type=float)
    parser.add_argument('--hyperparameters_optimization', help='Algorithm for search for the parameters of the model', 
                        type=str, default='GridSearchCV', choices=('GridSearchCV', 'RandomSearchCV'))
    return parser.parse_args()

def read_data(Path):
    try:
        text = np.genfromtxt(Path, delimiter=',', dtype=str)
        columns = text[0,:]
        numbers = np.genfromtxt(Path, delimiter=',', dtype=float)[1:,:]
    except:
        raise Exception(f"{Path} no such file.")
    string_indexes = [key for key, value in enumerate(numbers.T) if np.isnan(value).all()]
    label_encoder = preprocessing.LabelEncoder()
    numbers[:, string_indexes] = label_encoder.fit_transform(text[1:, string_indexes], text[0,string_indexes])
    return numbers, columns

def get_metrics(argument):
    if argument.task == 'classification':
        metrics = [metrics.accuracy, metrics.precision, metrics.recall]
        metric_names = ["accuracy", "precision", "recall"]
    else:
        metrics = [metrics.mean_squared_error, metrics.mean_absolute_error, metrics.root_mean_squared_error]
        metric_names = ["MSE", "MAE", "RMSE"]
    return metrics, metric_names

def get_model(arguments):
    if arguments.task == 'classification':
        model = logistic.LogisticRegression()
    else:
        model = linear.LinearRegression()
    if arguments.validation_type == 'KFold':
        validation = validation.KFold
    elif arguments.validation_type == 'LeaveOneOut':
        validation = validation.LeaveOneOut
    else:
        validation = validation.train_test_split
    if arguments.hyperparameters_optimization == 'GridSearchCV':
        optimization = model_selection.GridSearchCV
    else:
        optimization = model_selection.RandomSearchCV
    return model, validation, optimization

def select_nans(data):
    nan_rows = [key for key, value in enumerate(data) if np.isnan(value).any()]
    return nan_rows

def split_data_target(data, columns, target_column):
    try:
        target = np.where(columns == target_column)[0][0]
    except IndexError:
        raise Exception(f"Target column {target_column} not found.")
    index = list(set(range(len(columns))) - set([target]))
    return data[:, index], data[:, target].reshape(len(data), 1)

def run():
    args = parse_arguments()
    model, validation, optimization = get_model(args)
    data, columns = read_data(Path(args.path))
    metrics, metric_names = get_metrics(args.task)
    
    if np.isnan(data).any():
        data = np.delete(select_nans(data), axis=0)
    x_data, y_data = split_data_target(data, columns, args.target_variable)
    scaler = preprocessing.StandardScaler()
    x_data = scaler.fit_transform(x_data)
    
    tune_hyperparameters = model_selection.GridSearchCV(model, metrics[0]) if args.hyperparameters_optimization =='GridSearchCV' else model_selection.RandomSearchCV(model, metrics[0])
    result = tune_hyperparameters.evaluate_model(x_data, y_data)
    result = result[max(result)]
    
    if args.validation_type == 'train_test_split':
        model = set_up(model, result)
        x_train, y_train, x_test, y_test = validation.train_test_split(x_data, y_data, test_size=args.test_size)
        model.fit(x_train, y_train)
        history = np.zeros((len(metrics), 2))
        for key, score in enumerate(metrics):
            history[key] += (score(y_train, model.predict(x_train)), score(y_test, model.predict(x_test)))
    else:
        if args.validation_type == 'KFold':
            validation = validation.KForld()
        else:
            validation = validation.LeaveOneOut()
    history = np.zeros((len(metrics),2))
    for fold, (train_index, validation_index) in enumerate(validation.split(x_data)):
        model = set_up(model, result)
        
        x_train, x_test = x_data[train_index], x_data[validation_index]
        y_train, y_test = y_data[train_index], y_data[validation_index]
        
        model.fit(x_train, y_train)
        
        for key, score in enumerate(metrics):
            history[key] += (score(y_train, model.predict(x_train)), score(y_test, model.predict(x_test)))
    history /= fold + 1

    with open('model.info', 'w+') as f:
        f.write('Metrics\n')
        for key, name in enumerate(names):
            file.write(f'{name} train: {history[key][0]}; test: {history[key][1]}\n')
        f.write('Feature importance')
        start = 0
        if model.fit_intercept:
            f.write(f'Intercept: {model.weights[0]}\n')
        start = 1
        for key, name in enumerate(columns):
            f.write(f'{name}: {model.weights[key+start-1][0]}\n')
    with open('model.model', 'w+') as f:
        f.write(f"Type: {args.task}\n")
        f.write(f'Best parameters: {res["params"]}\n')
        f.write(f"Weights:\n")
        f.write(str(model.weights))

if __name__ == '__main__':
    run()