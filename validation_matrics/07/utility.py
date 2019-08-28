import argparse
import pathlib
import time
import json
from parameter_fitting import *
from cross_validation import *
from load_data import *
from metrics import *
from models.LinearRegression import LinearRegression
from models.LogisticRegression import LogisticRegression

def main(dataset_path,
        target,
        task,
        output_path,
        validation_split_type,
        validation_split_size,
        parameter_fitting_algo,
        parameters_to_fit):

    X, y = linear_data(dataset_path, target) if task == 'regression' else logistic_data(dataset_path, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LinearRegression if task == 'regression' else LogisticRegression

    start = time.time()
    lr = model().fit(X_train, y_train)
    fit_time = time.time() - start
    y_pred_test = lr.predict(X_test)
    y_pred_train = lr.predict(X_train) if task == 'regression' else lr.predict_proba(X_train)

    if task == 'regression':
        m = {'mse': mse(y_test, y_pred_test),
            'rmse': rmse(y_test, y_pred_test),
            'mae': mae(y_test, y_pred_test),
            'mape': mape(y_test, y_pred_test),
            'mpe': mpe(y_test, y_pred_test),
            'r2': r2(y_test, y_pred_test)}
    else:
        m = {'accuracy': accuracy(y_test, y_pred_test),
            'recall': recall(y_test, y_pred_test),
            'precision': precision(y_test, y_pred_test),
            'f1': f1(y_test, y_pred_test),
            'log-loss': log_loss(y_test, lr.predict_proba(X_test))}

    params = json.loads(parameters_to_fit.read_text())
    search =  RandomizedSearch(model, params, cv=validation_split_size) if parameter_fitting_algo == 'randomized' else GridSearch(model, params, cv=validation_split_size)
    search.fit(X_train, y_train)

    metrics_out = '\n'.join('  {!s}: {!r}'.format(p, v) for (p, v) in m.items())
    loss = mse(y_train, y_pred_train) if task == 'regression' else log_loss(y_train, y_pred_train)
    features = np.sort(np.delete(search.best_weights, 0)).ravel()[-3:]
    features_out = '\n'.join('  {!r}'.format(f) for f in features.tolist())
    output_info = (f'METRICS:\n{metrics_out}\n\n'
                    f'TRAINING PHASE:\n  time: {fit_time}\n  loss: {loss}\n\n'
                    f'TOP FEATURES:\n{features_out}')
    params_out = '\n'.join('  {!s}: {!r}'.format(p, v) for (p, v) in search.best_hyperparameters.items())
    weights_out = '\n'.join('  {!s}'.format(v[0]) for v in search.best_weights)
    output_model = (f'TYPE: {task}\n\n'
                    f'BEST HYPERPARAMETERS:\n{params_out}\n\n'
                    f'WEIGHTS:\n{weights_out}\n')
    
    output_path.joinpath('output_info.txt').write_text(output_info)
    output_path.joinpath('output_model.txt').write_text(output_model)

    print('Done. Output info and models info contains in files \'output_info.txt\' and \'output_model.txt\'')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--output_path', type=pathlib.Path, required=True)
    parser.add_argument('--validation_split_type', type=str, required=False, default='kfold')
    parser.add_argument('--validation_split_size', type=int, required=False, default=5)
    parser.add_argument('--parameter_fitting_algo', type=str, required=False, default='grid')
    parser.add_argument('--parameters_to_fit', type=pathlib.Path, required=True)

    args = parser.parse_args()

    main(args.dataset_path,
        args.target,
        args.task,
        args.output_path,
        args.validation_split_type,
        args.validation_split_size,
        args.parameter_fitting_algo,
        args.parameters_to_fit)
