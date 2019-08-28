import numpy as np
import requests as rqst
import argparse
import models as m
import for_validation as v
import metrics
import time

metr = {
        'accuracy': metrics.accuracy,
        'precision': metrics.precision,
        'recall': metrics.recall,
        'mse': metrics.mse,
        'rmse': metrics.rmse,
        'mae': metrics.mae,
        'f1': metrics.f1
    }


def main(
    ds: str, target: int, task: str, validation: str, split_size: float,
    timeseries: int, hp_search: str, output: str, n_fold: int, metric: str,
    regulization: str
):
    dataset = np.genfromtxt(
        rqst.get(ds).content.decode('utf-8') if ds.startswith('http') else ds,
        delimiter=',', dtype=str
    )

    model = m.LogRegression if task == 'classification' else m.LinearRegressor

    if validation == 'train-test split':
        val = v.train_test_split
    elif validation == 'k-fold':
        val = v.k_fold
        split_size = n_fold
    elif validation == 'leave-one-out':
        val = v.leave_one_out
    else:
        target = (target, timeseries)

    if not metric:
        metric = 'accuracy' if task == 'classification' else 'rmse'

    if output:
        info_out = dict()
        model_out = dict()

        model_out['type'] = task

        epoch = 0

    for X_train, y_train, X_test, y_test in val(dataset[1:, :], split_size, target):
        X_train = v.preprocessing(X_train)
        X_test = v.preprocessing(X_test)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        print('Model with default parametrs:\nlr = 0.01, max_iter = 100000')
        mdl = model(0.01, 100000, regulization)

        start_time = time.time()
        mdl.fit(X_train, y_train)
        total_time = time.time() - start_time

        testing_score = mdl.score(X_test, y_test, metr[metric])
        training_score = mdl.score(X_train, y_train, metr[metric])
        weight = mdl.get_theta()
        loss = mdl.get_loss()

        print(f"\n{metric} test:  {testing_score}")
        print(f"{metric} train: {training_score}")
        print(f"\nFinal weight:\n {weight}")

        if output:
            info_out[f'features_importance_{epoch}'] = np.argsort(weight).tolist()
            info_out[f'train_time_{epoch}'] = total_time
            info_out[f'loss_{epoch}'] = loss
            info_out[f'test_score_{epoch}'] = testing_score
            info_out[f'train_score_{epoch}'] = training_score
            info_out['metric'] = metric

            model_out[f'weights_{epoch}'] = weight

            epoch += 1

        if hp_search:
            print('\nParametrs from tuning:')
            tune_hp = v.grid_search if hp_search == 'grid_search' else v.rd_search
            result = tune_hp(model, X_train, y_train, X_test, y_test, metr[metric])

            if output:
                result = result[max(result)] if len(result) > 1 else result
                model_out[f'{hp_search}_best_hp'] = result

    if output:
        from pathlib import Path
        import pprint

        pp = pprint.PrettyPrinter()

        try:
            with (
                (Path(output) / 'output.info').open('w', encoding='utf-8')
            ) as info_file:
                info_file.write(pp.pformat(info_out))

            with (
                (Path(output) / 'output.model').open('w', encoding='utf-8')
            ) as model_file:
                model_file.write(pp.pformat(model_out))

        except Exception as e:
            print(f'\nSmth went wrong...\n{e}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="end2end model training lifecycle utility"
        )
    parser.add_argument(
        '--dataset', help='path to local .csv file or url to file', type=str,
        default=None, metavar='ds', required=True
        )
    parser.add_argument(
        '--target', help='target variable column index', type=int,
        default=None, metavar='t', required=True
        )
    parser.add_argument(
        '--task', help='classification / regression', type=str,
        default=None, required=True
        )
    parser.add_argument(
        '--output', help='path for output model: output.info and output.model',
        type=str, default=None, required=True
        )
    parser.add_argument(
        '--validation', help='validation split type \
(train-test split / k-fold / leave-one-out)', type=str,
        default='train-test split'
        )
    parser.add_argument(
        '--split_size', help='parametr for validation split size', type=float,
        default='0.2'
        )
    parser.add_argument(
        '--n_fold', help='parametr for k-fold validation', type=int,
        default='5'
        )
    parser.add_argument(
        '--timeseries', help='specify timeseries column to perform \
timeseries validation', type=int, default=None
        )
    parser.add_argument(
        '--hp_search', help='parameter for hyperparameter fitting algo: \
grid search / random search', type=str, default=None
        )
    parser.add_argument(
        '--metric', help='Specify metrics. Available:\n accuracy, precision, \
recall, f1, mse, rmse, mae', type=str, default=None
        )
    parser.add_argument(
        '--regulization', help='L1, L2 or L1_L2', type=str, default='L1'
        )

    args = parser.parse_args()

    main(
        args.dataset, args.target, args.task, args.validation,
        args.split_size, args.timeseries, args.hp_search,
        args.output, args.n_fold, args.metric, args.regulization
        )
