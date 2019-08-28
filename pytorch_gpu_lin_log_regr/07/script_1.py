import itertools
import logging
import time

from hw4.cross_validation import train_test_split
from hw4.metrics import *
from hw4.utilities import set_up_logging, read_file
from hw6.models import LogisticRegression, LinearRegression
from hw6.utilities import parse_args

if __name__ == '__main__':
    args = parse_args()

    random_state = 42
    set_up_logging(args.log, args.verbose)

    X, y, columns = read_file(args.dataset, args.target, args.na, args.categorical)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    X_train, X_test, y_train, y_test = (
        torch.from_numpy(X_train).float(),
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_train).float(),
        torch.from_numpy(y_test).float()
    )

    if args.task == 'logistic':
        model = LogisticRegression
    else:
        model = LinearRegression

    params = {
        "C": [0.01, 0.1, 0.5, 0.05],
        "num_iterations": [2000, 3000, 4000, 1000],
        "learning_rate": [0.01, 0.1, 0.5, 0.05],
        "batch_size": [32, 48, 64]
    }

    keys, values = zip(*params.items())
    params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('gpu')
    else:
        device = torch.device('cpu')

    test_data_scores = []
    fit_data_scores = []
    metrics = []

    for param in params:  # simple params tuning
        start_time = time.time()
        logging.info(f'CURR PARAMS: {param}')
        lin_model = model(
            C=param['C'],
            num_iterations=param['num_iterations'],
            learning_rate=param['learning_rate'],
            batch_size=param['batch_size'],
            device=device
        ).fit(X_train, y_train)

        fit_time = time.time() - start_time

        y_pred_test = lin_model.predict(X_test)
        y_pred_train = lin_model.predict_prob(X_train) if args.task == 'logistic' else lin_model.predict(X_train)

        logging.info(f'model score for test data is: {lin_model.score(X_test, y_test)}')
        logging.info(f'model score for train data is: {lin_model.score(X_train, y_train)}')
        logging.info(f'model fit time is {fit_time}')

        test_data_scores.append(lin_model.score(X_test, y_test))
        fit_data_scores.append(fit_time)

        if args.task == 'logistic':
            m = {'accuracy': lin_model.score(X_test, y_test),
                 'recall': recall(y_test, y_pred_test),
                 'precision': precision(y_test, y_pred_test),
                 'f1': f1(y_test, y_pred_test),
                 'log-loss': log_loss(y_test, lin_model.predict_prob(X_test))}
        else:
            m = {'mse': mse(y_test, y_pred_test),
                 'rmse': rmse(y_test, y_pred_test),
                 'mae': mae(y_test, y_pred_test),
                 'mape': mape(y_test, y_pred_test),
                 'mpe': mpe(y_test, y_pred_test),
                 'r2': r2(y_test, y_pred_test)}

        metrics_out = '\n'.join(f'               {k}: {v}' for k, v in m.items())
        metrics.append(metrics_out)
        logging.info(f'Model metrics: \n{metrics_out}')

    max_score = max(test_data_scores)
    logging.info(f'Max score: {max_score}, params: {params[test_data_scores.index(max_score)]}')
    logging.info(f'Model metrics: \n{metrics[test_data_scores.index(max_score)]}')
    min_fit_time = min(fit_data_scores)
    logging.info(f'Min fit time: {min_fit_time}, params: {params[fit_data_scores.index(min_fit_time)]}')
    logging.info(f'Model metrics: \n{metrics[fit_data_scores.index(min_fit_time)]}')
