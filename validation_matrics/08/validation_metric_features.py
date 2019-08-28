import time

from hw4.cross_validation import train_test_split, KFold, LeaveOneOut, feature_importance
from hw3.lin_reg import LinearRegression
from hw3.log_reg import LogisticRegression
from hw4.metrics import *
from hw4.params_fitting import GridSearch, RandSearch
from hw4.utilities import *

if __name__ == '__main__':
    args = parse_args()
    random_state = 42
    set_up_logging(args.log, args.verbose)

    X, y, columns = read_file(args.dataset, args.target, args.na, args.categorical)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    if args.task == 'logistic':
        model = LogisticRegression
    else:
        model = LinearRegression

    start_time = time.time()
    lin_model = model(C=0.1).fit(X_train, y_train)
    fit_time = time.time() - start_time

    y_pred_test = lin_model.predict(X_test)
    y_pred_train = lin_model.predict_prob(X_train) if args.task == 'logistic' else lin_model.predict(X_train)

    logging.info(f'model score for test data is: {lin_model.score(X_test, y_test)}')
    logging.info(f'model score for train data is: {lin_model.score(X_train, y_train)}')
    logging.info(f'model fit time is {fit_time}')

    params = PARAMS

    logging.info(f'Tuning params, algo: {args.algo}, split type: {args.split_type}, split_size: '
                 f'{args.split_size}, params: {params}')

    search_type = GridSearch if args.algo == 'grid' else RandSearch
    split_type = KFold if args.split_type == 'k-fold' else LeaveOneOut

    search = search_type(model,
                         params,
                         args.split_size)
    search.fit(X, y, split_class=split_type)

    params_out = '\n'.join(f'                     {p}: {v}' for p, v in search.best_params.items())
    weights_out = '\n'.join(f'        {v[0]}' for v in search.best_weights)

    features_out = '\n'.join(f'             {y}: {x}' for x, y in feature_importance(search.best_weights, columns)[:3])

    loss = log_loss(y_train, y_pred_train) if args.task == 'logistic' else mse(y_train, y_pred_train)

    if args.task == 'logistic':
        m = {'accuracy': accuracy(y_test, y_pred_test),
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

    metrics_out = '\n'.join(f'        {k}: {v}' for k, v in m.items())

    output_info = (f'METRICS:\n{metrics_out}\n\n'
                   f'TRAINING PHASE:\n               time: {fit_time}\n               loss: {loss}\n\n'
                   f'TOP FEATURES:\n{features_out}')

    output_model = (f'TYPE:\n     {args.task}\n\n'
                    f'BEST HYPERPARAMETERS:\n{params_out}\n\n'
                    f'WEIGHTS:\n{weights_out}\n')

    logging.info(f'Saving model files into {args.path}')

    output_path = pathlib.Path(args.path)

    with open(output_path / 'output.info', 'w') as output:
        output.write(output_info)

    with open(output_path / 'output.model', 'w') as output:
        output.write(output_model)
