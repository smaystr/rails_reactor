import argparse
import json
import pathlib
import torch
import time

from utils.metrics import f_score, rmse
from script_1.models import LinearRegression, LogisticRegression
from utils.preprocessing import load_data


def main(task, train, test, config, device_type, output_path):
    start_time = time.time()
    if device_type == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    config.update({'device': device})

    X_train, y_train, X_test, y_test = load_data(train, test, task, device)

    if task == 'Classification':
        model = LogisticRegression(**config).fit(X_train, y_train)
    else:
        model = LinearRegression(**config).fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    torch.save(model.coef, output_path)

    if task == 'Classification':
        print(
            f""" 
    Time : {time.time() - start_time}
    Coefficients: {model.coef.reshape(-1, )}                                        
    F1-Score fot training set: {f_score(y_train, y_pred_train)} 
    F1-Score for test set: {f_score(y_test, y_pred_test)}                
    """)

    else:
        print(f"""
    Time : {time.time() - start_time}
    Coefficients: {model.coef.reshape(-1, )}
    RMSE fot training set: {rmse(y_train, y_pred_train)}
    RMSE for test set: {rmse(y_test, y_pred_test)}
    """)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='type of task', choices=['Regression', 'Classification'])
    parser.add_argument('train', type=str, help='path to the train dataset')
    parser.add_argument('test', type=str, help='path to the test dataset')
    parser.add_argument('-dt', '--device_type', type=str, default='cpu', choices=['cpu', 'gpu'],
                        help='type of processing unit')
    parser.add_argument('-op', '--output_path', type=pathlib.Path, default='./output.txt',
                        help='path for saving weights')
    parser.add_argument('-cp', '--config_path', type=pathlib.Path, default='./config.json',
                        help='path to the configuration file')
    args = parser.parse_args()

    config = json.loads(args.config_path.read_text())
    print(args.task, args.train, args.test, config, args.device_type, args.output_path)
    main(args.task, args.train, args.test, config, args.device_type, args.output_path)
