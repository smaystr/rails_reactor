from models.linear_regression_module import LinearRegression
from models.logistic_regression_module import LogisticRegression
from utils import split_to_X_and_y, get_scores
import torch


def process(args):
    task, target_name, fold, use_gpu = args
    train, test = fold
    X_train, y_train = split_to_X_and_y(train, target_name)
    X_test, y_test = split_to_X_and_y(test, target_name)
    mapTaskToModel = {
        'classification': LogisticRegression(use_gpu=use_gpu),
        'regression': LinearRegression(use_gpu=use_gpu),
    }
    model = mapTaskToModel[task]
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores = get_scores(task, y_test, torch.Tensor.cpu(y_pred))

    return model.coef, scores
