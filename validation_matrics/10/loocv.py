from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression
from utils import split_to_X_and_y, get_fold, get_scores
from tqdm import tqdm
import multiprocessing
import numpy as np
import torch


def process_loocv(args):
    task, target_name, fold, use_gpu = args
    train, test = fold
    X_train, y_train = split_to_X_and_y(train, target_name)
    X_test, y_test = split_to_X_and_y(test, target_name)
    mapTaskToModel = {
        'classification': LogisticRegression(use_gpu=use_gpu),
        'regression': LinearRegression(use_gpu=use_gpu),
    }
    model = mapTaskToModel[task]
    clf = model.fit(X_train, y_train, epochs=10000)
    y_pred = int(clf.predict(X_test))

    return model.coef, y_pred


def loocv(task, df, target_name, use_gpu):
    N = df.shape[0]
    params = [(task, target_name, fold, use_gpu) for fold in get_fold(df, N)]
    with torch.multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        clfs = list(tqdm(pool.imap(process_loocv, params), total=N))

    mean_coefs = torch.mean(torch.stack([coef for coef, y_pred in clfs]))
    y_pred = [y_pred for coef, y_pred in clfs]
    X, y_true = split_to_X_and_y(df, target_name)
    scores = get_scores(task, np.array(y_true), np.array(y_pred))

    return mean_coefs, scores
