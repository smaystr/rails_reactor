import numpy as np

def precision(Y_true, Y_pred):
    Y_true, Y_pred = np.array(Y_true).astype(np.bool),\
                     np.array(Y_pred).astype(np.bool)
    tp = np.sum(np.bitwise_and(Y_true, Y_pred))
    fp = np.sum(np.bitwise_and(Y_true==0, Y_pred))
    return tp/(tp+fp)


def recall(Y_true, Y_pred):
    Y_true, Y_pred = np.array(Y_true).astype(np.bool),\
                     np.array(Y_pred).astype(np.bool)

    tp = np.sum(np.bitwise_and(Y_true, Y_pred))
    fn = np.sum(np.bitwise_and(Y_true, Y_pred==0))
    return tp/(tp+fn)


def f1_score(Y_true, Y_pred):
    precision_val = precision(Y_true, Y_pred)
    recall_val = recall(Y_true, Y_pred)
    return 2*precision_val*recall_val/(precision_val+recall_val)

def mape(Y_true, Y_pred):
    Y_true, Y_pred = np.array(Y_true), np.array(Y_pred)
    return np.mean(100*(np.abs((Y_true - Y_pred)/Y_true)))

def rmse(y_pred, y_true):
    return np.sqrt(np.mean(np.square(y_pred-y_true)))


