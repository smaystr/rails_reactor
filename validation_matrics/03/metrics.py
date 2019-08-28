import numpy


def RMSE(y_test, prediction):
    """
    Calculating the Root Mean Squared Error
    :type y_test: numpy.ndarray
    :type prediction: numpy.ndarray
    """
    return numpy.sqrt((numpy.power(y_test - prediction, 2)).mean())


def MSE(y_test, prediction):
    """
    Calculating the Mean Squared Error
    :type y_test: numpy.ndarray
    :type prediction: numpy.ndarray
    """
    return numpy.power(y_test - prediction, 2).mean()


def MAE(y_test, prediction):
    """
    Calculating the Mean Absolute Error
    :type y_test: numpy.ndarray
    :type prediction: numpy.ndarray
    """
    return (numpy.abs(y_test - prediction)).mean()


def MAPE(y_test, prediction):
    """
    Calculating the Mean Absolute Percentage Error
    :type y_test: numpy.ndarray
    :type prediction: numpy.ndarray
    """
    return (numpy.abs((y_test - prediction) / y_test)).mean() * 100


def MPE(y_test, prediction):
    """
    Calculating the Mean Percentage Error
    :type y_test: numpy.ndarray
    :type prediction: numpy.ndarray
    """
    return ((y_test - prediction) / y_test).mean() * 100


def precision(y_test, prediction):
    """
    Calculating the precision
    :type y_test: numpy.ndarray
    :type prediction: numpy.ndarray
    """
    tp = numpy.sum(numpy.logical_and(y_test == 1, prediction == 1))
    fp = numpy.sum(numpy.logical_and(y_test == 1, prediction == 0))
    return tp / (tp + fp)


def recall(y_test, prediction):
    """
    Calculating the precision
    :type y_test: numpy.ndarray
    :type prediction: numpy.ndarray
    """
    tp = numpy.sum(numpy.logical_and(y_test == 1, prediction == 1))
    fn = numpy.sum(numpy.logical_and(y_test == 0, prediction == 1))
    return tp / (tp + fn)


def F1_score(y_test, prediction, beta=1.):
    """
    Calculating the F1-score
    :type y_test: numpy.ndarray
    :type prediction: numpy.ndarray
    :type beta: float
    """
    rec = recall(y_test, prediction)
    prec = precision(y_test, prediction)
    return (1 + beta ** 2) * ((prec * rec) / (beta ** 2 * prec + rec))


def accuracy(y_test, prediction):
    """
    Calculating the accuracy rate in percents
    :type y_test: numpy.ndarray
    :type prediction: numpy.ndarray
    """
    return numpy.sum(y_test == prediction) / len(y_test)
