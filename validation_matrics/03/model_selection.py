import random
import numpy


def test_train_split(model, X, y, test_size=.1, random_state=42, metric=None):
    """
    Splitting the X ndarray into the X_train and X_test data
    :param model: object
    :type X: numpy.ndarray
    :type y: numpy.ndarray
    :type test_size: float
    :type random_state: int
    :param metric: method
    """
    random.seed(random_state)
    rows = random.sample(range(len(X)), int(test_size * len(X)))
    X_train, y_train = X[[row for row in range(len(X)) if not (row in rows)], :], y[[row for row in range(len(y)) if
                                                                                     not (row in rows)], :]
    X_test, y_test = X[rows, :], y[rows, :]
    if metric is None:
        raise AttributeError('Error! The metric hasn\'t been defined.')
    else:
        _, prediction, _ = model.predict(X_test)
        score = metric(y_test, prediction)
    return score


def K_fold(model, X, y, n_splits=10, metric=None):
    """
    K-fold split
    :param model: object
    :type X: numpy.ndarray
    :param y: numpy.ndarray
    :param n_splits: int
    :param metric: method
    """
    scores = []
    indices = numpy.arange(len(X))
    fold_indices = numpy.array_split(indices, n_splits)

    for fold_index in range(n_splits):
        train_fold_index = numpy.concatenate(numpy.delete(fold_indices, fold_index, axis=0), axis=-1)
        test_fold_index = fold_indices[fold_index]

        X_train, y_train = X[train_fold_index], y[train_fold_index]
        X_test, y_test = X[test_fold_index], y[test_fold_index]
        model.fit(X_train, y_train)

        if metric is None:
            raise AttributeError('Error! The metric hasn\'t been defined.')
        else:
            _, prediction, _ = model.predict(X_test)
            score = metric(y_test, prediction)
        scores.append(score)
    return numpy.array(scores).mean()


def leave_one_out(model, X, y, metric=None):
    """
    Leave one out
    :param model: object
    :type X: numpy.ndarray
    :type y: numpy.ndarray
    :type metric: method
    """
    return K_fold(model, X, y, n_splits=len(X), metric=metric)


def time_series(model, X, y, time_column, n_splits=10, metric=None):
    """
    Time series
    :type model: object
    :param X: numpy.ndarray
    :param y: numpy.ndarray
    :param time_column: str
    :param n_splits: int
    :param metric: method
    """
    scores = []
    indices = numpy.argsort(time_column)
    X, y = X[indices], y[indices]
    indices = numpy.arange(len(X))
    fold_indices = numpy.array_split(indices, n_splits + 1)

    for fold in range(n_splits):
        test_fold_index = fold_indices[fold + 1]
        train_fold_index = numpy.concatenate(fold_indices[:fold + 1])

        X_train, y_train, = X[train_fold_index], y[train_fold_index]
        X_test, y_test = X[test_fold_index], y[test_fold_index]
        model.fit(X_train, y_train)

        if metric is None:
            raise AttributeError('Error! The metric hasn\'t been defined.')
        else:
            _, prediction, _ = model.predict(X_test)
            score = metric(y_test, prediction)
        scores.append(score)
    return scores


def cross_validation_score(model, X, y, split_type,
                           test_size=.1,
                           n_splits=10,
                           time_column=None,
                           metric=None):
    score = 0
    if split_type == 0:
        score = test_train_split(model, X, y, test_size=test_size, metric=metric)
    elif split_type == 1:
        score = K_fold(model, X, y, metric=metric)
    elif split_type == 2:
        score = leave_one_out(model, X, y, metric=metric)
    elif split_type == 3:
        if time_column is None:
            raise Exception("Can not do time validation without time_column values")
        score = time_series(model, X, y, time_column=time_column, n_splits=n_splits, metric=metric)
    return score
