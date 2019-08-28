import numpy as np

np.random.seed(42)


def train_test_split(dataset, size, target):
    n = dataset.shape[0]
    random_col = np.random.choice(n, int(n * size), replace=False)

    dataset_test = dataset[random_col, :]
    dataset_train = np.delete(dataset.copy(), random_col, axis=0)

    yield (
        np.delete(dataset_train, target, axis=1), dataset_train[:, target],
        np.delete(dataset_test, target, axis=1), dataset_test[:, target],
    )


def k_fold(X, k, target):
    indices = np.arange(X.shape[0])
    splits = np.array_split(indices, k)

    for fold in range(k):
        test = X[splits[fold]]
        train = np.concatenate(
            [X[x] for i, x in enumerate(splits) if i != fold]
        )

        yield (
            np.delete(train, target, axis=1), train[:, target],
            np.delete(test, target, axis=1), test[:, target],
        )


def leave_one_out(X, _, target):
    return k_fold(X, X.shape[0], target)


def timeseries_cv(X, k, trgt):
    target = trgt[0]
    time_col_num = trgt[1]

    indices = np.argsort(X[time_col_num])
    splits = np.array_split(indices, k+1)

    for fold in range(k):
        test = X[splits[fold + 1]]
        train = X[splits[fold]]

        yield (
            np.delete(train, target, axis=1), train[:, target],
            np.delete(test, target, axis=1), test[:, target],
        )


def grid_search(model, X, y, X_test, y_test, scoring):
    lr = (0.1, 0.01, 0.001, 0.0001)
    max_iter = (10000, 100000, 1000000)
    res = dict()

    for rate in lr:
        for iter in max_iter:
            result = model(rate, iter).fit(X, y).score(X_test, y_test, scoring)
            res[result] = (rate, iter)
            print(f'WITH lr: {rate} and iter: {iter} score is {result}')

    return res


def rd_search(model, X, y, X_test, y_test, scoring):
    lr = np.random.uniform(0.0001, 0.1, 4)
    max_iter = np.random.randint(10000, 1000000, 3)
    res = dict()

    for rate in lr:
        for iter in max_iter:
            result = model(rate, iter).fit(X, y).score(X_test, y_test, scoring)
            res[result] = (rate, iter)
            print(f'WITH lr: {rate} and iter: {iter} score is {result}')

    return res


def preprocessing(X):
    for i in range(X.shape[1]):
        try:
            X[:, i] = np.float32(X[:, i])

        except Exception:
            uniq = np.unique(X[:, i])
            classes = dict()

            cl_num = 1
            for cl in uniq:
                classes[cl] = cl_num
                cl_num += 1

            for j in range(X.shape[0]):
                X[j, i] = classes[X[j, i]]

    X = X.astype(np.float32)

    np.seterr(divide='ignore', invalid='ignore')
    min_ = X.min(axis=0)
    max_ = X.max(axis=0)
    return np.nan_to_num((X - min_) / (max_ - min_))
