from abc import ABC, abstractmethod
from my_utils import *

class ParamSearch(ABC):
    def __init__(self, estimator, param_grid, num_folds=5, cv_type=1,
                 metric=None, num_iterations=None, verbose=True):

        #multiple metrics are not allowed
        assert((type(metric) is not list) or
               (type(metric) is list and len(metric) == 1))

        self.estimator_constructor = estimator.__class__
        self.estimator_params = estimator.__dict__

        self.validate_grid(estimator, param_grid)

        self.param_grid = self.create_grid(param_grid, num_iterations)
        self.num_folds = num_folds
        self.cv_type = cv_type
        self.metric = metric
        self.verbose  = verbose
        self.best_score = None
        self.best_params = None


    @abstractmethod
    def create_grid(self, param_grid, num_iterations):
        pass

    def validate_grid(self,estimator, param_grid):
        for key in param_grid:
            if key not in self.estimator_params:
                raise AttributeError(f"{key} is not present in {estimator} set of attributes")

    def fit(self, X, Y, time_column_value=None):
        scores=[]
        for params in self.param_grid:
            estimator = self.estimator_constructor(**params)
            score = cross_val_score(estimator, X, Y , split_type = self.cv_type,
                                    time_column_value = time_column_value,
                                    metrics = self.metric,
                                    verbose = self.verbose)
            scores.append(score)

        scores = np.array(scores)
        best_score_arg = np.argmax(scores)
        best_score_val = scores[best_score_arg]
        best_params = self.param_grid[best_score_arg]
        self.best_score = best_score_val
        self.best_params = best_params
        return self.estimator_constructor(**best_params)


class GridSearch(ParamSearch):
    g
    def create_grid(self, param_grid, num_iterations):
        return list(product_dict(param_grid))


class RandomSerach(ParamSearch):

    def create_grid(self, param_grid, num_iterations):
        param_values = []
        for key,value in param_grid.items():
            if type(value) is list or type(value) is np.ndarray:
                values = np.random.choice(np.array(value), size=num_iterations)
            elif hasattr(value,'rvs'):
                values = value.rvs(size = num_iterations)
            else:
                raise AttributeError(f"Unrecognized object for param generation {value}.\n\
                Should be list or some random variable with rvs method implemented")
            param_values.append(values)
        param_values = np.array(param_values)

        keys = param_grid.keys()
        final_params = [{key:value for (key,value) in zip(keys,value)} for value in param_values.T]
        return final_params



def cross_val_score(model, X, Y, split_type,
                    train_test_split_size=.2,
                    num_folds=5,
                    time_column_value=None,
                    verbose=True,
                    metrics=None):

    if split_type == 0:
        score = train_test_validation(model, X, Y, train_test_split_size= train_test_split_size,
                                      verbose=verbose,metrics=metrics)
    elif split_type == 1:
        score = k_fold_validation(model, X, Y, num_folds=num_folds,
                                  verbose=verbose, metrics=metrics)

    # LOO is just a special case of K-fold, so it could be reused.
    
    elif split_type == 2:
        score = k_fold_validation(model, X, Y, num_folds=len(Y),
                                  verbose=verbose, metrics=metrics)
    elif split_type == 3:
        if time_column_value is None:
            raise Exception("Can not do time validation without time_column values")

        score = time_based_validation(model, X, Y, time_column_value=time_column_value,
                                      num_folds=num_folds,
                                      verbose=verbose, metrics=metrics)
    else:
        raise Exception(f"Unknown split type {split_type}")
    
    return score


def train_test_validation(model , X, Y, train_test_split_size=.2, verbose=True, metrics=None, shuffle = True):

    indecies = np.arange(len(X))
    train_size = int(len(X)*(1-train_test_split_size))
    if shuffle:
        train_ind = np.random.choice(indecies, size=train_size, replace=False).flatten()
    else:
        train_ind = indecies[:train_size].flatten()
    test_ind = ~np.isin(indecies, train_ind)
    X_train, X_test = X[train_ind], X[test_ind]
    Y_train, Y_test = Y[train_ind], Y[test_ind]
    if verbose:
        print(f"Validating {model} using train-test split ")
    model.fit(X_train, Y_train)
    if metrics is None:
        score = model.score(X_test, Y_test)
    else:
        if metrics is not list: metrics = list(metrics)
        score = [metric(model.predict(X_test) ,Y_test)
                     for metric in metrics]
    if verbose:
        print(f"Test score {score}")
    return np.array(score)

def k_fold_validation(model, X, Y, num_folds=5, verbose=True, metrics=None, shuffle = True):

    scores = []

    indecies = np.arange(len(X))
    if shuffle:
        np.random.shuffle(indecies)

    fold_indecies = np.array_split(indecies, num_folds)

    if verbose:
        print(f"Validating {model} using k-fold validation")

    for i in range(num_folds):
        test_fold_ind = fold_indecies[i]
        train_folds_ind = np.concatenate(np.delete(fold_indecies, i,axis=0)
                                         ,axis=-1)
        X_train, X_test = X[train_folds_ind], X[test_fold_ind]
        Y_train, Y_test = Y[train_folds_ind], Y[test_fold_ind]
        model.fit(X_train, Y_train)

        if metrics is None:
            score = model.score(X_test, Y_test)
        else:
            if metrics is not list: metrics = list(metrics)
            score = [metric(model.predict(X_test) ,Y_test)
                     for metric in metrics]
        if verbose:
            print(f"Score on {i} fold is {score}")
        scores.append(score)
    print(f"Scores on all folds {scores}")
    return  np.array(scores).mean(axis=0)

def time_based_validation(model, X, Y, time_column_value,
                          num_folds=5, verbose=True, metrics=None):
    scores = []
    indecies = np.argsort(time_column_value)
    X, Y = X[indecies], Y[indecies]
    indecies = np.arange(len(X))
    fold_indecies = np.array_split(indecies, num_folds+1)

    if verbose:
        print(f"Validating {model} using time-based validation")

    for i in range(num_folds):
        test_fold_ind = fold_indecies[i+1]
        train_folds_ind = np.concatenate(fold_indecies[:i+1])

        X_train, X_test = X[train_folds_ind], X[test_fold_ind]
        Y_train, Y_test = Y[train_folds_ind], Y[test_fold_ind]
        model.fit(X_train, Y_train)

        if metrics is None:
            score = model.score(X_test, Y_test)
        else:
            if metrics is not list: metrics = list(metrics)
            score = [metric(model.predict(X_test) ,Y_test)
                     for metric in metrics]
        if verbose:
            print(f"Score on {i} fold is {score}")
        scores.append(score)
    print(f"Scores on all folds {scores}")
    return np.array(scores).mean(axis=0)
