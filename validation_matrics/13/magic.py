import numpy as np
import random
import itertools
import time

from preprocessing import prepare_data
from logistic_regression import LogisticRegression
from metrics import rmse, F1, mse, recall, mae, accuracy,precision


class Validation:

    def __init__(self, dataset, target, model, output, split_type, cv_size, time_series, hyper_fit, d):

        self.dataset = dataset
        self.target = target
        self.model = model
        self.output = output
        self.split_type = split_type
        self.cv_size = cv_size
        self.time_serries = time_series
        self.hyper_fit_algo = hyper_fit
        self.n_samples_params = 5
        self.folds = list()
        self.kfolds_indexes = list()
        self.d = d
        self.params = {
            'lr': [1e-3, 1e-4, 1e-5],
            'epoch': [1000, 10000, 100000],
            'regularization': ['L1', 'L2'],
            'alpha': [0.1, 0.5, 1, 2]
        }

    def make_test_and_validation_dataset(self, random_state=22):
        random.seed(random_state)

        m = len(self.dataset)
        test_size = int(0.1 * m)

        test_rows = random.sample(range(m), test_size)
        self.test_dataset = self.dataset[test_rows, :]
        self.validation_dataset = self.dataset[[row for row in range(m) if not (row in test_rows)], :]

    def hold_out(self, random_state=228, k_fold=False, val_indexes=None):
        random.seed(random_state)
        m = len(self.validation_dataset)

        if k_fold:
            val_dataset = self.validation_dataset[val_indexes, :]
            train_dataset = self.validation_dataset[[row for row in range(m) if not (row in val_indexes)], :]
        else:
            val_size = int(self.cv_size * m)
            val_rows = random.sample(range(m), val_size)
            val_dataset = self.validation_dataset[val_rows, :]
            train_dataset = self.validation_dataset[[row for row in range(m) if not (row in val_rows)], :]

        X_train = np.delete(train_dataset, self.target, 1)
        y_train = train_dataset[:, self.target].reshape((-1, 1)).astype(dtype=np.float64)
        X_val = np.delete(val_dataset, self.target, 1)
        y_val = val_dataset[:, self.target].reshape((-1, 1)).astype(dtype=np.float64)

        self.folds.append(np.array([X_train, y_train, X_val, y_val]))

    def partition(self):
        k = self.cv_size
        index = random.shuffle(list(range(len(self.validation_dataset))))
        self.kfolds_indexes = [index[i::k] for i in range(k)]

    def k_fold(self, loocv=False):
        if loocv:
            self.kfolds_indexes = [[i] for i in range(len(self.validation_dataset))]
        else:
            self.partition()

        for fold_indexes in self.kfolds_indexes:
            self.hold_out(k_fold=True, val_indexes=fold_indexes)

    def hyperparameter_fitting(self):
        if self.model == LogisticRegression:
            self.params['threshold'] = [0.4, 0.5, 0.6]
            self.metric = F1
            best_params = (1e-4, 10000, 'L2', 1, 0.5)
        else:
            self.metric = rmse
            best_params = (1e-4, 10000, 'L2', 1)
        search_params = list()

        if self.hyper_fit_algo == 'random search':
            for i in range(self.n_samples_params):
                search_params.append(tuple(random.sample(j, 1)[0] for j in self.params.values()))
        elif self.hyper_fit_algo == 'grid search':
            search_params = list(itertools.product(*self.params.values()))

        try:
            search_params.remove(best_params)
        except Exception:
            pass
        print(f'I am saerching in {search_params}\n')
        f = self.folds[0]
        X_train, y_train, X_val, y_val = f[0], f[1], f[2], f[3]

        best_model = self.model(*best_params)
        best_model.fit(X_train, y_train)
        best_score = best_model.score(X_val, y_val, self.metric)

        for params in search_params:
            model = self.model(*params)
            score = self.fitting(model)
            if best_score < score:
                best_params = params
                best_score = score
                best_model = model

        return best_params, best_score, best_model

    def fitting(self, model):
        '''
        Estimating our model at folds
        :param folds: if 'k-fold': folds=k, if 'train-test split': folds=1, if 'leave one-out': k=m.
        :return: mean score
        '''
        scores = list()
        for fold in self.folds:
            train_x, train_y, val_x, val_y = fold[0], fold[1], fold[2], fold[3],
            md = model
            md.fit(train_x.astype(dtype=np.float64), train_y.reshape((-1, 1)).astype(dtype=np.float64))
            scores.append(md.score(val_x.astype(dtype=np.float64), val_y.reshape((-1, 1)).astype(dtype=np.float64), self.metric))

        return np.mean(scores)

    def make_folds(self):
        if self.split_type == 'train-test_split':
            print('you have train/test split')
            self.hold_out()
        elif self.split_type == 'k-fold':
            print('you have k-fold')
            self.k_fold()
        elif self.split_type == 'leave_one-out':
            print('you have leave one-out')
            self.k_fold(loocv=True)
        else:
            print('you have TROUBLES')

    def write_output_model(self, params, model):
        with open('output.model', 'w') as f:
            f.write(f'best hyperparameters - {params},\nweights - {model.w}')

    def write_output_info(self, o):
        with open('output.info', 'w') as f:
            f.write(o)

    def fit(self):
        start_time = time.time()
        if self.d == 1:
            print('Probably this is heart disease dataset, I know how to preprocess him')
            columns_to_transform = {'to_standartise': [0, 3, 4, 7, 9, ],
                       'to_onehot_encoding': [2, 6, 11, 12]}
            #print(self.dataset.astype(dtype=np.float32))
            self.dataset = self.dataset.astype(dtype=np.float32)
            self.dataset = prepare_data(self.dataset, columns_to_transform)
            metrics_to_show = [F1, recall, precision, accuracy]
        else:
            print('Probably this is Medical Cost Personal dataset, I know how to preprocess him')
            columns_to_transform = {'to_standartise': [0, 2, 3, -1],
                                    'to_onehot_encoding': [5]}
            columns_to_numeric = [1, 4, 5]
            self.dataset = prepare_data(self.dataset, columns_to_transform, columns_to_numeric)
            metrics_to_show = [mae, rmse, mse]
        self.make_test_and_validation_dataset()
        self.make_folds()
        self.folds = np.array(self.folds)

        params, score, model = self.hyperparameter_fitting()
        print(params, score)
        self.write_output_model(params, model)
        x_test = np.delete(self.test_dataset, self.target, 1)
        y_test = self.test_dataset[:, self.target].reshape((-1, 1)).astype(dtype=np.float64)

        output_info = f'time -- {time.time() - start_time} sec\n '
        for f in metrics_to_show:
            output_info += f'{f.__name__}  --  {f(model.predict(x_test), y_test)}\n'
        print(output_info)
        self.write_output_info(output_info)





