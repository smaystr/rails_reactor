import utils
import linear_models
import model_selection
import numpy as np
#  python main.py http://ps2.railsreactor.net/datasets/medicine/heart_test.csv target classification res --split_type k-fold --hyper_param_fit random_search


def main():
    args = utils.parse_arguments()

    target = args.target
    path = args.path_data
    task = args.task_type
    path_output = args.path_output
    split_type = args.split_type

    if not path_output.is_dir():
        path_output.mkdir()

    X, y, columns = utils.preprocess_csv(path, target)

    scaler = utils.StandardScaler()
    X = scaler.fit_transform(X)

    if task == 'classification':
        model = linear_models.LogisticRegression()
    else:
        model = linear_models.LinearRegression()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y)

    if args.hyper_param_fit is None:

        if split_type == 'train-test':
            model.fit(X_train, y_train)
        else:
            if split_type == 'k-fold':
                kfold = model_selection.KFold()
            else:
                kfold = model_selection.KFold(X.shape[0])
            split_func = kfold.split(X_train)

            scores = []

            for train_ind, test_ind in split_func:
                X__train, X__test = X_train[train_ind], X_train[test_ind]
                y__train, y__test = y_train[train_ind], y_train[test_ind]

                model.fit(X__train, y__train)

                scores.append(model.score(X__test, y__test))
            score = np.mean(scores)
            print('Average score on k-fold is ', score)

    elif args.hyper_param_fit is not None:

        parameters = {
            'max_iter': [1e2, 1e3],  # , 1e4],
            'lr': [1, 1e-1],  # , 1e-2],
            'C': [1, 1e-1, 1e-2],
            'penalty':  [None, 'l1', 'l2']
        }

        if args.hyper_param_fit == 'grid_search':
            param_search = model_selection.GridSearchCV(
                model, parameters, cv=3)
        else:
            param_search = model_selection.RandomizedSearchCV(
                model, parameters, cv=3)

        param_search.fit(X_train, y_train)

        model = param_search.best_estimator

    utils.create_model_file(path_output, task, model)
    utils.create_info_file(path_output, task, model, X_test, y_test, columns)


if __name__ == '__main__':
    main()
