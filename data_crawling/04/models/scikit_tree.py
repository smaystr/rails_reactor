from os import cpu_count
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def cross_validate(X, y, random_state, metric='neg_mean_absolute_error', n_splits=5, n_jobs=1):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = DecisionTreeRegressor(splitter='random',
                                max_depth=20,
                                min_samples_split=50,
                                min_samples_leaf=5,
                                random_state=random_state)

    scores = cross_val_score(estimator=model, X=X, y=y, scoring=metric, cv=kf, n_jobs=n_jobs)
    model.fit(X, y)

    return model, scores


def grid_search(X, y, estimator, param_grid, scoring='neg_mean_absolute_error', n_jobs=1):
    gs = GridSearchCV(estimator=estimator,
                        param_grid=param_grid,
                        n_jobs=n_jobs, 
                        scoring=scoring)
    gs.fit(X, y)

    return gs.best_score_, gs.best_params_


def get_xy(df, target):
    return df.drop(target, axis=1), df[target]


def main():
    random_state = 0
    n_jobs = max(cpu_count() - 1, 1)

    apartments = pd.read_csv('apartments_reduced_types.csv')
    apartments = pd.get_dummies(apartments)
    X, y = get_xy(apartments, 'price')

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.1,
                                                        random_state=random_state)

    model, score = cross_validate(X_train, y_train, random_state=random_state, n_jobs=n_jobs)

    param_grid = {'splitter': ['best', 'random'],
                'max_depth': [5, 8, 10, 20],
                'min_samples_split': [10, 20, 50],
                'min_samples_leaf': [5, 10]}
    gs_score, params = grid_search(X_train, y_train, estimator=model, param_grid=param_grid, n_jobs=n_jobs)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    res = (f'Grid search score: {gs_score}\n'
            f'Grid search best params: {params}\n'
            f'Train r2(cross_val): {score}\n'
            f'Test mae: {mean_absolute_error(y_test, y_pred_test)}\n'
            f'Train mae: {mean_absolute_error(y_train, y_pred_train)}\n'
            f'Test mse: {mean_squared_error(y_test, y_pred_test)}\n'
            f'Train mse: {mean_squared_error(y_train, y_pred_train)}\n'
            f'Train r2: {model.score(X_train, y_train)}\n'
            f'Test r2: {model.score(X_test, y_test)}\n')

    print(res)


if __name__ == '__main__':
    main()
