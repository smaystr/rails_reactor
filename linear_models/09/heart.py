from pathlib import Path
from logistic_regression import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression as SKLR
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np

data_path = Path(Path(__file__).parent) / 'data'


def get_X_y(filename):
    data = pd.read_csv(data_path / filename)
    data['sex_oh'] = LabelBinarizer().fit_transform(data['sex'])
    X = data[['sex_oh', 'age', 'trestbps', 'chol', 'thalach', 'oldpeak']]
    y = data['target']

    return X, y


X_train, y_train = get_X_y('heart_train.csv')
X_test, y_test = get_X_y('heart_test.csv')
#
clf = LogisticRegression().fit(X_train, y_train, epochs=10000)
y_pred = clf.predict(X_test)
y_pred_train = clf.predict(X_train)


sklearn_clf = SKLR(solver='lbfgs', max_iter=1000).fit(X_train, y_train)
y_pred_sklearn = sklearn_clf.predict(X_test)
y_pred_sklearn_train = sklearn_clf.predict(X_train)

print('sklearn LR coef', sklearn_clf.coef_)

random_y_pred = np.random.binomial(size=y_test.shape[0], n=1, p=0.5)

print('own implementation train f1 score', f1_score(y_pred_train, y_train, average='macro')) # 0.74-0.76
print('own implementation test f1 score', f1_score(y_pred, y_test, average='macro')) # 0.74-0.76
print('sklearn train f1 score', f1_score(y_pred_sklearn_train, y_train, average='macro')) # 0.73-0.74
print('sklearn test f1 score', f1_score(y_pred_sklearn, y_test, average='macro')) # 0.73-0.74
print('random test f1 score', f1_score(random_y_pred, y_test, average='macro')) # 0.49-0.57


