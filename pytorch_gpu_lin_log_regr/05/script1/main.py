import argparse
import pathlib
import torch
import json
from models import LinearRegression, LogisticRegression
from preprossesing import load_data, normalize
from hw6.metrics import *


def main(model: str, train: str, test: str, pu_type: str, output_path:pathlib.Path, config: dict):
	device = torch.device('cuda') if (pu_type == 'gpu' and torch.cuda.is_available()) else torch.device('cpu')
	config.update({'device': device})
	X, y = load_data(train, model, device)
	X_test, y_test = load_data(test, model, device)
	X = normalize(X)
	X_test = normalize(X_test)

	lr = LogisticRegression(**config).fit(X, y) if model == 'logistic' else LinearRegression(**config).fit(X, y)
	y_pred_train = lr.predict(X)
	y_pred_test = lr.predict(X_test)
	torch.save(lr.coef, output_path)

	if model == 'logistic':
		print(f"""
				Coefficients:
				{lr.coef}
				Score fot training set: {accuracy(y, y_pred_train)}
				Score for test set: {accuracy(y_test, y_pred_test)}
				Prediction for test set (first 5 rows):
				{lr.predict(X_test)[0:5, :]}
				Probability prediction for test set (first 5 rows):
				{lr.predict_proba(X_test)[0:5, :]}
				Model's weights saved in {output_path}
				""")
	else:
		print(f"""
				Coefficients:
				{lr.coef}
				Score fot training set: {r2(y, y_pred_train)}
				Score for test set: {r2(y_test, y_pred_test)}
				Prediction for test set (first 5 rows):
				{lr.predict(X_test)[0:5, :]}
				Model's weights saved in {output_path}
				""")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='type of regression (linear or logistic)', type=str, required=True)
	parser.add_argument('--train', help='path to the train dataset', type=str, required=True)
	parser.add_argument('--test', help='path to the test dataset', type=str, required=True)
	parser.add_argument('--pu_type', help='type of processing unit', type=str, required=True)
	parser.add_argument("--output_path", help="path for saving weights", type=pathlib.Path, required=True)
	parser.add_argument('--config_path', help='path to the configuration file', type=pathlib.Path, required=True)
	args = parser.parse_args()

	config = json.loads(args.config_path.read_text())
	main(args.model, args.train, args.test, args.pu_type, args.output_path, config)