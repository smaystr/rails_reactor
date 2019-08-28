import argparse
import pathlib
import torch
import json
from models import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

def load_data_regression(data_path: str, device: torch.device):
	data = torch.tensor(pd.get_dummies(pd.read_csv(data_path)).values, dtype=torch.float, device=devie)
	X_data = data[:, [0,1,2,4,5,6,7,8,9,10,11]]
	y_data = data[:, [3]]
	return X_data, y_data

def load_data_classification(data_path: str, device: torch.device):
	data = torch.tensor(pd.read_csv(data_path).values, dtype=torch.float, device=device)
	X_data = data[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]]
	y_data = data[:, [13]]
	return X_data, y_data

def main(model: str, train_path: str, test_path: str, use_gpu: bool, output_path: pathlib.Path, config: dict):
	if (use_gpu and torch.cuda.is_available()) : 
		device = torch.device('cuda') 
	else: 
		device = torch.device('cpu')
	config.update({'device': device})
	scaler = StandardScaler()

	if model == 'linear':
		X_train, y_train = load_data_regression(train_path, device)
		X_test, y_test = load_data_linear(train_path, device)
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)
		model = LinearRegression(**config).fit(X_train, y_train)
		preds_train = model.predict(X_train)
		preds_test = model.predict(X_test)
		torch.save(model.coef, output_path)
		print(f"""
				Coef:
				{model.coef}
				R2 for train: {r2_score(y_train, preds_train)}
				R2 for test: {r2_score(y_test, preds_test)}
				Prediction for test set (first 5 rows):
				{model.predict(X_test)[:5, :]}
				Model's weights saved in {output_path}
				""")
	else:
		X_train, y_train = load_data_classification(train_path, device)
		X_test, y_test = load_data_classification(test_path, device)
		X_train = scaler.fit_transform(X_trian)
		X_test = scaler.transform(X_test)
		model == LogisticRegression(**config).fit(X_train, y_train)
		preds_train = model.predict(X_train)
		preds_test = model.predict(X_test)
		torch.save(model.coef, output_path)
		print(f"""
				Coef:
				{model.coef}
				Train accuracy: {accuracy_score(y_train, preds_train)}
				Test accuracy: {accuracy_score(y_test, preds_test)}
				Prediction for test set (first 5 rows):
				{model.predict(X_test)[:5, :]}
				Prob prediction for test set (first 5 rows):
				{model.predict_proba(X_test)[:5, :]}
				Model's weights saved in {output_path}
				""")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help ='Linear or Logistic regression', type=str, required=True)
	parser.add_argument('--train_path', help='path to train dataset', type=str, required=True)
	parser.add_argument('--test_path', help='path to test dataset', type=str, required=True)
	parser.add_argument('--use_gpu', help='Want to use GPU?', type=bool, required=True)
	parser.add_argument('--output_path', help='Output path', type=pathlib.Path, required=True)
	parser.add_argument('--config_path', help='Config file path', type=pathlib.Path, required=True)
	args = parser.parse_args()
	config = json.load(args.config_path.read_text())
	main(args.model, args.train_path, args.test_path, args.use_gpu, args.output_path, args.config_path)


