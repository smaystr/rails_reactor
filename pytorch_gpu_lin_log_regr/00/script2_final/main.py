import argparse
import pathlib
import torch
from torch.utils.data import Dataset, DataLoader

import json
from models import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score


class Loader_regression(Dataset):
	def __init__(self, path: pathlib.Path):
		data = torch.tensor(pd.read_csv(path).values, dtype=torch.float32)
		self.x_data = data[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]]
		self.y_data = data[:, [13]]
		self.len = data.shape[0]
		self.dim = self.x_data.shape[1]

	def dim(self):
		return self.dim

class Loader_classification(Dataset):
	def __init__(self, path: pathlib.Path):
		data = torch.tensor(pd.get_dummies(pd.read_csv(path)).values, dtype=torch.float32)
		self.x_data = data[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]]
		self.y_data = data[:, [3]]
		self.len = data.shape[0]
		self.dim = self.x_data.shape[1]
		
	def dim(self):
		return self.dim


def main(model: str, train_path: str, test_path: str, use_gpu: bool, input_path: pathlib.Path, output_path: pathlib.Path, config: dict):
	if (use_gpu and torch.cuda.is_available()): 
		device = torch.device('cuda') 
	else: 
		device = torch.device('cpu')

	if model == 'linear':
		train_data = Loader_regression(train_path)
		test_data = Loader_regression(test_path, model)
		model = LinearRegression(train_data.dim(), 1, device=device)
	else:
		train_data = Loader_classification(train_path, model)
		test_data = Loader_classification(test_path, model)
		model = LogisticRegression(train_data.dim(), 1, device=device)

	train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=config["batch_size"])
	test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=config["batch_size"])

	if not input_path is None:
		model.load_state_dict(torch.load(input_path))

	train_model(model, config["learning_rate"], config["optimizer"], config["epochs"], train_loader, device)

	X_test, y_test = next(iter(test_loader))
	X_test, y_test = X_test.to(device), y_test.to(device)
	with torch.no_grad():
		if model == 'linear':
			metric = (r2_score(y_test, model(X_test)))
		else: 
			metircs = accuracy_score(y_test, model(X_test))
		preds = model(X_test)

	torch.save(model.state_dict(), output_path)
	print(f"""
			Coef:
			{model.state_dict()}
			Test score: {metric}
			Validation loss: {validate_model(linear_model, test_loader, device)}
			Prediction for test set (first 5 rows):
			{pred[0:5, :]}
			Model's weights saved in {output_path}
			""")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help ='Linear or Logistic regression', type=str, required=True)
	parser.add_argument('--train_path', help='path to train dataset', type=str, required=True)
	parser.add_argument('--test_path', help='path to test dataset', type=str, required=True)
	parser.add_argument('--use_gpu', help='Want to use GPU?', type=bool, required=True)
	parser.add_argument('--input_path', help='Input path', type=pathlib.Path, required=False)
	parser.add_argument('--output_path', help='Output path', type=pathlib.Path, required=True)
	parser.add_argument('--config_path', help='Config file path', type=pathlib.Path, required=True)
	args = parser.parse_args()
	config = json.load(args.config_path.read_text())
	main(args.model, args.train_path, args.test_path, args.use_gpu, args.output_path, args.config_path)


