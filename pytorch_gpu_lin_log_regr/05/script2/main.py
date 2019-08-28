import argparse
import pathlib
import torch
import json
from models import LinearRegression, LogisticRegression, train_model, validate_model
from load_data import HeartDataset, InsuranceDataset
from hw6.metrics import *


def main(model: str, train: str, test: str, pu_type: str, input_path: pathlib.Path, output_path: pathlib.Path, config: dict):
	device = (torch.device("cuda") if (pu_type == "gpu" and torch.cuda.is_available()) else torch.device("cpu"))

	train_dataset = (InsuranceDataset(train) if model == "linear" else HeartDataset(train))
	test_dataset = InsuranceDataset(test) if model == "linear" else HeartDataset(test)
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config["batch_size"])
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config["batch_size"])
	linear_model = (LinearRegression(train_dataset.get_dim(), 1, device=device) if model == "linear" else LogisticRegression(train_dataset.get_dim(), 1, device=device))
	if not input_path is None:
		linear_model.load_state_dict(torch.load(input_path))

	train_model(linear_model, config["alpha"], config["optim"], config["epochs"], train_loader, device)

	X_test, y_test = next(iter(test_loader))
	X_test, y_test = X_test.to(device), y_test.to(device)
	with torch.no_grad():
		metric = (r2(y_test, linear_model(X_test)) if model == "linear" else accuracy(y_test, linear_model(X_test)))
		pred = linear_model(X_test)

	torch.save(linear_model.state_dict(), output_path)

	print(f"""
			Coefficients:
			{linear_model.state_dict()}
			Score for test dataset: {metric}
			Validation loss: {validate_model(linear_model, test_loader, device)}
			Prediction for test set (first 5 rows):
			{pred[0:5, :]}
			Model's weights saved in {output_path}
			""")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", help="type of regression (linear or logistic)", type=str, required=True)
	parser.add_argument("--train", help="path to the train dataset", type=str, required=True)
	parser.add_argument("--test", help="path to the test dataset", type=str, required=True)
	parser.add_argument("--pu_type", help="type of processing unit", type=str, required=True)
	parser.add_argument("--input_path", help="path for loading weights", type=pathlib.Path, required=False)
	parser.add_argument("--output_path", help="path for saving weights", type=pathlib.Path, required=True)
	parser.add_argument("--config_path", help="path to the configuration file", type=pathlib.Path, required=True)
	args = parser.parse_args()

	config = json.loads(args.config_path.read_text())
	main(args.model, args.train, args.test, args.pu_type, args.input_path, args.output_path, config)
