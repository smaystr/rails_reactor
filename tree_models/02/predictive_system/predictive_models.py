import lightgbm as lgb
import pickle
from pathlib import Path
import os
import numpy as np
from predictive_system.network import Network
import torch


class PredictiveModels():
    def __init__(self, path_to_models: Path):
        self.path_to_models = path_to_models
        self.lgb_models = []
        self.decision_tree = None
        self.network = None
        print(path_to_models)
        self.load_lgb()
        self.load_decisions()
        self.load_network()

    def load_lgb(self):
        try:
            for file in self.path_to_models.glob('boosting_*.txt'):
                self.lgb_models.append(lgb.Booster(model_file=file.absolute().as_posix()))
        except Exception:
            print('lightgbm models not found')

    def load_decisions(self):
        try:
            with open(next(self.path_to_models.glob('decisiontree.pkl')), 'rb') as tree:
                self.decision_tree = pickle.load(tree)
        except Exception:
            print('Decision tree not found')

    def load_network(self):
        try:
            checkpoint = torch.load(next(self.path_to_models.glob('checkpoint.pth')))
            self.network = Network(1045, 256)
            self.network.load_state_dict(checkpoint['state_dict'])
            for parameter in self.network.parameters():
                parameter.requires_grad = False

            self.network.eval()
        except Exception:
            print('FNN not found')

    def predict_decision(self, data):
        return self.decision_tree.predict(data)[0]

    def predict_network(self, data):

        return self.network(torch.Tensor(data)).tolist()[0][0]

    def predict_boosting(self, data):
        predictions = np.zeros((len(self.lgb_models)))
        for key, model in enumerate(self.lgb_models):
            predictions[key] = model.predict(data)
        return predictions.mean()
