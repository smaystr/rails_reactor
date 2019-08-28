from flask import Flask, jsonify, request
import pickle
import torch
import pathlib
import time
from apartment_repository import ApartmentRepository
from preprocessing import preprocess_row
from models.fnn import FNN

web_app = Flask(__name__)
web_app.config['JSON_AS_ASCII'] = False

db = ApartmentRepository()

with open(pathlib.Path.cwd().joinpath('app/models/scikit_tree.pkl'), 'rb') as file:
    scikit_tree = pickle.load(file)

with open(pathlib.Path.cwd().joinpath('app/models/xgboost_tree.pkl'), 'rb') as file:
    xgboost_tree = pickle.load(file)

fnn = FNN(input_dim=71, hidden_dim=25, hidden_num=3)
fnn.load_state_dict(torch.load(pathlib.Path.cwd().joinpath('app/models/fnn.pt')))


@web_app.route('/api/v1/statistics')
def statistics():
    return jsonify({
        'number_of_apartments': db.count_apartments(),
        'number_of_params': db.count_params(),
        'state_price_stats': db.state_price_statistics(),
        'property_stats': db.properties_statistics()
    })


@web_app.route('/api/v1/records')
def records():
    limit = int(request.args.get('limit'))
    offset = int(request.args.get('offset'))

    return jsonify({'records': db.records(limit, offset)})


@web_app.route('/api/v1/price/predict')
def predict():
    params = request.args.get('params')
    model, features = preprocess_row(params)

    if model == 'decision_tree':
        prediction = scikit_tree.predict(features)[0]
    elif model == 'gradient_boosting':
        prediction = float(xgboost_tree.predict(features)[0])
    elif model == 'fnn':
        with torch.no_grad():
            prediction = fnn(torch.tensor(features.values, dtype=torch.float32)).item()

    return jsonify({'predicted_price': prediction})


if __name__ == '__main__':
    web_app.run('0.0.0.0', 8080)
