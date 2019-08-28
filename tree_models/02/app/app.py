from flask import Flask, jsonify, request
from predictive_system.predictive_models import PredictiveModels
from pathlib import Path
from predictive_system.preprocess import Dataset
import numpy as np
import pandas as pd

app = Flask(__name__)
path_to_package = '../predictive_system'
prediction = PredictiveModels(Path(path_to_package) / 'models')
preprocessing = Dataset(to_save_encoder=False, db=False)

columns_in_dataset = ['description', 'street', 'region',
                      'total_area', 'room_count', 'construction_year', 'heating', 'seller',
                      'wall_material', 'verified_price', 'verified_apartment', 'latitude',
                      'longitude', 'city', 'title'
                      ]
numeric_columns = ['total_area', 'room_count', 'latitude', 'longitude', 'verified_price', 'verified_apartment']
models = {
    'boosting': prediction.predict_boosting,
    'decision': prediction.predict_decision,
    'network': prediction.predict_network
}

preprocessing.load_encoders_and_stuff(Path(path_to_package) / 'encoders')


@app.route("/api/v1/price/predict", methods=["GET"])
def get_stats():
    to_predict = [[request.args.get(x, default=np.nan) for x in columns_in_dataset]]
    for column in numeric_columns:
        ind = columns_in_dataset.index(column)
        try:
            to_predict[0, ind] = float(to_predict[0, ind])
        except Exception:
            to_predict[0, ind] = 0

    model_to_use = request.args.get('model', default=np.nan)
    if np.isnan(model_to_use):
        return jsonify{"Error": "Provide a valid model to predict with."}
    preprocessing.data = pd.DataFrame(to_predict, columns=columns_in_dataset)
    preprocessing.get_text_features()
    preprocessing.encode_labels()
    preprocessing.encode_text()
    preprocessing.one_hot_encode()
    preprocessing.replace_nans()
    preprocessing.to_numpy(with_target=False)

    if model_to_use == 'network':
        preprocessing.scale_data()

    return jsonify({'predicted_price': models[model_to_use](preprocessing.data)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
