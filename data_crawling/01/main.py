from flask import Flask, jsonify, request
from app.apartment_repository import FlatData
from modeling.prod.decision_tree import DecisionTree
from modeling.prod.catboost_model import CatBoost
from modeling.prod.neural_network import NN

app = Flask(__name__)
flat = FlatData()

@app.route('/app/v1/statistics')
def statistics():
	# return number of apartments stored in DB
    return jsonify(flat.get_statistics_json())

@app.route('/app/v1/records')
def records():
	# return list of apartments sorted by the publication date
    limit = int(request.args.get('limit'))
    offset = int(request.args.get('offset'))
    return jsonify(flat.get_records(limit, offset))

@app.route('/app/v1/health')
def health():
	return {'status': 'ok'}

@app.route('/app/v1/words')
def most_common_words():
	number_of_words = int(request.args.get('number_of_words'))
	return jsonify(dict(flat.most_common_words(5)))

@app.route('/app/v1/price/predict')
def predict_flat_price():
	'''
	take parameters for the model:
	- type_of_proposal: 
	- city_name:
	- street_name:
	- total_area:
	- living_area:
	- kitchen_area:
	- floor:
	- total_number_of_floors:
	- number_of_rooms:
	- year_of_construction:
	- heating_type:
	- walls_type:
	- latitude:
	- longitude:
	'''
	#example_sample  = {"type_of_proposal":"Ð¾Ñ Ð¿Ð¾ÑÑÐµÐ´Ð½Ð¸ÐºÐ°", "date_created":"2019-08-13 17:33:39", "city_name":"ÐÐ¸Ð½Ð½Ð¸ÑÐ°", "total_area":63.0, "living_area":40.0, "kitchen_area":8.0,
	#"floor":5, "total_number_of_floors":9, "number_of_rooms":3, "year_of_construction":-1, "heating_type":"ÑÐµÐ½ÑÑÐ°Ð»Ð¸Ð·Ð¾Ð²Ð°Ð½Ð½Ð¾Ðµ", "walls_type":"Ð¿Ð°Ð½ÐµÐ»Ñ", 
	#"latitude":0.0, "longitude":0.0,
	#"description":"ÑÑÐ¾ÑÐ½Ð¾ Ð¿ÑÐ¾Ð´Ð°Ð²Ð°ÑÑÑÑ ÐºÐ²Ð°ÑÑÐ¸ÑÐ° ÑÐ¾ÑÐ¾ÑÐ¸Ð¹ ÑÐ°Ð¹Ð¾Ð½ Ð¾ÐºÐ½Ð¾ Ð¿Ð»Ð°ÑÑÐ¸ÐºÐ¾Ð²ÑÐ¹ Ð¿Ð¾ÑÐ¾Ð»Ð¾Ðº ÑÑÐµÐ½Ð° Ð²ÑÑÐ°Ð²Ð½Ð¸Ð²Ð°ÑÑ Ð²Ð°Ð½Ð½Ð°Ñ ÑÐ°Ð½ÑÐ·ÐµÐ» ÑÐ¾Ð²ÑÐµÐ¼ÐµÐ½Ð½ÑÐ¹ Ð¿Ð»Ð¸ÑÐºÐ° ÐºÐ²Ð°ÑÑÐ¸ÑÐ° ÑÐ¸ÑÑÑÐ¹ ÑÑÐ¾Ð¶ÐµÐ½Ð½ÑÐ¹ Ð±Ð°Ð»ÐºÐ¾Ð½ Ð·Ð°ÑÑÐµÐºÐ»ÑÑÑ ÑÑÐ¾ÑÐº Ð¿Ð¾Ð¼ÐµÐ½ÑÑÑ Ð¿Ð¾Ð» Ð»Ð°Ð¼Ð¸Ð½Ð°Ñ ÑÑÐ´ Ð±Ð°Ð·Ð°Ñ Ð²Ð¾ÐºÐ·Ð°Ð» ÑÑÐ°Ð½ÑÐ¿Ð¾ÑÑÐ½ÑÐ¹ ÑÐ°Ð·Ð²ÑÐ·ÐºÐ°"}
	sample = request.get_json()
	model_type = sample['model']
	if model_type == 'decision_tree':
		model = DecisionTree()
		prediction = model.predict(example_sample)
	elif model_type == 'catboost':
		model = CatBoost()
		prediction = model.predict(example_sample)
	else:
		model = NN()
		prediction = model.predict(example_sample)

	return jsonify({'predicted_price': int(prediction)})

if __name__ == '__main__':
    app.run('0.0.0.0', 8080)

