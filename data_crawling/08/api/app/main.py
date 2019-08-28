import dbaccess as db

from model.train_model import train_model, predict_price
from decimal import Decimal
from flask import Flask, request, jsonify, render_template
from ast import literal_eval

web_app = Flask(__name__)
web_app.config["JSON_AS_ASCII"] = False


@web_app.route("/app/v1/statistics")
def statistics():
    conn = db.open_db()

    response = {'number of apartments': db.count_items(connection=conn),
                'number of params per apartment': len(db.make_query(conn, "select * from items limit 1;")[0]) + 1,
                'statistics': {'price_usd': db.main_statistics(conn, 'price_usd'),
                               'price_uah': db.main_statistics(conn, 'price_uah'),
                               'square_total': db.main_statistics(conn, 'square_total')}}
    return response


@web_app.route("/app/v1/records")
def records():
    limit = request.args.get("limit", 10)
    offset = request.args.get("offset", 0)
    conn = db.open_db()
    return jsonify(db.select_apartments(conn, limit, offset))


@web_app.route('/app/v1/pandas_profiling')
def pandas_profiling():
    return render_template("pandas_profile.html")


@web_app.route("/app/v1/health")
def health():
    return {'status': 'ok'}


@web_app.route("/app/v1/predict")
def predict():
    p = request.args.get("params")
    to_train = request.args.get("train",  '0')
    params = literal_eval(p)
    print(params, type(params), '----1111111-----')
    if to_train == '1':
        test_score = train_model()
        print('------2---------')
        predicted_price, true_price = predict_price(params)
        return jsonify({'predicted price ': predicted_price,
                        'RMSE for test': int(test_score),
                        'true price': int(true_price)})

    elif to_train == '0':
        predicted_price, true_price = predict_price(params)
        return jsonify({'predicted price ': predicted_price,
                        'true price': int(true_price)})

    else:
        return f"I dont understand you, {to_train}"


def my_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj


if __name__ == "__main__":
    web_app.run("0.0.0.0", 8080)
