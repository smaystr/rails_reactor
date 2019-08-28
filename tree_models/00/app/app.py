from flask import request, Flask
from sqlalchemy.sql import func

from settings import DB_HOSTNAME, DB_PASSWORD, DB_NAME, DB_USERNAME
from db_models import db, Apartment
from helpers import *

app = Flask(__name__)
app.config["DEBUG"] = True

app.config[
    "SQLALCHEMY_DATABASE_URI"
] = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOSTNAME}/{DB_NAME}"

db.init_app(app)


@app.route("/app/v1/statistics", methods=["GET"])
def get_statistics():
    count = Apartment.query.count()
    agg_stats = db.session.query(
        func.avg(Apartment.price_uah).label("mean_price"),
        func.stddev_samp(Apartment.price_uah).label("std_price"),
        func.avg(Apartment.apartment_area).label("mean_area"),
        func.stddev_samp(Apartment.apartment_area).label("std_area"),
    )

    mean_price, std_price, mean_area, std_area = list(map(float, agg_stats.all()[0]))

    return {
        "number_of_apartments": count,
        "mean_price": mean_price,
        "std_price": std_price,
        "mean_area": mean_area,
        "std_area": std_area,
    }


@app.route("/app/v1/record", methods=["GET"])
def get_records():
    limit = request.args.get("limit")
    offset = request.args.get("offset")

    query = Apartment.query
    if limit:
        query = query.limit(limit)
    if offset:
        query = query.offset(offset)

    apartments = query.all()
    apartments = [item.as_dict() for item in apartments]
    return {"apartments": apartments}


@app.route("/app/v1/predict", methods=["GET"])
def get_prediction():
    model_type = request.args.get("model")
    model = load_model(model_type)
    features = request.args.get("features")
    features = check_features(features)
    price = predict(model, features)
    return {"predicted_price": price}


if __name__ == "__main__":
    app.run()
