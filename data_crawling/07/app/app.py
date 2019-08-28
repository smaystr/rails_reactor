from flask import Flask, jsonify, request
from apartment_db import ApartmentOverview

app = Flask(__name__)
overview = ApartmentOverview()


@app.route("/app/v1/statistics", methods=["GET"])
def get_stats():
    return jsonify({"house_count": overview.get_count(),
                    "average_uah_price": overview.get_mean_uah(),
                    "average_usd_price": overview.get_mean_usd()})


@app.route("/app/v1/most_common_words", methods=['GET', 'POST'])
def get_most_common_words():
    try:
        words = abs(int(request.args.get("amount")))
    except ValueError:
        return "Bad input. Must be int"
    words = max(words, overview.number_of_words)

    return jsonify(dict(overview.most_common_words()[:words]))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
