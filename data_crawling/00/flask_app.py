from flask import Flask, request, jsonify
from database import get_statistics, get_records


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False  # i wanted to keep the order of features
# (especially date first since the records are ordered by it)


@app.route('/')
def health():
    return {'status': 'Ok'}


@app.route('/api/v1/statistics/')
def statistics():
    stat = get_statistics()
    return {'Number of rows': stat}


@app.route('/api/v1/records/')
def records():
    rec = get_records(request.args.get('limit'), request.args.get('offset'))
    keys = ('Publication date', 'Title', 'Price in UAH', 'Price in USD', 'Description', 'Street', 'Region',
            'Total area', 'Living area', 'Number of rooms', 'Floor', 'Year of construction', 'Heating', 'Seller name',
            'Seller url', 'Walls material', 'Verified price', 'Verified apartment', 'Latitude', 'Longitude', 'Images')
    rec = [dict(zip(keys, i)) for i in rec]
    return jsonify(rec)
