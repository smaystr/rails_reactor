from flask import request, Flask
from HA7.house_parsing.house_parsing.settings import DB_HOSTNAME, DB_PASSWORD, DB_NAME, DB_USERNAME
from HA7.house_parsing.app.models import db, Apartment

app = Flask(__name__)
app.config["DEBUG"] = True

app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOSTNAME}/{DB_NAME}"

db.init_app(app)

with app.app_context():
    db.create_all()


@app.route('/app/v1/statistics', methods=['GET'])
def get_count():
    count = Apartment.query.count()
    return {'count_houses': count}

@app.route('/app/v1/record', methods=['GET'])
def get_records():
    limit = request.args.get('limit')
    offset = request.args.get('offset')

    query = Apartment.query
    if limit:
        query = query.limit(limit)
    if offset:
        query = query.offset(offset)

    apartments = query.all()
    apartments = [item.as_dict() for item in apartments]
    return {'apartments': apartments}

if __name__ == '__main__':
    app.run()