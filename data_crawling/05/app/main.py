import os

from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy

from app.utilities import set_up_logging, load_env

web_app = Flask(__name__)
web_app.config.from_object(os.environ['APP_SETTINGS'])
web_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(web_app)


@web_app.route('/api/v1/statistics')
def statistics():
    from app.pg_db.queries import get_apartments_count
    return jsonify({'apartments_number': get_apartments_count(db)})


@web_app.route('/api/v1/records')
def records():
    limit = request.args.get('limit')
    offset = request.args.get('offset')

    from app.pg_db.queries import get_apartments
    return jsonify({'apartments': [x.serialize for x in get_apartments(db, limit, offset)]})


if __name__ == '__main__':
    load_env()
    set_up_logging(os.getenv('LOG_FILE'), bool(os.getenv('VERBOSE')))

    web_app.run(host=web_app.config['HOST'], port=web_app.config['PORT'])
