from flask import Flask, jsonify, request
from apartment_repository import ApartmentRepository

apartments_app = Flask(__name__)
repository = ApartmentRepository()


@apartments_app.route('/api/v1/statistics')
def statistics():
    return jsonify(repository.get_statistics_json())


@apartments_app.route('/api/v1/records')
def records():
    limit = int(request.args.get('limit'))
    offset = int(request.args.get('offset'))
    return jsonify(repository.get_records(limit, offset))


if __name__ == '__main__':
    apartments_app.run('0.0.0.0', 5000)
