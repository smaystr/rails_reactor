from flask import request, jsonify

from project.app import db, app
from project.app.queries import get_statistics, get_records


@app.route('/api/v1/statistics')
def show_statistics():
    status, result = get_statistics(
        db=db,
        url=app.config.get('SQLALCHEMY_DATABASE_URI')
    )
    return jsonify({
        'status': status,
        'result': result
    })


@app.route('/api/v1/records')
def show_records():
    limit = request.args.get('limit')
    offset = request.args.get('offset')
    status, result = get_records(
        db=db,
        url=app.config.get('SQLALCHEMY_DATABASE_URI'),
        limit=limit,
        offset=offset
    )
    return jsonify({
        'status': status,
        'result': result
    })