from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from project.app.config import DevelopmentConfig, TestingConfig, ProductionConfig

app = Flask(__name__)
app.config.from_object(DevelopmentConfig())
db = SQLAlchemy(app)

from project.app.models import Offer, Subinfo, Apartment, Additions, Location
from project.app import routes


def run():
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )
