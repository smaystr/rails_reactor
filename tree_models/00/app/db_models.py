from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class BaseModel(db.Model):
    """Base data model for all objects"""

    __abstract__ = True

    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        """Define a base way to print models"""
        return "%s(%s)" % (
            self.__class__.__name__,
            {column: value for column, value in self.as_dict().items()},
        )

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Apartment(BaseModel, db.Model):
    """Model for the stations table"""

    __tablename__ = "apartments"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.Text)
    position = db.Column(db.Integer)
    price_usd = db.Column(db.Float)
    price_uah = db.Column(db.Float)
    number_rooms = db.Column(db.Integer)
    floor_located = db.Column(db.Integer)
    number_of_floors_in_the_house = db.Column(db.Integer)
    apartment_area = db.Column(db.Integer)
    offer_type = db.Column(db.Text)
    wall_type = db.Column(db.Text)
    construction_period = db.Column(db.Text)
    heating = db.Column(db.Text)
    city_id = db.Column(db.Integer)
    city_name = db.Column(db.Text)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    description = db.Column(db.Text)
    tags = db.Column(db.ARRAY(db.Text))
    image_urls = db.Column(db.ARRAY(db.Text))
    absolute_url = db.Column(db.Text)
