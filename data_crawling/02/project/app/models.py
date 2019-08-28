from project.app import db


class Additions(db.Model):
    id = db.Column('id', db.Integer, primary_key=True)
    description = db.Column('description', db.Text, unique=False, nullable=True)
    heating = db.Column('heating', db.Text, unique=False, nullable=True)
    walls = db.Column('walls', db.Text, unique=False, nullable=True)
    apartment_id = db.Column('apartment_id', db.Integer, db.ForeignKey('apartment.id'), nullable=False)

    def __repr__(self):
        return f"Additions('{self.id}', '{self.description}')"


class Apartment(db.Model):
    id = db.Column('id', db.Integer, primary_key=True)
    year = db.Column('year', db.SmallInteger, unique=False, nullable=True)
    floor = db.Column('floor', db.SmallInteger, unique=False, nullable=True)
    rooms = db.Column('rooms', db.SmallInteger, unique=False, nullable=True)
    total_area = db.Column('total_area', db.Float, unique=False, nullable=True)
    living_area = db.Column('living_area', db.Float, unique=False, nullable=True)
    kitchen_area = db.Column('kitchen_area', db.Float, unique=False, nullable=True)
    offer_id = db.Column('offer_id', db.Integer, db.ForeignKey('offer.id'), nullable=False)

    def __repr__(self):
        return f"Apartment('{self.id}', '{self.square}')"


class Location(db.Model):
    id = db.Column('id', db.Integer, primary_key=True)
    longitude = db.Column('longitude', db.Float, unique=False, nullable=True)
    latitude = db.Column('latitude', db.Float, unique=False, nullable=True)
    region = db.Column('region', db.Text, unique=False, nullable=True)
    city = db.Column('city', db.Text, unique=False, nullable=True)
    street = db.Column('street', db.Text, unique=False, nullable=True)
    apartment_id = db.Column('apartment_id', db.Integer, db.ForeignKey('apartment.id'), nullable=False)

    def __repr__(self):
        return f"Location('{self.id}', '{self.longitude}')"


class Offer(db.Model):
    id = db.Column('id', db.Integer, primary_key=True)
    title = db.Column('title', db.Text, unique=False, nullable=True)
    seller = db.Column('seller', db.Text, unique=False, nullable=True)
    price = db.Column('price', db.Float, unique=False, nullable=True)
    price_verification = db.Column('price_verification', db.Boolean, unique=False, nullable=True)
    apartment_verification = db.Column('apartment_verification', db.Boolean, unique=False, nullable=True)

    def __repr__(self):
        return f"Offer('{self.id}', '{self.title}')"


class Subinfo(db.Model):
    id = db.Column('id', db.Integer, primary_key=True)
    publish_date = db.Column('publish_date', db.Date, unique=False, nullable=True)
    image_urls = db.Column('image_urls', db.ARRAY(db.Text), unique=False, nullable=True)
    offer_id = db.Column('offer_id', db.Integer, db.ForeignKey('offer.id'), nullable=False)

    def __repr__(self):
        return f"Subinfo('{self.id}', '{self.publish_date}')"