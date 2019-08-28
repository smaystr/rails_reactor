from app.main import db


class Apartment(db.Model):
    __tablename__ = 'apartment'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column('title', db.String)
    created_at = db.Column('created_at', db.DateTime)
    price_usd = db.Column('price_usd', db.Integer)
    price_uah = db.Column('price_uah', db.Integer)
    description = db.Column('description', db.Text)
    street_name = db.Column('street_name', db.String, nullable=True)
    state_name = db.Column('state_name', db.String, nullable=True)
    city_name = db.Column('city_name', db.String, nullable=True)
    total_square_meters = db.Column('total_square_meters', db.Float, nullable=True)
    living_square_meters = db.Column('living_square_meters', db.Float, nullable=True)
    kitchen_square_meters = db.Column('kitchen_square_meters', db.Float, nullable=True)
    rooms_count = db.Column('rooms_count', db.Integer, nullable=True)
    floor = db.Column('floor', db.String, nullable=True)
    floors_count = db.Column('floors_count', db.String, nullable=True)
    wall_type = db.Column('wall_type', db.String, nullable=True)
    inspected = db.Column('inspected', db.String, nullable=True)
    verified_price = db.Column('verified_price', db.String, nullable=True)
    latitude = db.Column('latitude', db.Float, nullable=True)
    longitude = db.Column('longitude', db.Float, nullable=True)
    construction_year = db.Column('construction_year', db.String, nullable=True)
    heating = db.Column('heating', db.String, nullable=True)
    seller = db.Column('seller', db.String, nullable=True)
    water = db.Column('water', db.String, nullable=True)
    building_condition = db.Column('building_condition', db.String, nullable=True)
    dist_to_center = db.Column('dist_to_center', db.String, nullable=True)
    dist_to_railway_station = db.Column('dist_to_railway_station', db.String, nullable=True)
    dist_to_airport = db.Column('dist_to_airport', db.String, nullable=True)

    def __init__(self,
                 title,
                 created_at,
                 price_usd,
                 price_uah,
                 description,
                 street_name=None,
                 state_name=None,
                 city_name=None,
                 total_square_meters=None,
                 living_square_meters=None,
                 kitchen_square_meters=None,
                 rooms_count=None,
                 floor=None,
                 floors_count=None,
                 wall_type=None,
                 inspected=None,
                 verified_price=None,
                 latitude=None,
                 longitude=None,
                 construction_year=None,
                 heating=None,
                 seller=None,
                 water=None,
                 building_condition=None,
                 dist_to_center=None,
                 dist_to_railway_station=None,
                 dist_to_airport=None):
        self.title = title
        self.created_at = created_at
        self.price_usd = price_usd
        self.price_uah = price_uah
        self.description = description
        self.street_name = street_name
        self.state_name = state_name
        self.city_name = city_name
        self.total_square_meters = total_square_meters
        self.living_square_meters = living_square_meters
        self.kitchen_square_meters = kitchen_square_meters
        self.rooms_count = rooms_count
        self.floor = floor
        self.floors_count = floors_count
        self.wall_type = wall_type
        self.inspected = inspected
        self.verified_price = verified_price
        self.latitude = latitude
        self.longitude = longitude
        self.construction_year = construction_year
        self.heating = heating
        self.seller = seller
        self.water = water
        self.building_condition = building_condition
        self.dist_to_airport = dist_to_airport
        self.dist_to_center = dist_to_center
        self.dist_to_railway_station = dist_to_railway_station

    @property
    def serialize(self):
        return {
            'title': self.title,
            'created_at': self.created_at,
            'price_usd': self.price_usd,
            'price_uah': self.price_uah,
            'description': self.description,
            'street_name': self.street_name,
            'state_name': self.state_name,
            'city_name': self.city_name,
            'total_square_meters': self.total_square_meters,
            'living_square_meters': self.living_square_meters,
            'kitchen_square_meters': self.kitchen_square_meters,
            'rooms_count': self.rooms_count,
            'floor': self.floor,
            'floors_count': self.floors_count,
            'wall_type': self.wall_type,
            'inspected': self.inspected,
            'verified_price': self.verified_price,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'construction_year': self.construction_year,
            'heating': self.heating,
            'seller': self.seller,
            'water': self.water,
            'building_condition': self.building_condition,
            'dist_to_airport': self.dist_to_airport,
            'dist_to_center': self.dist_to_center,
            'dist_to_railway_station': self.dist_to_railway_station,
        }


class ApartmentImage(db.Model):
    __tablename__ = 'apartment_images'

    id = db.Column(db.Integer, primary_key=True)
    apartment_id = db.Column('apartment_fk', db.Integer,
                             db.ForeignKey('apartment.id', ondelete='CASCADE', onupdate='CASCADE'))
    link_url = db.Column('link_url', db.String)

    def __init__(self, apartment_id, link_url):
        self.apartment_id = apartment_id
        self.link_url = link_url


class SellerInfo(db.Model):
    __tablename__ = 'seller_info'

    id = db.Column(db.Integer, primary_key=True)
    apartment_id = db.Column('apartment_fk', db.Integer,
                             db.ForeignKey('apartment.id', ondelete='CASCADE', onupdate='CASCADE'))
    user_id = db.Column('user_id', db.Integer)
    name = db.Column('name', db.String, nullable=True)
    phone = db.Column('phone', db.String, nullable=True)
    location = db.Column('location', db.String, nullable=True)

    def __init__(self, user_id, apartment_id, name=None, phone=None, location=None):
        self.user_id = user_id
        self.apartment_id = apartment_id
        self.name = name
        self.phone = phone
        self.location = location
