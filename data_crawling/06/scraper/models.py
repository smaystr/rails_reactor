from sqlalchemy import create_engine, Column, ARRAY, Integer, Float, Boolean, String, Text, Date
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from settings import DATABASE

DeclarativeBase = declarative_base()

def db_connect():
    return create_engine(URL(**DATABASE))


def create_table(engine):
    DeclarativeBase.metadata.create_all(engine)


class ApartmentsModel(DeclarativeBase):
    __tablename__ = 'apartments'

    id = Column(Integer, primary_key=True)
    title = Column('title', String)
    creation_date = Column('creation_date', Date)
    price_usd = Column('price', Integer)
    price_uah = Column('price_uah', Integer)
    description = Column('description', Text)
    rooms_count = Column('rooms_count', Integer, nullable=True)
    floor = Column('floor', String, nullable=True)
    seller = Column('seller', String, nullable=True)
    wall_type = Column('wall_type', String, nullable=True)
    construction_year = Column('construction_year', String, nullable=True)
    heating = Column('heating', String, nullable=True)
    is_verified_price = Column('is_verified_price', Boolean, nullable=True)
    is_verified_flat = Column('is_verified_flat', Boolean, nullable=True)
    realty_id = Column('realty_id', Integer, nullable=True)
    street_name = Column('street_name', String, nullable=True)
    city_name = Column('city_name', String, nullable=True)
    district_name = Column('district_name', String, nullable=True)
    longitude = Column('longitude', Float, nullable=True)
    latitude = Column('latitude', Float, nullable=True)
    total_square_meters = Column('total_square_meters', Float, nullable=True)
    living_square_meters = Column('living_square_meters', Float, nullable=True)
    kitchen_square_meters = Column('kitchen_square_meters', Float, nullable=True)
    photos = Column('photos', ARRAY(String))
    water = Column('water', String, nullable=True)
    building_condition = Column('building_condition', String, nullable=True)
    elevators_count = Column('elevators_count', Integer, nullable=True)
    dist_to_center = Column('dist_to_center', String, nullable=True)
    dist_to_school = Column('dist_to_school', String, nullable=True)
    dist_to_kindergarten = Column('dist_to_kindergarten', String, nullable=True)
    dist_to_bus_terminal = Column('dist_to_bus_terminal', String, nullable=True)
    dist_to_railway_station = Column('dist_to_railway_station', String, nullable=True)
    dist_to_airport = Column('dist_to_airport', String, nullable=True)
    dist_to_hospital = Column('dist_to_hospital', String, nullable=True)
    dist_to_shop = Column('dist_to_shop', String, nullable=True)
    dist_to_parking = Column('dist_to_parking', String, nullable=True)
    dist_to_rest_area = Column('dist_to_rest_area', String, nullable=True)
