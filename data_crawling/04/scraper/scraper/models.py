from sqlalchemy import create_engine, Column, Integer, Float, Boolean, String, Text, Date
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from scraper.settings import DATABASE


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
    price = Column('price', Integer)
    price_uah = Column('price_uah', Integer)
    images = Column('images', String)
    description = Column('description', Text)
    street_name = Column('street_name', String, nullable=True)
    state_name = Column('state_name', String, nullable=True)
    total_square_meters = Column('total_square_meters', Float, nullable=True)
    living_square_meters = Column('living_square_meters', Float, nullable=True)
    kitchen_square_meters = Column('kitchen_square_meters', Float, nullable=True)
    rooms_count = Column('rooms_count', Integer, nullable=True)
    floor = Column('floor', String, nullable=True)
    wall_type = Column('wall_type', String, nullable=True)
    inspected = Column('inspected', String, nullable=True)
    verified_price = Column('verified_price', String, nullable=True)
    latitude = Column('latitude', Float, nullable=True)
    longitude = Column('longitude', Float, nullable=True)
    construction_year = Column('construction_year', String, nullable=True)
    heating = Column('heating', String, nullable=True)
    seller = Column('seller', String, nullable=True)
    water = Column('water', String, nullable=True)
    building_condition = Column('building_condition', String, nullable=True)
    dist_to_center = Column('dist_to_center', String, nullable=True)
    dist_to_kindergarten = Column('dist_to_kindergarten', String, nullable=True)
    dist_to_school = Column('dist_to_school', String, nullable=True)
    dist_to_hospital = Column('dist_to_hospital', String, nullable=True)
    dist_to_bus_station = Column('dist_to_bus_station', String, nullable=True)
    dist_to_railway_station = Column('dist_to_railway_station', String, nullable=True)
    dist_to_airport = Column('dist_to_airport', String, nullable=True)
