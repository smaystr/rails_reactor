from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
import sqlalchemy
from settings import DATABASE


class ApartmentRepository:
    def __init__(self):
        engine = create_engine(URL(**DATABASE))
        self.session = sessionmaker(bind=engine)
        meta = sqlalchemy.MetaData()
        meta.reflect(bind=engine)
        self.table = meta.tables.get('apartments')

    @contextmanager
    def session_scope(self):
        session = self.session()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def get_statistics_json(self):
        return {
            'totalNumber': self.count(),
            'countryPriceStats': self.price_statistics(),
            'cityPriceStats': self.state_price_statistics(),
            'propertyStats': self.properties_statistics()
        }

    def get_records(self, limit, offset):
        return {'records': self.records(limit, offset)}

    def count(self):
        with self.session_scope() as session:
            res = session.query(func.count(self.table.c.id)).scalar()
        return res

    def records(self, limit, offset):
        with self.session_scope() as session:
            res = session.query(self.table).order_by(self.table.c.creation_date).offset(offset).limit(limit).all()
        return res

    def price_statistics(self):
        with self.session_scope() as session:
            statistics = {
                'min_price': session.query(func.min(self.table.c.price)).scalar(),
                'max_price': session.query(func.max(self.table.c.price)).scalar(),
                'avg_price': int(session.query(func.avg(self.table.c.price)).scalar())
            }
        return statistics

    def state_price_statistics(self):
        with self.session_scope() as session:
            states = session.query(self.table.c.state_name).group_by(self.table.c.state_name).all()
            statistics = {}
            for state in states:
                min_price = session.query(func.min(self.table.c.price)).filter(
                    self.table.c.state_name == state).scalar()
                max_price = session.query(func.max(self.table.c.price)).filter(
                    self.table.c.state_name == state).scalar()
                avg_price = session.query(func.avg(self.table.c.price)).filter(
                    self.table.c.state_name == state).scalar()
                statistics.update({state[0]: {'min': int(min_price), 'max': int(max_price), 'avg': int(avg_price)}})
        return statistics

    def properties_statistics(self):
        props = {
            'city_name': self.table.c.state_name,
            'district_name': self.table.c.district_name,
            'rooms_count': self.table.c.rooms_count,
            'wall_type': self.table.c.wall_type,
            'heating': self.table.c.heating,
            'seller': self.table.c.seller,
            'water': self.table.c.water
        }
        statistics = {}
        for prop in props:
            statistics.update({prop: self._property_types(props[prop])})
        return statistics

    def _property_types(self, prop):
        with self.session_scope() as session:
            all_types = session.query(prop).group_by(prop).all()
            statistics = {'number_of_types': len(all_types)}
            for prop_type in all_types:
                statistics.update({prop_type[0]: session.query(prop).filter(prop == prop_type).count()})
        return statistics
