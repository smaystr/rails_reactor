from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker, Session
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

    def count_apartments(self):
        with self.session_scope() as session:
            res = session.query(func.count(self.table.c.id)).scalar()

        return res

    def count_params(self):
        with self.session_scope() as session:
            res = len(self.table.c) - 1

        return res

    def records(self, limit, offset):
        with self.session_scope() as session:
            res = session.query(self.table).order_by(self.table.c.creation_date).offset(offset).limit(limit).all()

        return res

    def state_price_statistics(self):
        with self.session_scope() as session:
            states = session.query(self.table.c.state_name).group_by(self.table.c.state_name).all()
            statistics = {}
            for state in states:
                statistics.update({state[0]: {
                    'min': session.query(func.min(self.table.c.price)).filter(self.table.c.state_name == state).scalar(),
                    'max': session.query(func.max(self.table.c.price)).filter(self.table.c.state_name == state).scalar(),
                    'mean': int(session.query(func.avg(self.table.c.price)).filter(self.table.c.state_name == state).scalar()),
                    'std': session.query(func.stddev_samp(self.table.c.price)).filter(self.table.c.state_name == state).scalar()
                }})

        return statistics

    def properties_statistics(self):
        categorical_props = {
            'state_name': self.table.c.state_name,
            'rooms_count': self.table.c.rooms_count,
            'floor': self.table.c.floor,
            'wall_type': self.table.c.wall_type,
            'heating': self.table.c.heating,
            'seller': self.table.c.seller,
            'water': self.table.c.water,
            'building_condition': self.table.c.building_condition
        }

        numeric_props = {
            'total_square_meters': self.table.c.total_square_meters,
            'price': self.table.c.price,
            'price_uah': self.table.c.price_uah,
            'construction_year': self.table.c.construction_year
        }

        statistics = {}
        for prop in categorical_props:
            statistics.update({prop: self._property_types(categorical_props[prop])})

        for prop in numeric_props:
            statistics.update(
                {prop: self._property_statistics(numeric_props[prop])})

        return statistics

    def _property_statistics(self, prop):
        with self.session_scope() as session:
            statistics = {
                'min': session.query(func.min(prop)).scalar(),
                'max': session.query(func.max(prop)).scalar(),
                'mean': int(session.query(func.avg(prop)).scalar()),
                'std': int(session.query(func.stddev_samp(prop)).scalar())
            }

        return statistics

    def _property_types(self, prop):
        with self.session_scope() as session:
            all_types = session.query(prop).group_by(prop).all()
            statistics = {'number_of_types': len(all_types)}
            for prop_type in all_types:
                statistics.update({prop_type[0]: session.query(prop).filter(prop == prop_type).count()})

        return statistics
