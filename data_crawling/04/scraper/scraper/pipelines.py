from sqlalchemy.orm import sessionmaker
from scraper.models import db_connect, create_table, ApartmentsModel


class ApartmentsPipeline(object):
    def __init__(self):
        engine = db_connect()
        create_table(engine)
        self.Session = sessionmaker(bind=engine)


    def process_item(self, item, spider):
        session = self.Session()
        apartmentsDB = ApartmentsModel(**item)

        try:
            session.add(apartmentsDB)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

        return item