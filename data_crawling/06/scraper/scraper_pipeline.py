from sqlalchemy.orm import sessionmaker
from apartments_urls_scraper import ApartmentUrlsScraper
from apartment_scraper import ApartmentScraper
from models import db_connect, create_table, ApartmentsModel


class ScraperPipeline:
    def __init__(self):
        engine = db_connect()
        create_table(engine)
        self.Session = sessionmaker(bind=engine)

    def _save_apartment(self, item):
        session = self.Session()
        apartments_model = ApartmentsModel(**item)
        try:
            session.add(apartments_model)
            session.commit()
        except:
            session.rollback()
        finally:
            session.close()

    def scrap_apartments(self):
        urls_scraper = ApartmentUrlsScraper()
        apartment_urls = urls_scraper.scrap_dom_ria_pages()
        apartment_scraper = ApartmentScraper()
        for url in apartment_urls:
            try:
                apartment_item = apartment_scraper.parse_apartment(url)
                if apartment_item:
                    self._save_apartment(apartment_item)
            except:
                print(f'ups something went wrong, url is {url}')


if __name__ == '__main__':
    scraper_pipeline = ScraperPipeline()
    scraper_pipeline.scrap_apartments()
