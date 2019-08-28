import multiprocessing
import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.signalmanager import dispatcher

from app.pg_db.queries import add_apartment
from app.scraper.advertisement_scraper import AdvertisementScrapper
from app.utilities import load_env

web_app = Flask(__name__)
web_app.config.from_object(os.environ['APP_SETTINGS'])
web_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(web_app)


def queue_worker(queue, db):
    for item in iter(queue.get, 'STOP'):
        add_apartment(db, item)
    # logging.info(f'All url processed by this worker')
    return


def start_crawler(start_url, max_parsed_pages, num_processes, db):
    queue = multiprocessing.Queue()
    pool = [multiprocessing.Process(target=queue_worker, args=(queue, db)) for _ in range(num_processes)]
    for process in pool:
        process.start()

    def crawler_results(signal, sender, item, response, spider):
        """
        help function for getting result when one page scrapped
        :param signal:
        :param sender:
        :param item:
        :param response:
        :param spider:
        :return:
        """
        queue.put(item)

    dispatcher.connect(crawler_results, signal=signals.item_passed)
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })
    process.crawl(
        AdvertisementScrapper,
        start_url=start_url,
        max_parsed_pages=max_parsed_pages
    )
    process.start()


if __name__ == '__main__':
    load_env()
    start_crawler('https://dom.ria.com/prodazha-kvartir/', int(os.getenv('PAGES')), int(os.getenv('CPU_COUNT')), db)
