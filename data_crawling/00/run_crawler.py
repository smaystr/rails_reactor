from scrapy.crawler import CrawlerProcess
from spiders.spider import ApartmentsSpider

process = CrawlerProcess({'EXTENSIONS': {'scrapy.extensions.closespider.CloseSpider': 100},
                          'ITEM_PIPELINES': {'pipelines.ApartmentsPipeline': 300},
                          'CLOSESPIDER_ITEMCOUNT': 20000,
                          'DOWNLOAD_DELAY': 1,
                          'AUTOTHROTTLE_ENABLED': True,
                          'USER AGENT': 'Andrey13771 Homework. Contact: andrey13771@gmail.com'})
process.crawl(ApartmentsSpider)
process.start()
