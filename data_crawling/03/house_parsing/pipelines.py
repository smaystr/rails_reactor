# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


import psycopg2
from HA7.house_parsing.house_parsing.settings import DB_HOSTNAME, DB_PASSWORD, DB_NAME, DB_USERNAME
from HA7.house_parsing.app.models import Apartment, db
from HA7.house_parsing.app.app import app




class ApartmentPipeline(object):

    def open_spider(self, spider):

        db.app = app
        db.init_app(app)

    def close_spider(self, spider):
        db.close_all_sessions()

    def validate_data(self, item):
        process_value = lambda value: value if value else None
        return {item: process_value(value) for item, value in item.items()}

    def process_item(self, item, spider):

        dict_to_insert = self.validate_data(item)
        apartment_item = Apartment(**dict_to_insert)

        try:
            db.session.add(apartment_item)
            db.session.commit()

        except Exception as e:
            print(f"Error raised {e}")
            db.session.rollback()
        return item
