# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import psycopg2
import json
import pathlib

CONFIGURATION_FILE = pathlib.Path('../../secrets/dontdothis/please/ok/secrets.json')
COLUMNS = 'item_id, page_url, title, price_uah, price_usd, description, street, district_name, square_total, '\
          'square_living, square_kitchen, number_of_rooms, floor, number_of_floors, year_of_construction, type_of_sentence, ' \
          'walls_material, heating, longitude, latitude, price_verified, apartment_verified, publishing_date '


class DatabasePipeline(object):
    def __init__(self):
        self.connection = None
        self.cur = None
        self.items_ids = None

    def open_spider(self, spider):
        """
        Open connectivity to db with such configs
        :type spider: scrapy.Spider)
        """
        with open(CONFIGURATION_FILE, 'r') as file:
            config = json.load(file)
        self.connection = psycopg2.connect(
            host=config['HOST'],
            user=config['USER'],
            password=config['PASSWORD'],
            dbname=config['DATABASE']
        )

    def close_spider(self, spider):
        print('idi pospi 2')
        self.connection.commit()
        self.cur.close()
        self.connection.close()

    def process_item(self, item, spider):
        self.cur = self.connection.cursor()
        self.cur.execute('''
                    SELECT item_id FROM items;
                ''')
        self.items_ids = [id[0] for id in self.cur.fetchall()]

        if item['item_id'] not in self.items_ids and len(item['image_urls']) >= 2 and item['square_total'] <= 900:
            self.cur = self.connection.cursor()
            self.cur.execute(f"""
                INSERT INTO items({COLUMNS})
                VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                             (
                                 item['item_id'],
                                 item['page_url'],
                                 item['title'],
                                 item['price_uah'],
                                 item['price_usd'],
                                 item['description'],
                                 item['street'],
                                 item['district_name'],
                                 item['square_total'],
                                 item['square_living'],
                                 item['square_kitchen'],
                                 item['number_of_rooms'],
                                 item['floor'],
                                 item['number_of_floors'],
                                 item['year_of_construction'],
                                 item['type_of_sentence'],
                                 item['walls_material'],
                                 item['heating'],
                                 item['longitude'],
                                 item['latitude'],
                                 item['price_verified'],
                                 item['apartment_verified'],
                                 item['publishing_date']
                             )
                             )
            for image in item['image_urls']:
                self.cur.execute(f"INSERT INTO images (item_id, link) VALUES(%s, %s)", (item['item_id'], image))

        return item
