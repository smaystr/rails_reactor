from os import environ
from psycopg2 import connect
from bot.utilities import load_environment


class DatabasePipeline(object):
    def __init__(self):
        self.connection = None
        self.cur = None
        load_environment(
            path='bot/environments',
            filename='.env'
        )

    def open_spider(self, spider):
        """
        Open connectivity to db with such configs
        :type spider: scrapy.Spider
        """
        self.connection = connect(
            host=environ['DB_HOST'],
            user=environ['DB_USER'],
            password=environ['DB_PASSWORD'],
            dbname=environ['DB_NAME']
        )

    def close_spider(self):
        self.cur.close()
        self.connection.close()

    def process_item(self, item, spider):
        """
        Add the item to the database
        :type item: scrapy.Item
        :type spider: scrapy.Spider
        """
        # Pushing items into the offer table
        self.cur = self.connection.cursor()
        try:
            self.cur.execute("""
                INSERT INTO public.offer(id, title, seller, price, price_verification, apartment_verification)
                VALUES(%s, %s, %s, %s, %s, %s)""",
                    (
                        item['id'],
                        item['title'],
                        item['seller'],
                        item['price'],
                        item['price_verification'],
                        item['apartment_verification'],
                    )
            )
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
        # # Pushing items into the subinfo table
        self.cur = self.connection.cursor()
        try:
            self.cur.execute("""
                INSERT INTO public.subinfo(id, publish_date, image_urls, offer_id)
                VALUES(%s, %s, %s, %s)""",
                    (
                        item['id'],
                        item['publish_date'],
                        item['image_urls'],
                        item['id']
                    )
            )
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
        # # Pushing items into the apartment table
        self.cur = self.connection.cursor()
        try:
            self.cur.execute("""
                INSERT INTO public.apartment(id, year, floor, rooms, total_area, living_area, kitchen_area, offer_id)
                VALUES(%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        item['id'],
                        item['year'],
                        item['floor'],
                        item['rooms'],
                        item['total_area'],
                        item['living_area'],
                        item['kitchen_area'],
                        item['id']
                    )
            )
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
        # # Pushing items into the additions table
        self.cur = self.connection.cursor()
        try:
            self.cur.execute("""
                INSERT INTO public.additions(id, description, heating, walls, apartment_id)
                VALUES(%s, %s, %s, %s, %s)""",
                    (
                        item['id'],
                        item['description'],
                        item['walls'],
                        item['heating'],
                        item['id'],
                    )
            )
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
        # # Pushing items into the location table
        self.cur = self.connection.cursor()
        try:
            self.cur.execute("""
                INSERT INTO public.location(id, longitude, latitude, region, city, street, apartment_id)
                VALUES(%s, %s, %s, %s, %s, %s, %s)""",
                    (
                        item['id'],
                        item['longitude'],
                        item['latitude'],
                        item['region'],
                        item['city'],
                        item['street'],
                        item['id']
                    )
            )
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
        return item
