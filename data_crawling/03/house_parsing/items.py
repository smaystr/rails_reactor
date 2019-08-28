# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ApartmentItem(scrapy.Item):

    id = scrapy.Field()
    title = scrapy.Field()
    position = scrapy.Field()
    price_usd = scrapy.Field()
    price_uah = scrapy.Field()
    number_rooms = scrapy.Field()
    floor_located = scrapy.Field()
    number_of_floors_in_the_house = scrapy.Field()
    apartment_area = scrapy.Field()
    offer_type = scrapy.Field()
    wall_type = scrapy.Field()
    construction_period = scrapy.Field()
    heating = scrapy.Field()
    city_id = scrapy.Field()
    city_name = scrapy.Field()
    latitude = scrapy.Field()
    longitude = scrapy.Field()
    description = scrapy.Field()
    tags = scrapy.Field()
    image_urls = scrapy.Field()
    absolute_url = scrapy.Field()