# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy import Item, Field


class DomRiaItem(Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    item_id = Field()
    page_url = Field()
    title = Field()
    price_uah = Field()
    price_usd = Field()
    image_urls = Field()
    description = Field()
    street = Field()
    district_name = Field()
    square_total = Field()
    square_living = Field()
    square_kitchen = Field()
    number_of_rooms = Field()
    floor = Field()
    year_of_construction = Field()
    number_of_floors = Field()
    type_of_sentence = Field()
    walls_material = Field()
    heating = Field()
    price_verified = Field()
    apartment_verified = Field()
    longitude = Field()
    latitude = Field()
    publishing_date = Field()
