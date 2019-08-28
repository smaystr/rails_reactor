from __future__ import absolute_import

import scrapy
from os import environ
from json import loads

from bot.spiders import getters
from bot.items import DomRiaItem
from bot.utilities import find, find_feature_in
from bot.parsers import parse_year, parse_images, parse_verification, parse_price


class DomRiaSpider(scrapy.Spider):
    name = 'dom-ria-spider'

    domain = 'https://dom.ria.com'

    custom_settings = {
        'CLOSESPIDER_ITEMCOUNT': environ.get('CLOSESPIDER_ITEMCOUNT'),
    }

    def start_requests(self):
        urls = [
            'https://dom.ria.com/prodazha-kvartir/',
        ]
        for url in urls:
            yield scrapy.Request(url, self.parse)

    def parse(self, response):
        for href in response \
                .xpath(getters.GET_PAGE_CONTENT) \
                .getall():
            yield scrapy.Request(self.domain + href, self.parse_page)

        next_page = response \
            .xpath(getters.GET_NEXT_PAGE) \
            .extract_first()
        if not (next_page is None):
            yield scrapy.Request(self.domain + next_page, self.parse)

    def parse_page(self, response):

        window_initial_state = response \
            .xpath(getters.GET_WINDOW_INITIAL_STATE) \
            .re("window.__INITIAL_STATE__=(.+}}}})")[0]

        dictionary = loads(window_initial_state)

        apartment_id = dictionary \
            .get('dataForFinalPage', None) \
            .get('realty', None) \
            .get('realty_id', None)

        title = response \
            .xpath(getters.GET_TITLE) \
            .extract_first()

        seller = dictionary \
            .get('dataForFinalPage', None) \
            .get('agencyOwner', None) \
            .get('owner', None) \
            .get('name', None)

        price = response \
            .xpath(getters.GET_PRICE) \
            .extract_first()
        price = parse_price(price)

        description = response \
            .xpath(getters.GET_DESCRIPTION) \
            .extract_first()

        image_urls = response \
            .xpath(getters.GET_IMAGE_URLS) \
            .extract()
        image_urls = parse_images(image_urls)

        additional_information = find(
            dictionary=dictionary,
            feature='chars'
        )

        year = find_feature_in(
            generator_obj=additional_information,
            feature='Год постройки'
        )
        year = parse_year(year)

        additional_information = find(
            dictionary=dictionary,
            feature='chars'
        )

        heating = find_feature_in(
            generator_obj=additional_information,
            feature='Отопление'
        )

        walls = dictionary \
            .get('dataForFinalPage', None) \
            .get('realty', None) \
            .get('wall_type', None)

        total_area = dictionary \
            .get('dataForFinalPage', None) \
            .get('realty', None) \
            .get('total_square_meters', None)

        living_area = dictionary \
            .get('dataForFinalPage', None) \
            .get('realty', None) \
            .get('living_square_meters', None)

        kitchen_area = dictionary \
            .get('dataForFinalPage', None) \
            .get('realty', None) \
            .get('kitchen_square_meters', None)

        rooms = dictionary \
            .get('dataForFinalPage', None) \
            .get('realty', None) \
            .get('rooms_count', None)

        floor = dictionary \
            .get('dataForFinalPage', None) \
            .get('realty', None) \
            .get('floor', None)

        street = dictionary \
            .get('dataForFinalPage', None) \
            .get('realty', None) \
            .get('street_name', None)

        city = dictionary \
            .get('dataForFinalPage', None) \
            .get('realty', None) \
            .get('city_name_uk', None)

        region = dictionary \
            .get('dataForFinalPage', None) \
            .get('realty', None) \
            .get('state_name_uk', None)

        latitude = dictionary \
            .get('dataForFinalPage', None) \
            .get('realty', None) \
            .get('latitude', None)
        if latitude == '':
            latitude = None

        longitude = dictionary \
            .get('dataForFinalPage', None) \
            .get('realty', None) \
            .get('longitude', None)
        if longitude == '':
            longitude = None

        verification = response.xpath(getters.GET_PRICE_VERIFICATION).extract()
        price_verification, apartment_verification = parse_verification(verification)

        publish_date = dictionary \
            .get('dataForFinalPage', None) \
            .get('realty', None) \
            .get('publishing_date', None)

        item = DomRiaItem(
            # Item id
            id=apartment_id,
            # Offer
            title=title,
            seller=seller,
            price=price,
            price_verification=price_verification,
            apartment_verification=apartment_verification,
            # Subinfo
            publish_date=publish_date,
            image_urls=image_urls,
            # Apartment
            year=year,
            floor=floor,
            rooms=rooms,
            total_area=total_area,
            living_area=living_area,
            kitchen_area=kitchen_area,
            # Additions
            description=description,
            heating=heating,
            walls=walls,
            # Location
            longitude=longitude,
            latitude=latitude,
            region=region,
            city=city,
            street=street
        )
        return item
