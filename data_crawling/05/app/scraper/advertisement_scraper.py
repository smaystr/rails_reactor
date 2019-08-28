import json
import re
from urllib.parse import urlparse

import scrapy
from scrapy import Request


class AdvertisementScrapper(scrapy.Spider):
    name = 'advertisements'

    def __init__(self, start_url, max_parsed_pages, **kwargs):
        super().__init__(**kwargs)
        self.start_urls = [start_url]

        parsed_uri = urlparse(start_url)
        result = '{uri.scheme}://{uri.netloc}'.format(uri=parsed_uri)
        self.base_url = result
        self.parsed_pages = 0
        self.max_parsed_pages = max_parsed_pages
        self.properties = {
            'Комнат': 'rooms_count',
            'Этаж': 'floor',
            'Тип предложения': 'seller',
            'Тип стен': 'wall_type',
            'Отопление': 'heating',
            'Год постройки': 'construction_year',
            'характеристика здания': 'building_condition',
            'вода': 'water',
            'до центра города': 'dist_to_center',
            'жд вокзал': 'dist_to_railway_station',
            'аэропорт': 'dist_to_airport'
        }
        self.initial_state_props = [
            'rooms_count',
            'city_name',
            'street_name',
            'state_name',
            'created_at',
            'total_square_meters',
            'living_square_meters',
            'kitchen_square_meters',
            'inspected',
            'latitude',
            'longitude',
            'floors_count'
        ]

    def parse(self, response):
        self.parsed_pages += 1

        btn = response.css("a.blue ::attr(href)")
        links = [x.get() for x in btn if not x.get().startswith('http')]

        for link in links:
            yield Request(self.base_url + link, callback=self.parse_article)

        next_page = response.css('.page-item.next.text-r .page-link::attr(href)').get()
        if next_page is not None and self.parsed_pages <= self.max_parsed_pages:
            yield Request(self.base_url + next_page, callback=self.parse)

    def parse_article(self, response):
        apartment = {
            'title': response.css('.finalPage h1::text').extract_first(),
            'price_usd': int(response.css('.price::text').extract_first().replace('$', '').replace(' ', '')),
            'price_uah': int(
                response.css('.grey.size13 span::text').extract_first().replace('грн', '').replace(' ', '')),
            'description': response.css(
                '.mb-15.print_height.description.descriptionHidden .boxed::text').extract_first(),
            'verified_price': response.css('.row.finalPage .unstyle .mb-15 .ml-30 .blue ::text').extract_first(),
            'images': [url.css('img::attr(src)').extract_first() for url in response.css('.photo-74x56')]
        }

        initial_state = re.findall(
            r'window.__INITIAL_STATE__={.*};',
            response.body.decode('utf-8'),
            flags=re.DOTALL)[0].replace('window.__INITIAL_STATE__=', '').replace(';', '')
        initial_state = json.loads(initial_state)['dataForFinalPage']['realty']
        for prop in self.initial_state_props:
            if prop in initial_state.keys():
                if not initial_state[prop] == '':
                    apartment[prop] = initial_state[prop]

        for prop in response.css('.mt-15.boxed.v-top'):
            key = prop.css('.label.grey::text').extract_first().strip()
            if key in self.properties:
                apartment[self.properties[key]] = prop.css('.indent::text').extract_first().strip()

        for prop in response.css('.mt-20'):
            key = prop.css('.label.grey::text').extract_first().strip()
            if key in self.properties:
                apartment[self.properties[key]] = prop.css('.boxed::text').extract_first().strip()

        yield Request(self.base_url + f'/ru/realtor-{initial_state["user_id"]}.html',
                      callback=self.parse_seller,
                      cb_kwargs=dict(apartment=apartment, user_id=initial_state["user_id"]))

    def parse_seller(self, response, apartment, user_id):
        apartment['seller_info'] = {
            'user_id': user_id,
            'name': " ".join(
                [x for x in response.css(".header-page h1::text").get().strip().replace('\n', '').split(' ') if
                 x != '']),
            'phone': response.css(".phoneShowLink::attr(data-phone)").get(),
            'location': response.css(".ml-20::text").get()
        }
        yield apartment
