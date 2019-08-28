import scrapy
import re
import json
from urllib.parse import urljoin
from scraper.items import Apartment
from scraper.util import date_converter, MAIN_PROPERTIES, ADDITIONAL_PROPERTIES, JSON_PROPERTIES


class ApartmentsSpider(scrapy.Spider):
    name = 'apartmentsspider'
    start_urls = ['https://dom.ria.com/prodazha-kvartir/']
    domain = 'https://dom.ria.com'
    pages_number = 0
    total_pages_number = 15000


    def parse(self, response):
        for href in response.css('.size18.tit.mb-0'):
            self.pages_number += 1
            print('PAGE NUMBER: ',self.pages_number)
            yield scrapy.Request(urljoin(self.domain, href.css('.blue ::attr(href)').extract_first()), self.parse_page)

        next_page = response.css('.page-item.next.text-r .page-link::attr(href)').extract_first()
        if not next_page is None and self.pages_number <= self.total_pages_number:
            yield scrapy.Request(urljoin(self.domain, next_page), self.parse)


    def parse_page(self, response):
        apartment = Apartment()
        apartment['title'] = response.css('.finalPage h1::text').extract_first()
        apartment['creation_date'] = date_converter(response.css('.row.finalPage .greyLight.size13.unstyle .mt-15 b::text').extract_first())
        apartment['price'] = int(response.css('.price::text').extract_first().replace('$', '').replace(' ', ''))
        apartment['price_uah'] = int(response.css('.grey.size13 span::text').extract_first().replace('грн', '').replace(' ', ''))
        apartment['description'] = response.css('.mb-15.print_height.description.descriptionHidden .boxed::text').extract_first()
        apartment['verified_price'] = response.css('.row.finalPage .unstyle .mb-15 .ml-30 .blue ::text').extract_first()

        urls = ''
        for url in response.css('.photo-74x56'):
            urls += url.css('img::attr(src)').extract_first() + ';'
        apartment['images'] = urls

        self.find_values(response, apartment, '.mt-15.boxed.v-top', '.label.grey::text', '.indent::text', MAIN_PROPERTIES)
        self.find_values(response, apartment, '.mt-20', '.label.grey::text', '.boxed::text', ADDITIONAL_PROPERTIES)

        initial_state = re.findall(r'window.__INITIAL_STATE__={.*};', response.body.decode('utf-8'), flags=re.DOTALL)[0].replace('window.__INITIAL_STATE__=', '').replace(';', '')
        parsed = json.loads(initial_state)['dataForFinalPage']['realty']

        keys = parsed.keys()
        for prop in JSON_PROPERTIES:
            if prop in keys:
                if not parsed[prop] == '':
                    apartment[prop] = parsed[prop]
                else:
                    apartment[prop] = None

        all_keys = apartment.fields.keys()
        present_keys = apartment.keys()
        for key in all_keys:
            if not key in present_keys:
                apartment[key] = None

        return apartment


    def find_values(self, response, apartment, list_selector, key_selector, value_selector, props):
        for prop in response.css(list_selector):
            key = prop.css(key_selector).extract_first().strip()
            if key in props:
                apartment[props[key]] = prop.css(value_selector).extract_first().strip()
