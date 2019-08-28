import scrapy
import json

from operator import itemgetter
from collections import defaultdict

import house_parsing.item_selectors as item_selectors
from house_parsing.items import ApartmentItem


class ApartmentSpider(scrapy.Spider):

    name = "apartments"

    start_urls = [
        'https://dom.ria.com/prodazha-kvartir/',
    ]

    def extract_image_urls(self, response):

        image_urls = response.xpath(item_selectors.IMAGE_URL_PATTERN).extract()
        filtered_images = filter(lambda url: 'support' not in url, image_urls)
        get_big_pic_url = lambda url: url.replace('i.jpg', 'l.jpg')\
                                         .replace('m.jpg', 'f.jpg')
        return list(map(get_big_pic_url, filtered_images))

    def parse_initial_state(self, response):

        apartment_metadata = json.loads(response.xpath("//script")
                                 .re(item_selectors.INITIAL_STATE_PATTERN)[0],
                                  object_pairs_hook=lambda data: defaultdict(lambda: None, data))

        realty_dict = apartment_metadata['dataForFinalPage']['realty']

        attribute_getter = itemgetter(*item_selectors.ATTRIBUTES_FROM_INIT_WINDOW)
        attributes_extracted = attribute_getter(realty_dict)

        price_getter = itemgetter('1', '3')
        price_dict = realty_dict['priceArr']

        price_usd, price_uah = map(lambda x: float(x.replace(' ', '')) if x else None,
                                   price_getter(price_dict))

        return (price_usd, price_uah) + attributes_extracted

    def parse_page(self, response):

        preprocess_item = lambda string: '' if not string else string.strip ()
        preprocess_items = lambda iterable: '' if not iterable else list (map(preprocess_item, iterable))

        search_patterns = [item_selectors.TITLE_PATTERN,
                           item_selectors.HEATING_PATTERN,
                           item_selectors.OFFER_TYPE_PATTERN,
                           item_selectors.DESCRIPTION_PATTERN,
                           item_selectors.BUILT_TIME_PATTERN,
                           ]

        parsed_values = [preprocess_item(response.xpath(pattern).get())
                         for pattern in search_patterns]

        parsed_tags = preprocess_items(response.xpath(item_selectors.TAGS_PATTERN).extract())

        parsed_values.append(parsed_tags)

        return parsed_values


    def parse(self, response):

        # finding all apartments on a page
        for href in response.xpath(item_selectors.APARTMENT_PAGE_PATTERN):
            yield response.follow(href, self.parse_apartment)

        next_page_link = response.xpath(item_selectors.NEXT_PAGE_PATTERN)[0]
        yield response.follow(next_page_link, self.parse)

    def parse_apartment(self, response):

        price_usd, price_uah, rooms_count, publishing_date, \
            floor, position,apartment_id, area, total_floors, \
            city_id, city_name, wall_type, latitude, longitude = self.parse_initial_state(response)

        image_urls = self.extract_image_urls(response)

        title, heating, offer_type, description, built_time, tags = self.parse_page(response)

        apartment_item = ApartmentItem(
                                        id=apartment_id,
                                        title=title,
                                        position=position,
                                        price_usd=price_usd,
                                        price_uah=price_uah,
                                        number_rooms=rooms_count,
                                        floor_located=floor,
                                        number_of_floors_in_the_house=total_floors,
                                        apartment_area=area,
                                        offer_type=offer_type,
                                        wall_type=wall_type,
                                        construction_period=built_time,
                                        heating=heating,
                                        city_id=city_id,
                                        city_name=city_name,
                                        latitude=latitude,
                                        longitude=longitude,
                                        description=description,
                                        tags=tags,
                                        image_urls=image_urls,
                                        absolute_url=response.url,
                                        )
        yield apartment_item

