import scrapy
import re
import json

from scrapy import Spider
from my_crawler.items import DomRiaItem


class DomRiaSpider(Spider):
    name = 'domria'
    allowed_domains = ['dom.ria.com']
    start_url = 'https://dom.ria.com/prodazha-kvartir/'
    domain = 'https://dom.ria.com'
    current_page, max_page = 1, 900

    def start_requests(self):
        urls = [
            'https://dom.ria.com/prodazha-kvartir/',
        ]
        for url in urls:
            yield scrapy.Request(url, self.parse)

    def parse(self, response):

        short_info_about_all_appartments_at_page = json.loads(
            re.findall("ld\+json\">(.+?)</script>", response.body.decode("utf-8"), re.S)[0][1:-1].rsplit(',{"@context"',
                                                                                                         maxsplit=1)[0])

        for i in range(len(short_info_about_all_appartments_at_page['@graph'])):
            url_page = short_info_about_all_appartments_at_page['@graph'][i]['@id']
            print(f'------------parsed {self.current_page} pages------------')
            yield scrapy.Request(url_page, self.parse_page)

        next_page = response.xpath(
            "//div[@id='pagination']//span[@class='page-item next text-r']//a[@class='page-link']/@href").extract_first()
        if not (next_page is None) and self.current_page <= self.max_page:
            self.current_page += 1
            yield scrapy.Request(self.domain + next_page, self.parse)

    def parse_page(self, response):
        item = DomRiaItem()
        item['title'] = response.xpath("//div[@class='finalPage']/h1/text()").extract_first()
        item['price_uah'] = int(
            response.xpath("//span[@class='grey size13']/span/text()").extract_first()[:-4].replace(' ', ''))
        item['price_usd'] = int(response.xpath("//span[@class='price']/text()").extract_first()[:-2].replace(' ', ''))
        item['image_urls'] = ['https://cdn.riastatic.com/photosnew/dom/photo/' + link + 'xg.jpg' for link in
                              re.findall("beautifulUrl\":\"dom\\\\u002Fphoto\\\\u002F(.+?).jpg",
                                         response.body.decode("utf-8"), re.S)]
        item['description'] = response.xpath(
            "//div[@class='row finalPage']//main[@class='span8']//div[@id='descriptionBlock']/text()").extract()
        item['street'] = response.xpath("//span[@itemprop='title']/text()").extract()[-1]
        region = re.findall("district_name\":\"(.+?)\"", response.body.decode("utf-8"), re.S)
        item['district_name'] = None if not region else region[0]
        item['square_total'] = float(re.findall("214\":(.+?),", response.body.decode("utf-8"), re.S)[0])
        sq_l = re.findall("216\":(.+?),", response.body.decode("utf-8"), re.S)
        item['square_living'] = None if not sq_l else float(sq_l[0])
        sq_k = re.findall("218\":(.+?),", response.body.decode("utf-8"), re.S)
        item['square_kitchen'] = None if not sq_k else float(sq_k[0])
        item['number_of_rooms'] = int(
            response.xpath("//div[@id='description']//li[1]//div[@class='indent']/text()").extract_first().strip())

        floor = response.xpath("//div[@id='description']//li[2]//div[@class='indent']/text()").extract_first().strip()
        if floor == 'цокольный':
            item['floor'] = -1
        else:
            item['floor'] = int(floor)

        init_state = \
        response.xpath('//script[contains(.,"window.__INITIAL_STATE")]/text()').extract_first()[25:].split('};')[0] + '}'
        main_info = json.loads(init_state)
        item['item_id'] = main_info['dataForFinalPage']['realty']['realty_id']
        item['page_url'] = "https://dom.ria.com/ru/" + main_info['dataForFinalPage']['realty']['beautiful_url']
        params = main_info.get('dataForFinalPage', None).get('realty', None).get('mainCharacteristics', None).get(
            'chars', None)
        item['number_of_floors'] = None
        item['type_of_sentence'] = None
        item['walls_material'] = None
        item['year_of_construction'] = None
        item['heating'] = None

        if params is not None:
            for info in params:
                name = info['name']
                if name == 'Этажность':
                    item['number_of_floors'] = info['value']
                elif name == 'Тип предложения':
                    item['type_of_sentence'] = info['value']
                elif name == 'Тип стен':
                    item['walls_material'] = info['value']
                elif name == 'Год постройки':
                    item['year_of_construction'] = info['value']
                elif name == 'Отопление':
                    item['heating'] = info['value']
        longtitude = main_info.get('dataForFinalPage', None).get('realty', None).get('longitude', None)
        item['longitude'] = None if not longtitude else longtitude
        latitude = main_info.get('dataForFinalPage', None).get('realty', None).get('latitude', None)
        item['latitude'] = None if not latitude else latitude
        item['publishing_date'] = main_info.get('dataForFinalPage', None).get('realty', None).get('publishing_date',
                                                                                                  None)

        proven = response.xpath(
            "//aside[@class='span4']//ul[@class='unstyle']//div[@class='ml-30']/span/text()").extract()
        item['price_verified'] = ('Перевірена ціна' in proven)
        item['apartment_verified'] = ('Перевірена квартира' in proven)

        return item
