import re
from requests.compat import urljoin
from scrapy import Spider
from items import ApartmentItem


class ApartmentsSpider(Spider):
    name = 'apartments'
    start_urls = [
        'https://dom.ria.com/prodazha-kvartir/',
    ]

    def parse(self, response):
        for apartment in response.css('a.realtyPhoto::attr(href)').getall():
            url = urljoin('https://dom.ria.com/', apartment)
            yield response.follow(url, self.parse_apartment)
        next_page = response.css('span.page-item.next.text-r a::attr(href)').get()
        if next_page is not None:
            next_page = urljoin('https://dom.ria.com/', next_page)
            yield response.follow(next_page, self.parse)

    def parse_apartment(self, response):
        title = response.css('h1::text').get()
        title = title.replace("'", "''")
        price_uah = response.css('span.grey.size13 span::text').get()[:-4]
        price_usd = response.css('span.price::text').get()[:-3]
        price_uah = int(price_uah.replace(' ', ''))
        price_usd = int(price_usd.replace(' ', ''))
        description = response.css('#descriptionBlock::text').get()
        if description:
            description = description.replace("'", "''")
        else:
            description = 'NULL'
        street = response.css('#final_page__breadcrumbs_container>li>span::text').get()
        if street:
            street = street.replace("'", "''")
        else:
            street = 'NULL'
        region = response.css('#final_page__breadcrumbs_container>li>a>span::text').getall()[-2]
        area = response.css('div.indent::text').re('\s{2,}(.+) м')[0].split(' • ')
        area_total = float(area[0])
        if len(area) > 1:
            area_living = float(area[1])
        else:
            area_living = None
        rooms = response.xpath('//li/div[text()[contains(.,"Комнат")]]')[1].xpath('..').css('div.indent::text').get()
        rooms = int(rooms.split()[0])
        floor = response.xpath('//li/div[text()[contains(.,"Этаж")]]')[0].xpath('..').css('div.indent::text').get()
        floor = floor.split()[0]
        if floor == 'цокольный':
            floor = 0
        else:
            floor = int(floor)

        year = response.xpath('//li/div[text()[contains(.,"Год")]]')
        if year:
            year = year[0].xpath('..').css('div.indent::text').get()
            year = ' '.join(year.split())
        else:
            year = 'NULL'
        heating = response.xpath('//div[text()[contains(.,"Отопление")]]')
        if heating:
            heating = heating[0].xpath('..').css('.indent::text').get()
            heating = heating.split()[0]
        else:
            heating = 'NULL'
        seller_id = response.css('script').re('user_id":(\d+)')[0]
        seller_url = f'https://dom.ria.com/ru/realtor-{seller_id}.html'
        seller_name = response.css('script').re('{"owner":{"name":"(\w+ ?\w+)"}')
        if seller_name:
            seller_name = seller_name[0]
            seller_name = seller_name.replace("'", "''")
        else:
            seller_name = 'NULL'
        walls = response.xpath('//div[text()[contains(.,"Тип стен")]]')[0].xpath('..').css('.indent::text').get()
        walls = walls.split()[0]
        if response.css('span.blue').re('Перевірена ціна'):
            verified_price = True
        else:
            verified_price = False
        if response.css('span.blue').re('Перевірена квартира'):
            verified_apartment = True
        else:
            verified_apartment = False
        latitude = response.css('script').re('latitude":(\d+.\d+)')
        longitude = response.css('script').re('longitude":(\d+.\d+)')
        if latitude:
            latitude = latitude[0]
            longitude = longitude[0]
        else:
            latitude = None
            longitude = None
        photos = response.css('script').re('file":"[a-z1-9\\\\u002F-]+\\\\u002F(\d+)')
        panoramas = response.css('script').re('img":"[a-z1-9\\\\u002F-]+\\\\u002F(\d+)')
        apartment = re.search('realty-((\w+-)+)\d+.html', str(response)).group(1)[:-1]
        url = 'https://cdn.riastatic.com/photosnew/dom/'
        photos = list(map(lambda ph: url + 'photo/' + apartment + '__' + ph + 'fl.jpg', photos))
        panoramas = list(map(lambda pan: url + 'panoramas/' + apartment + '__' + pan + 'l.jpg', panoramas))
        publication_date = response.css('script').re('publishing_date":"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"')[0]
        # views = int(response.xpath('//li[text()[contains(.,"Просмотров")]]')[0].xpath('/b/text()').get())
        # doesn't seem to work (always returns '-')
        apartment_item = ApartmentItem(title=title, price_uah=price_uah, price_usd=price_usd, photos=photos,
                                       panoramas=panoramas, description=description, street=street, region=region,
                                       area_total=area_total, area_living=area_living, rooms=rooms, floor=floor,
                                       year=year, heating=heating, walls=walls, seller_name=seller_name,
                                       seller_url=seller_url, verified_price=verified_price,
                                       verified_apartment=verified_apartment, latitude=latitude, longitude=longitude,
                                       publication_date=publication_date)
        yield apartment_item
        # TODO:add more geolocation; add optional;
