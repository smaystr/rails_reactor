import datetime
import re
import json
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from scraping_model import Apartment


class ApartmentScraper:
    def __init__(self):
        self.json_re = re.compile(r'window.__INITIAL_STATE__={.*};', flags=re.DOTALL)
        self.mi_mapping = {
            'Комнат': 'rooms_count',
            'Этаж': 'floor',
            'Тип предложения': 'seller',
            'Тип стен': 'wall_type',
            'Год постройки': 'construction_year',
            'Отопление': 'heating'
        }
        self.json_keys = ['realty_id', 'street_name', 'city_name', 'district_name', 'longitude', 'latitude',
                          'total_square_meters', 'living_square_meters', 'kitchen_square_meters',
                          'panoramas', 'photos', 'secondaryParams']
        self.image_base = 'https://cdn.riastatic.com/photos/'
        self.image_name_mapping = {
            'panoramas': 'img',
            'photos': 'file'
        }
        self.image_size_mapping = {
            'panoramas': 'l',
            'photos': 'fl'
        }
        self.secondary_mapping = {
            'характеристика здания': 'building_condition',
            'подъезд': 'building',
            'вода': 'water',
            'до центра города': 'dist_to_center',
            'школа': 'dist_to_school',
            'детский сад': 'dist_to_kindergarten',
            'автовокзал': 'dist_to_bus_terminal',
            'жд вокзал': 'dist_to_railway_station',
            'аэропорт': 'dist_to_airport',
            'больница': 'dist_to_hospital',
            'магазин / рынок': 'dist_to_shop',
            'ближайшая парковка': 'dist_to_parking',
            'ближайшая зона отдыха': 'dist_to_rest_area'
        }
        self.RUS_MONTHS = ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']

    def convert_to_date(self, date_str):
        day = int(date_str.split(' ')[0])
        month = date_str.split(' ')[1]
        month = (self.RUS_MONTHS.index(month)) + 1
        return datetime.date(year=2019, month=month, day=day)

    def parse_photo(self, item, img_type):
        splitted_url = item[self.image_name_mapping.get(img_type)].split('.jpg')
        return urljoin(self.image_base,
                       (splitted_url[0] + self.image_size_mapping.get(img_type) + '.jpg' + splitted_url[1]))

    @staticmethod
    def convert_secondary_params(param_name, item_arr, apartment):
        if param_name not in ['water', 'building']:
            items = []
            for i, item in enumerate(item_arr):
                if ':' in item:
                    items += item.split(':')
            setattr(apartment, param_name, ''.join(items[1::2]).strip())
        elif param_name == 'building':
            elevators_count = 0
            for item in item_arr:
                if 'лифты' in item:
                    elevators_count += int(item.split(':')[1].strip())
            setattr(apartment, 'elevators_count', elevators_count)
        else:
            setattr(apartment, param_name, ''.join(item_arr))

    def scrap_apartment(self, apartment_url):
        apartment = Apartment()
        response = requests.get(apartment_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'lxml')
            apartment.title = soup.find('div', class_='finalPage').find('h1').text
            apartment.creation_date = self.convert_to_date(soup.find('ul', class_='greyLight size13 unstyle')
                                                           .find('li', class_='mt-15').find('b').text.strip())
            apartment.price_usd = int(
                soup.find('span', class_='price').text.strip().replace('$', '').replace(' ', ''))
            apartment.price_uah = int(soup.find('span', class_='grey size13').text.split('•')[0].strip()
                                      .replace('грн', '').replace(' ', ''))
            apartment.description = soup.find('div', id='descriptionBlock').text.strip()
            apartment_main_info = soup.find_all('li', class_='mt-15 boxed v-top')
            for element in apartment_main_info:
                title_text = element.find('div', class_='label grey').text.strip()
                if title_text in self.mi_mapping.keys():
                    value_text = element.find('div', class_='indent').text.strip()
                    setattr(apartment, self.mi_mapping.get(title_text), value_text)
            apartment.is_verified_price = soup.find('span', text='Перевірена ціна') is not None
            apartment.is_verified_flat = soup.find('span', text='Перевірена квартира') is not None
            initial_state = self.json_re.search(response.text)[0].replace('window.__INITIAL_STATE__=', '').replace(';', '')
            realty_json = json.loads(initial_state)['dataForFinalPage']['realty']
            realty_keys = realty_json.keys()
            for key in self.json_keys:
                if key in realty_keys:
                    if realty_json[key] != '':
                        if key == 'panoramas' or key == 'photos':
                            setattr(apartment, 'photos', getattr(apartment, 'photos') +
                                    [self.parse_photo(item, key) for item in realty_json[key]])
                        elif key == 'secondaryParams':
                            if realty_json[key]:
                                for param in realty_json[key]:
                                    if param['groupName'] in self.secondary_mapping.keys():
                                        self.convert_secondary_params(self.secondary_mapping.get(param['groupName']),
                                                                      param['items'], apartment)
                        else:
                            setattr(apartment, key, realty_json[key])
            return apartment
        else:
            print(f'ups got {response.status_code}')
            return None
