import requests
from bs4 import BeautifulSoup
import urllib
import json
import re
import pandas as pd
import psycopg2

MAIN_URL  = 'https://dom.ria.com'
PHOTO_URL = 'https://cdn.riastatic.com/photosnew/'
df = pd.DataFrame(columns = ['page_url', 'title', 'price_UAH', 'price_USD', 'images_urls', 'description', 'street', 'city', 'total_area', 'living_area','kitchen_area', 'number_of_rooms',
                            'floor', 'total_number_of_floors', 'year_of_construction', 'heating_type', 'type_of_proposal', 'walls_type', 'verified', 'latitude', 'longitude', 
                             'date_announcement_created'])

def make_soup(url):
    try:
        page = requests.get(url)
        soupdata = BeautifulSoup(page.text, 'html.parser')
    except: 
        print('Connection error')
    else:
        return soupdata

def delete_values(connector,cursor):
    cursor.execute('TRUNCATE flat_info, announcement_info;')
    connector.commit()

def connect_to_database():
    connection = psycopg2.connect(
        host='localhost',
        database='flats_data',
        user='bogdanivanyuk',
        port='5431'
    )
    return (connection, connection.cursor())

def insert_values_to_database(cursor, connector, flat,flat_id, page_url):
    insert_query_flat = '''INSERT INTO flat_info (flat_id, street_name, city_name, total_area, living_area, kitchen_area, floor, total_number_of_floors, 
    number_of_rooms, year_of_construction, heating_type, walls_type, latitude, longitude) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);'''
    insert_data_flat = (flat_id, flat.street_name, flat.city_name, flat.total_square_meters, flat.living_square_meters, 
                       flat.kitchen_square_meters, flat.floor, flat.max_floor, flat.number_of_rooms, flat.year_of_construction,
                       flat.heating_type, flat.wall_type, flat.latitude, flat.longitude)
    cursor.execute(insert_query_flat, insert_data_flat)
    
    insert_query_announcement = '''INSERT INTO announcement_info (flat_id, page_url, title, price_UAH, price_USD, image_urls, description, type_of_proposal, 
    verified, date_created) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);'''
    insert_data_announcement = (flat_id, page_url, flat.announcement_title, flat.price_UAH, flat.price_USD, flat.photos, flat.announcement_description,
                               flat.type_of_proposal, flat.verified, flat.date_created)
    cursor.execute(insert_query_announcement, insert_data_announcement)
    connector.commit()

def parse_pages():
    pages_parsed = 0
    max_number_of_parsed_pages = 30000
    counter = 1
    connection, cursor = connect_to_database()
    delete_values(connection, cursor)
    while pages_parsed < max_number_of_parsed_pages:
        page_to_parse = urllib.parse.urljoin(MAIN_URL, '/prodazha-kvartir/?page=' + str(counter))
        print('========================='+page_to_parse+'=========================')
        parsed_page = make_soup(page_to_parse).findAll('a', attrs={'class':'realtyPhoto'})
        
        for flat in parsed_page:
            page_url = urllib.parse.urljoin(MAIN_URL, flat['href'])
            parsed_flat = make_soup(urllib.parse.urljoin(MAIN_URL, flat['href']))
            if parsed_flat is None:
                continue
            init_state_data = json.loads(re.search(r"window.__INITIAL_STATE__\s*=\s*({.*});", parsed_flat.text).group(1))
            try:
                flat.announcement_title = init_state_data['dataForFinalPage']['tagH']
                flat.price_UAH = int(init_state_data['dataForFinalPage']['realty']['priceArr']['3'].replace(' ', ''))
                flat.price_USD = int(init_state_data['dataForFinalPage']['realty']['priceArr']['1'].replace(' ', ''))
                flat.photos = [urllib.parse.urljoin(PHOTO_URL, d['beautifulUrl']) for d in init_state_data['dataForFinalPage']['realty']['photos']]
                flat.announcement_description = init_state_data['dataForFinalPage']['realty']['description']
                flat.street_name = init_state_data['dataForFinalPage']['realty']['street_name'] if 'street_name' in init_state_data['dataForFinalPage']['realty'].keys() else ''
                flat.city_name = init_state_data['dataForFinalPage']['realty']['city_name']
                flat.total_square_meters = float(init_state_data['dataForFinalPage']['realty']['mainCharacteristics']['baseInfo']['p2']['value'].split()[0])# if init_state_data['dataForFinalPage']['realty']['mainCharacteristics']['baseInfo']['p2']['value'].split()[0] !='-' else 0.0
                flat.living_square_meters = float(init_state_data['dataForFinalPage']['realty']['living_square_meters']) if 'living_square_meters' in init_state_data['dataForFinalPage']['realty'].keys() else 0.0
                flat.kitchen_square_meters = float(init_state_data['dataForFinalPage']['realty']['kitchen_square_meters']) if 'kitchen_square_meters' in init_state_data['dataForFinalPage']['realty'].keys() else 0.0
                flat.number_of_rooms = int(init_state_data['dataForFinalPage']['realty']['mainCharacteristics']['baseInfo']['p1']['value'])
                flat.floor = int(init_state_data['dataForFinalPage']['realty']['floor'])
                flat.max_floor = int(init_state_data['dataForFinalPage']['realty']['floors_count'])
                flat.year_of_construction = next(item['value'] for item in init_state_data['dataForFinalPage']['realty']['mainCharacteristics']['chars'] if item['name'] == 'Год постройки') if any(a['name'] == 'Год постройки' for a in init_state_data['dataForFinalPage']['realty']['mainCharacteristics']['chars']) else ''
                flat.heating_type = next(item['value'] for item in init_state_data['dataForFinalPage']['realty']['mainCharacteristics']['chars'] if item["name"] == "Отопление") if any(a['name'] == 'Отопление' for a in init_state_data['dataForFinalPage']['realty']['mainCharacteristics']['chars']) else ''
                flat.type_of_proposal = next(item['value'] for item in init_state_data['dataForFinalPage']['realty']['mainCharacteristics']['chars'] if item['name'] == 'Тип предложения') if any(a['name'] == 'Тип предложения' for a in init_state_data['dataForFinalPage']['realty']['mainCharacteristics']['chars']) else ''
                flat.wall_type = init_state_data['dataForFinalPage']['realty']['wall_type']
                flat.verified = True if 'inspected 'in init_state_data['dataForFinalPage']['realty'].keys() else False
                flat.latitude = float(init_state_data['dataForFinalPage']['realty']['latitude']) if ('latitude' in init_state_data['dataForFinalPage']['realty'].keys() and init_state_data['dataForFinalPage']['realty']['latitude'] != '') else 0.0
                flat.longitude = float(init_state_data['dataForFinalPage']['realty']['longitude']) if ('latitude' in init_state_data['dataForFinalPage']['realty'].keys() and init_state_data['dataForFinalPage']['realty']['longitude'] != '') else 0.0
                flat.date_created = init_state_data['dataForFinalPage']['realty']['created_at']
            
                df.loc[pages_parsed] = [page_url, flat.announcement_title, flat.price_UAH, flat.price_USD, flat.photos, flat.announcement_description, flat.street_name, flat.city_name, flat.total_square_meters, flat.living_square_meters, flat.kitchen_square_meters, flat.number_of_rooms,  flat.floor, flat.max_floor, flat.year_of_construction,flat.heating_type, flat.type_of_proposal, flat.wall_type, flat.verified, flat.latitude,  flat.longitude,  flat.date_created]
                insert_values_to_database(cursor, connection, flat,pages_parsed, page_url)
                pages_parsed += 1
            except:
                continue
            if pages_parsed % 50 == 0:
                print(str(pages_parsed) + ' pages parsed')
                print(df.shape)
        counter += 1
    df.to_csv('full_data.csv', index=False)
    cursor.close()
    connection.close()

if __name__ == '__main__':
	parse_pages()
