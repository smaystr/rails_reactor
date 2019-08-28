import datetime

MAIN_PROPERTIES = {
    'Комнат': 'rooms_count',
    'Этаж': 'floor',
    'Тип предложения': 'seller',
    'Тип стен': 'wall_type',
    'Отопление': 'heating',
    'Год постройки': 'construction_year'
}

ADDITIONAL_PROPERTIES = {
    'характеристика здания': 'building_condition',
    'вода': 'water',
    'до центра города': 'dist_to_center',
    'школа': 'dist_to_school',
    'детский сад': 'dist_to_kindergarten',
    'больница': 'dist_to_hospital',
    'автовокзал': 'dist_to_bus_station',
    'жд вокзал': 'dist_to_railway_station',
    'аэропорт': 'dist_to_airport'
}

JSON_PROPERTIES = [
    'street_name',
    'state_name',
    'total_square_meters',
    'living_square_meters',
    'kitchen_square_meters',
    'inspected',
    'latitude',
    'longitude'
]

RUS_MONTHS = ['янв','фев','мар','апр','май','июн','июл','авг','сен','окт','ноя','дек']
UK_MONTHS = ['січ', 'лют', 'бер', 'квіт', 'тра','чер', 'лип', 'сер', 'вер', 'жовт', 'лист', 'груд']

def date_converter(date_str):
    day = int(date_str.split(' ')[0])
    month = date_str.split(' ')[1]
    month = (UK_MONTHS.index(month) if month in UK_MONTHS else RUS_MONTHS.index(month)) + 1

    return datetime.date(year=2019, month=month, day=day)
