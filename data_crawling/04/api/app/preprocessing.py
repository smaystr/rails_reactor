import pandas as pd
import numpy as np
import json
from langdetect import detect

ALL_COLUMNS = ['total_square_meters', 'living_square_meters', 'kitchen_square_meters',
    'rooms_count', 'floor', 'inspected', 'desc_len', 'Month', 'Day', 'state_name_Винницкая',
    'state_name_Волынская', 'state_name_Днепропетровская', 'state_name_Житомирская',
    'state_name_Закарпатская', 'state_name_Запорожская', 'state_name_Ивано-Франковская',
    'state_name_Киевская', 'state_name_Львовская', 'state_name_Николаевская', 'state_name_Одесская',
    'state_name_Полтавская', 'state_name_Ровенская', 'state_name_Тернопольская', 'state_name_Харьковская',
    'state_name_Херсонская', 'state_name_Хмельницкая', 'state_name_Черкасская', 'state_name_Черновицкая',
    'wall_type_other', 'wall_type_блочно-кирпичный', 'wall_type_газобетон', 'wall_type_газоблок',
    'wall_type_керамзитобетон', 'wall_type_кирпич', 'wall_type_монолит', 'wall_type_монолитно-блочный',
    'wall_type_монолитно-каркасный', 'wall_type_монолитно-кирпичный', 'wall_type_панель', 'wall_type_пеноблок',
    'wall_type_ракушечник (ракушняк)', 'heating_без отопления', 'heating_индивидуальное', 'heating_централизованное',
    'seller_от застройщика', 'seller_от посредника', 'seller_от представителя застройщика',
    'seller_от представителя хозяина (без комиссионных)',
    'seller_от собственника', 'water_other', 'water_unknown', 'water_централизованное (водопровод)',
    'building_condition_unknown', 'building_condition_нормальное', 'building_condition_отличное',
    'building_condition_требует', 'building_condition_удовлетворительное', 'building_condition_хорошее',
    'street_type_other', 'street_type_бульвар', 'street_type_дорога', 'street_type_майдан',
    'street_type_переулок', 'street_type_плато', 'street_type_проезд', 'street_type_проспект', 'street_type_улица',
    'street_type_шоссе', 'desc_lang_ru', 'desc_lang_uk', 'desc_lang_unknown']

NUMERICAL_FEATURES = ['total_square_meters', 'living_square_meters',
    'kitchen_square_meters', 'rooms_count', 'floor', 'inspected',
    'desc_len', 'Month', 'Day']

CATEGORICAL_FEATURES = ['state_name', 'wall_type', 'heating', 'seller', 'water',
    'building_condition', 'street_type', 'desc']

WALL_TYPES = ['кирпич', 'газоблок', 'газобетон', 'панель', 'монолитно-каркасный',
    'монолит', 'керамзитобетон', 'монолитно-блочный', 'монолитно-кирпичный',
    'ракушечник (ракушняк)', 'пеноблок', 'блочно-кирпичный']

HEATING_TYPES = ['централизованное', 'индивидуальное', 'без отопления']

SELLER_TYPES = ['от собственника', 'от посредника', 'от представителя хозяина (без комиссионных)',
    'от представителя застройщика', 'от застройщика']

WATER_TYPES = ['централизованное (водопровод)', 'other']

BUILDING_CONDITIONS = ['хорошее', 'отличное', 'нормальное', 'требует', 'удовлетворительное']

STREET_TYPES = ['проспект', 'улица', 'шоссе', 'переулок', 'бульвар', 'дорога', 'плато', 'майдан','проезд']


def preprocess_row(row):
    params = json.loads(row.replace('\u0027', '\u0022'))
    features = params['features']

    data = pd.DataFrame(np.zeros((1, len(ALL_COLUMNS))), columns=ALL_COLUMNS)
    for prop in features:
        if prop in NUMERICAL_FEATURES:
            data.at[0, prop] = features[prop]
        elif prop in CATEGORICAL_FEATURES:
            if prop == 'state_name':
                fill_state_name(data, features[prop])
            elif prop == 'wall_type':
                fill_prop(data, prop, features[prop], 'other', WALL_TYPES)
            elif prop == 'heating':
                fill_prop(data, prop, features[prop], 'без отопления', HEATING_TYPES)
            elif prop == 'seller':
                fill_prop(data, prop, features[prop], 'от собственника', SELLER_TYPES)
            elif prop == 'water':
                fill_prop(data, prop, features[prop], 'unknown', WATER_TYPES)
            elif prop == 'building_condition':
                fill_prop(data, prop, features[prop], 'unknown', BUILDING_CONDITIONS)
            elif prop == 'street_type':
                fill_prop(data, prop, features[prop], 'other', STREET_TYPES)
            elif prop == 'desc':
                fill_desc(data, features[prop])

    return params['model'], data


def fill_prop(df, prop_name, prop_value, null_value, prop_types):
    if prop_value.lower() in prop_types:
        df.at[0, prop_name + '_' + prop_value.lower()] = 1
    else:
        df.at[0, prop_name + '_' + null_value] = 1


def fill_state_name(df, state_name):
    if 'state_name_' + state_name in ALL_COLUMNS:
        df.at[0, 'state_name_' + state_name] = 1


def fill_desc(df, desc):
    df.at[0, 'desc_len'] = len(desc)
    lang = detect(desc.lower())
    if (lang == 'uk') | (lang == 'ru'):
        df.at[0, 'desc_lang_' + lang] = 1
    else:
        df.at[0, 'desc_lang_unknown'] = 1
