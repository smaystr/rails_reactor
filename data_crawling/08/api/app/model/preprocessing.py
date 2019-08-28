import pandas as pd
import torch


def preprop_squares(db):
    blocks = [(0, 43), (43, 55), (55, 68), (68, 85), (85, db['square_total'].max() + 1)]

    mapped_living = dict()
    mapped_kitchen = dict()

    # print('square_living:')
    for l, r in blocks:
        counted = db[(db['square_total'] >= l) & (db['square_total'] < r)]['square_living'].count()
        meaned = db[(db['square_total'] >= l) & (db['square_total'] < r)]['square_living'].mean()
        mapped_living[(l, r)] = meaned
        # print(f'{l} -- {r} \t: counnt - {counted}, mean - {meaned}')

    # print('square_kitchen:')
    for l, r in blocks:
        counted = db[(db['square_total'] >= l) & (db['square_total'] < r)]['square_kitchen'].count()
        meaned = db[(db['square_total'] >= l) & (db['square_total'] < r)]['square_kitchen'].mean()
        mapped_kitchen[(l, r)] = meaned
        # print(f'{l} -- {r} \t: counnt - {counted}, mean - {meaned}')

    for l, r in blocks:
        mask = (db['square_total'] >= l) & (db['square_total'] < r)
        db.loc[mask, 'square_living'] = db.loc[mask, 'square_living'].fillna(mapped_living[(l, r)])
        db.loc[mask, 'square_kitchen'] = db.loc[mask, 'square_kitchen'].fillna(mapped_kitchen[(l, r)])

    return db


def preprop_streets(x):
    if x == ['Саперное Поле улица', 'Леси Украинки бульвар', 'Ивана Кудри улица', 'Саксаганского улица',
             'Оболонский проспект', 'Анри Барбюса улица', 'Героев Сталинграда проспект', 'Предславинская улица',
             'Голосеевский проспект', 'Патриса Лумумбы улица']:
        return 5
    elif x in ['Щорса улица', 'Большая Арнаутская улица', 'Маршала Тимошенко улица', 'Французский бульвар',
               'Большая Васильковская улица', 'Жилянская улица', 'Златоустовская улица', 'Литературная улица',
               'Антоновича улица', 'Глубочицкая улица' 'Французский бул. Пролетарский бул.', 'Демеевская улица',
               'Петра Григоренко проспект', 'Маршала Говорова улица', 'Литературная']:
        return 4
    elif x in ['Генуэзская', 'Днепровская набережная', 'Литературная', 'Богдановская улица', 'Вышгородская улица',
               'Гагаринское плато', 'Сикорского улица', 'Ломоносова улица', 'Липковского Василия Митрополита ул.',
               'Анатолія Бортняка улица', 'Урловская улица', 'Митрополита Василия Липковского улица',
               'Академика Филатова улица', 'Киев', 'Екатерининская улица', 'Николая  Бажана проспект',
               'Анны Ахматовой улица', 'Преображенская улица', 'Драгоманова улица', 'Заречная улица',
               'Канатная улица', 'Олени Пчілки улица', 'Генуэзская улица']:
        return 3
    elif x in ['Победы проспект', 'Победы пр-т', 'Большая арнаутская Чкалова', 'Фонтанская дор. Перекопской Дивизии',
               'Армейская улица', 'Академика Глушкова проспект', 'Панаса Мирного улица', 'Науки проспект',
               'Шевченко проспект', 'Ивана Франко улица', 'Соломенская улица', 'Метрологическая улица',
               'Онуфрия Трутенко улица', 'Светлый переулок', 'Академика Павлова ул.', 'Педагогическая',
               'Маршала Малиновского улица', 'Ясиноватский переулок', 'Педагогическая улица', 'Зодчих улица',
               'Костанди улица', 'Тургеневская улица', 'Фонтанская дорога', 'Среднефонтанская улица', 'Каманина',
               'Леся Курбаса проспект', 'Правды проспект', 'Чавдар Елизаветы ул.', 'Каманина улица',
               'Пирогова улица', 'Харьковское шоссе', 'Бориса Гмыри улица', 'Пишоновская улица',
               'Люстдорфская дор. Черноморская дор.', 'Жаботинского улица', 'Закревского Николая ул.',
               'Михайловская улица', 'Гмыри Бориса ул.', 'Академика Вильямса улица', 'Ляли Ратушной улица',
               'Академика Королева улица', 'Королева ак.', 'Вильямса ак.', 'Князей Кориатовичей улица']:
        return 2
    elif x in ['Одесса', 'Келецкая улица',
               'Люстдорфская дорога', 'Глушкова Академика пр-т',
               'Вячеслава Черновола улица', 'Бассейная улица', 'Миколаївська улица',
               'Черновола Вячеслава улица', 'Леваневского улица', 'Драгоманова ул.',
               'Академика Глушко проспект', 'Гагарина пр.', 'Перлинна улица',
               'Академика Заболотного улица', 'Революционная улица',
               'Князів Коріатовичів улица', 'Небесної Сотні проспект',
               'Салтовское шоссе', 'Киево-Святошинский', 'Радужный микрорайон',
               'Архитекторская улица', 'Дача Ковалевского улица',
               '50-летия Победы улица', 'Добровольского пр.', 'Радужная улица',
               'Беговая улица', 'Маршала Жукова проспект', 'Ильфа и Петрова улица',
               'Радужний массив', 'Чехова улица', 'Свердлова улица', 'Киевская улица',
               'Жемчужная', 'Шевченко улица', 'Зерновая ул.', '600-летия улица',
               'Литвиненко улица', 'Барское шоссе', 'Варненская улица', 'Радужный м-н',
               'Генерала Бочарова улица', 'Бассейная', 'Академика Сахарова улица',
               'Мира проспект', 'Космонавтов улица', 'Фрунзе улица',
               'Андрея Первозванного улица', 'Космонавтов проспект',
               'Василия Порика улица', 'Елизаветинская ул.', 'Юности проспект',
               'Університетська улица', 'Красноармейская улица', 'Трудовая улица',
               'Стрелецкая улица', 'Новооскольская улица', 'Заречанская улица',
               'Озерная улица', 'Мичурина улица', 'Мира ул.', 'Немировское шоссе',
               'Покрышкина улица', 'Хмельницкий', 'Старокостянтиновское шоссе',
               'Маяковского улица']:
        return 1
    else:
        return 0


def preprop_city(x):
    x = " ".join(x.split(" ")[-2:])
    if x == 'в Киеве':
        return 2
    elif x in ['в Днепропетровске', 'в Одессе', 'в Житомире', 'в Львове', 'в Харькове']:
        return 1
    else:
        return 0


def preprop_regions(x):
    if x == 'Печерский':
        return 5
    elif x in ['Шевченковский', 'Оболонский', 'Подольский', 'Голосеевский']:
        return 4
    elif x in ['Приморский', 'Днепровский', 'Соломенский', 'Дарницкий', 'Подолье', 'Святошинский', 'Центр']:
        return 3
    elif x in ['Деснянский', 'Таирова', 'Славянка', 'Киевский', 'Салтовка', 'Софиевская Борщаговка', 'Малиновский',
               'Вишенка', 'Ближнее замостье', 'Свердловский массив', 'Суворовский']:
        return 2
    elif x in ['Одесская', 'Ирпень', 'Замостье', 'Выставка', 'Старый город']:
        return 1
    else:
        return 0


def preprop_type_sentence(x):
    if x in ['от посредника', 'от застройщика', 'от представителя застройщика']:
        return x
    elif x in ['от собственника', 'от представителя хозяина (без комиссионных)']:
        return 'от хозяина'
    else:
        return 'от непятно кого'
#         raise Exception( 'I dont understand your type_of_sentence')

def preprop_walls(x):
    if x in ['монолитный железобетон', 'монолит', 'железобетон']:
        return 4
    elif x == 'кирпич':
        return 3
    elif x == 'панель':
        return 2
    elif x in ['керамзитобетон', 'ракушечник (ракушняк)', 'монолитно-каркасный', 'керамический блок', 'пеноблок',
               'газобетон', 'блочно-кирпичный', 'силикатный кирпич', 'монолитно-блочный', 'газоблок',
               'монолитно-кирпичный']:
        return 1
    else:
        return 0


def transform_new_data(params, main_db):
    for k in params.keys():
        params[k] = [params[k]]
    test_df = pd.get_dummies(pd.DataFrame(params),
                             columns=['type_of_sentence', 'heating', 'group_city', 'region_group', 'street_group',
                                      'walls_group'])

    return test_df.reindex(columns=main_db.columns, fill_value=0).drop(columns='price_usd').values[0], \
           test_df.reindex(columns=main_db.columns, fill_value=0)['price_usd'].values


def load_data(X_train, X_test, y_train, y_test, device):
    return torch.tensor(X_train.values, dtype=torch.float32, device=device), \
           torch.tensor(X_test.values, dtype=torch.float32, device=device), \
           torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32, device=device), \
           torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32, device=device)


def load_and_transform_database(connection):
    d = pd.read_sql('select * from items', connection, index_col='item_id')
    images = pd.read_sql('select * from images', connection)
    db = d.join(images.groupby('item_id').count()).rename(columns={'link': 'number_of_images'})

    db['group_city'] = db.title.apply(preprop_city)
    db['region_group'] = db.district_name.apply(preprop_regions)
    db['street_group'] = db.street.apply(preprop_streets)
    db['type_of_sentence'] = db.type_of_sentence.apply(preprop_type_sentence)
    db['walls_group'] = db.walls_material.apply(preprop_walls)
    db['heating'] = db.heating.fillna('странное отопление')
    db = preprop_squares(db)

    db.drop(
        [15993447, 15124093, 15722479, 15665875, 15773833, 15991832, 15806908, 15403487, 15819825, 15821729, 15464728,
         16000612], inplace=True)
    new_db = db.drop(
        columns=['apartment_verified', 'year_of_construction', 'title', 'number_of_images', 'walls_material', 'street',
                 'publishing_date', 'number_of_floors', 'page_url', 'price_uah', 'price_verified', 'description',
                 'district_name', 'latitude', 'longitude'])
    lets_try = pd.get_dummies(new_db,
                              columns=['type_of_sentence', 'heating', 'group_city', 'region_group', 'street_group',
                                       'walls_group'])
    lets_try.to_csv('transformed_db.csv')
    return lets_try
