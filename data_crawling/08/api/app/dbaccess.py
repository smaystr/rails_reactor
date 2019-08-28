import psycopg2
import json
import pathlib

CONFIGURATION_FILE = pathlib.Path.home().joinpath('summer-19', 'homeworks', 'sergey_milantiev', 'hw_7', 'secrets',
                                                  'dontdothis', 'please', 'ok', 'secrets.json')


def open_db():
    with open(CONFIGURATION_FILE, 'r') as file:
        config = json.load(file)
    connection = psycopg2.connect(
        host=config['HOST'],
        user=config['USER'],
        password=config['PASSWORD'],
        dbname=config['DATABASE']
    )
    return connection


def count_items(connection):
    cur = connection.cursor()
    cur.execute('''
                    SELECT count(*) FROM items;
                ''')
    amount_of_items = cur.fetchall()
    connection.commit()
    return amount_of_items[0][0]


def make_query(connection, query):
    cur = connection.cursor()
    cur.execute(query)
    response = cur.fetchall()
    connection.commit()
    return response


def select_apartments(connection, limit, offset):
    cur = connection.cursor()
    cur.execute(f'''
                    SELECT * FROM items order by publishing_date DESC limit {limit} offset {offset};
                ''')
    response = cur.fetchall()
    connection.commit()
    cur.close()
    connection.close()
    return response


def main_statistics(connection, column):
    cur = connection.cursor()
    cur.execute(f'''
                    select avg({column}), max({column}), min({column}), stddev({column})  from items;
                ''')
    response = cur.fetchone()
    connection.commit()
    return {'average': response[0],
            'maximum': response[1],
            'minimum': response[2],
            'std': response[3]}
