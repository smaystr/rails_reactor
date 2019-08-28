import psycopg2


def get_connection(host='localhost', database='postgres', user='postgres', password='postgres'):
    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    cur = conn.cursor()
    return conn, cur


def create_apartments():
    try:
        conn, cur = get_connection()
        conn.autocommit = True
        cur.execute("CREATE DATABASE apartments;")
        cur.close()
        conn.close()
    except psycopg2.errors.DuplicateDatabase:
        print('psycopg2.errors.DuplicateDatabase: Database already exists')
        return

    conn, cur = get_connection(database='apartments')
    cur.execute('''CREATE TABLE sellers (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(128),
                        url VARCHAR(128) UNIQUE
                    );''')
    cur.execute('''CREATE TABLE apartments (
                        id SERIAL PRIMARY KEY,
                        title VARCHAR(128),
                        price_uah INT,
                        price_usd INT,
                        description VARCHAR(2048),
                        street VARCHAR(128),
                        region VARCHAR(64),
                        area_total REAL,
                        area_living REAL,
                        rooms INT,
                        floor INT,
                        year VARCHAR(16),
                        heating VARCHAR(16),
                        seller_id INT REFERENCES sellers(id) ON DELETE CASCADE,
                        walls VARCHAR(32),
                        verified_price BOOL,
                        verified_apartment BOOL,
                        latitude REAL,
                        longitude REAL,
                        publication_date TIMESTAMP
                    );''')
    cur.execute('''CREATE TABLE images (
                        apartment_id INT REFERENCES apartments(id) ON DELETE CASCADE,
                        url VARCHAR(256),
                        panorama BOOL DEFAULT FALSE
                    );''')
    cur.close()
    conn.commit()
    conn.close()


def process_pipeline_item(conn, item):
    cur = conn.cursor()
    try:
        cur.execute('''INSERT INTO sellers (name, url)
                        VALUES (%s, %s)
                        ON CONFLICT (url) DO UPDATE SET name = sellers.name
                        RETURNING id
                        ;''', (item['seller_name'], item['seller_url']))
        seller_id = cur.fetchone()[0]
        cur.execute(f'''INSERT INTO apartments(title, price_uah, price_usd, description, street, region, area_total,
                        area_living, rooms, floor, year, heating, seller_id, walls, verified_price, verified_apartment,
                        latitude, longitude, publication_date)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;''', (item['title'], item['price_uah'], item['price_usd'],
                                           item['description'], item['street'], item['region'],
                                           item['area_total'], item['area_living'], item['rooms'],
                                           item['floor'], item['year'], item['heating'], seller_id,
                                           item['walls'], item['verified_price'], item['verified_apartment'],
                                           item['latitude'], item['longitude'], item['publication_date']))
        apartment_id = cur.fetchone()[0]
        for i in item['photos']:
            cur.execute('''INSERT INTO images (apartment_id, url)
                                VALUES (%s, %s)
                                ;''', (apartment_id, i))
        for i in item['panoramas']:
            cur.execute('''INSERT INTO images (apartment_id, url, panorama)
                                VALUES (%s, %s, %s)
                                ;''', (apartment_id, i, True))
        conn.commit()
        return 'process ok'
    except psycopg2.errors.InFailedSqlTransaction:
        return 'psycopg error'


def get_records(limit=10, offset=0):
    conn, cur = get_connection(database='apartments')
    cur.execute('SELECT * FROM apartments ORDER BY publication_date DESC LIMIT %s OFFSET %s', (limit, offset))
    records = cur.fetchall()

    def make_record(apartment):
        cur.execute('SELECT url, panorama FROM images WHERE apartment_id = %s', (apartment[0],))
        images = cur.fetchall()
        images = [{'url': i[0], 'panorama': i[1]} for i in images]
        cur.execute('SELECT name, url FROM sellers WHERE id = %s', (apartment[13],))
        seller_name, seller_url = cur.fetchone()
        record = list(apartment)
        del record[0]
        record[12] = seller_name
        record.insert(13, seller_url)
        record.insert(0, record.pop(-1))
        record.append(images)
        return record
    records = [make_record(i) for i in records]
    cur.close()
    conn.close()
    return records


def get_statistics():
    conn, cur = get_connection(database='apartments')
    # cur.execute('SELECT reltuples AS approximate_row_count FROM pg_class WHERE relname = %s', ('apartments',))
    # use this if the number of records is large
    cur.execute('SELECT COUNT(*) FROM apartments')
    row_count = cur.fetchone()[0]
    return row_count


if __name__ == '__main__':
    create_apartments()
