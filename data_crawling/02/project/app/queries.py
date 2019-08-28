from sqlalchemy import create_engine
from json import dumps

from project.app.utilities import alchemyencoder

def get_statistics(
        db,
        url
):
    """
    Get statistics from the database
    :type db: flask_sqlalchemy.SQLAlchemy
    :type url: str
    """
    status = 'OK'
    result = {}
    engine = create_engine(url)
    connection = engine.connect()
    offer_rs = connection.execute("""
        SELECT
        COUNT(seller) as sellers_amount,
        COUNT(id) as offers_amount,
        AVG(price) as mean_price,
        STDDEV(price) as std_price
            FROM offer;
    """)
    results = dumps(
        [dict(row) for row in offer_rs],
        default=alchemyencoder
    )
    result['offer'] = results
    image_rs = connection.execute("""
        SELECT
        COUNT(image_urls) as images_amount
            FROM subinfo
    """)
    results = dumps(
        [dict(row) for row in image_rs],
        default=alchemyencoder
    )
    result['subinfo'] = results
    apartment_rs = connection.execute("""
        SELECT
        AVG(year) AS mean_year,
        STDDEV(year) AS std_year,
        AVG(total_area) AS mean_total_area,
        STDDEV(total_area) AS std_total_area,
        AVG(living_area) AS mean_living_area,
        STDDEV(living_area) AS std_living_area,
        AVG(kitchen_area) AS mean_kitchen_area,
        STDDEV(kitchen_area) AS std_kitchen_area,
        AVG(rooms) AS mean_rooms_amount,
        STDDEV(rooms) AS std_rooms_amount
            FROM apartment
            WHERE year != 0;
    """)
    results = dumps(
        [dict(row) for row in apartment_rs],
        default=alchemyencoder
    )
    result['apartment'] = results
    return status, result


def get_records(
        db,
        url,
        limit,
        offset,
        limit_default=10,
        offset_default=0
):
    """
    Get records from the database
    :type db: flask_sqlalchemy.SQLAlchemy
    :type url: str
    :type limit: int
    :type offset: int
    :type limit_default: int
    :type offset_default: int
    """
    if limit is None:
        limit = limit_default
    if offset is None:
        offset = offset_default
    status = 'OK'
    engine = create_engine(url)
    connection = engine.connect()
    rs = connection.execute(
        f"""
        SELECT * from offer
            JOIN subinfo ON (offer.id = subinfo.id)
            ORDER BY publish_date ASC
            LIMIT {limit} OFFSET {offset};
        """
    )
    results = dumps(
        [dict(row) for row in rs],
        default=alchemyencoder
    )
    return status, results
