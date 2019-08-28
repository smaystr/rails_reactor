import logging

from sqlalchemy import desc

from app.pg_db.models import Apartment, ApartmentImage, SellerInfo


def add_apartment(db, kwargs):
    try:
        seller_info = kwargs.pop('seller_info')
        images = kwargs.pop('images')
        result = Apartment(
            **kwargs
        )
        db.session.add(result)
        db.session.commit()
        apartment_id = result.id

        for image in images:
            img = ApartmentImage(apartment_id, image)
            db.session.add(img)
        seller_info['apartment_id'] = apartment_id
        seller_info = SellerInfo(**seller_info)
        db.session.add(seller_info)
        db.session.commit()

    except Exception as e:
        logging.error(str(e))


def get_apartments_count(db):
    return db.session.query(Apartment).count()


def get_apartments(db, limit, offset):
    query = db.session.query(Apartment).order_by(desc(Apartment.created_at))
    if limit:
        query = query.limit(limit)
    if offset:
        query = query.offset(offset)
    return query.all()
