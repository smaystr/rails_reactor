CREATE TABLE items(
    item_id serial PRIMARY KEY,
    title TEXT,
    price_uah integer,
    price_usd integer,
    description TEXT,
    street TEXT,
    district_name TEXT,
    square_total real,
    square_living real,
    square_kitchen real,
    number_of_rooms integer,
    floor integer,
    number_of_floors integer,
    year_of_construction TEXT,
    type_of_sentence TEXT,
    walls_material TEXT,
    heating TEXT,
    longitude float(8),
    latitude float(8),
    price_verified boolean,
    apartment_verified boolean,
    publishing_date DATE
);

CREATE TABLE images(
    item_id integer NOT NULL,
    link TEXT,
    FOREIGN KEY (item_id) REFERENCES items (item_id)
        MATCH SIMPLE ON UPDATE NO ACTION ON DELETE NO ACTION
);
