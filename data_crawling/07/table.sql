SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_with_oids = false;


CREATE TABLE public.apartment (
    id integer NOT NULL,
    uah_price integer,
    usd_price integer,
    description text,
    street character varying(100),
    region character varying(100),
    total_area integer,
    room_count smallint,
    construction_year character varying(30),
    heating character varying(30),
    seller character varying(50),
    wall_material character varying(50),
    verified_price boolean,
    verified_apartment boolean,
    latitude double precision,
    longitude double precision,
    pictures text[],
    city character varying(40),
    title text
);


ALTER TABLE public.apartment OWNER TO craq;


CREATE SEQUENCE public.apartment_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.apartment_id_seq OWNER TO craq;


ALTER SEQUENCE public.apartment_id_seq OWNED BY public.apartment.id;


ALTER TABLE ONLY public.apartment ALTER COLUMN id SET DEFAULT nextval('public.apartment_id_seq'::regclass);

ALTER TABLE ONLY public.apartment
    ADD CONSTRAINT apartment_pkey PRIMARY KEY (id);
