#!/usr/bin/env bash

# set up all needed env variables
export APP_SETTINGS="app.config.DevelopmentConfig"
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/apartments_dev"

# making migrations
python app/pg_db/manage.py db migrate
python app/pg_db/manage.py db upgrade

# starting app
python app/main.py
