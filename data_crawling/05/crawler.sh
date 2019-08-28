#!/usr/bin/env bash

# set up all needed env variables
export APP_SETTINGS="app.config.DevelopmentConfig"
export DATABASE_URL="postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@postgres:5432/$POSTGRES_DB"

python app/scrapy_client.py
