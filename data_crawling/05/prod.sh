#!/usr/bin/env bash
export APP_SETTINGS="app.config.ProductionConfig"
export DATABASE_URL="postgresql://postgres:5432/apartments"
gunicorn -w 2 -t 120 -b 127.0.0.1:8088 app.main:web_app
