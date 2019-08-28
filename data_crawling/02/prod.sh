#!/bin/bash
export DB_HOST=$1
gunicorn -w 4 -t 120 -b 0.0.0.0:8080 "run"