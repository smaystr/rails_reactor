from dotenv import load_dotenv
import os

load_dotenv()

DATABASE = {
    'drivername': os.getenv('DB_DRIVERNAME'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'username': os.getenv('DB_USERNAME'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE')
}
