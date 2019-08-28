import os

from flask_migrate import Migrate, MigrateCommand
from flask_script import Manager

from app.main import web_app
from app.pg_db.models import *

web_app.config.from_object(os.environ['APP_SETTINGS'])

migrate = Migrate(web_app, db)
manager = Manager(web_app)

manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()
