from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path
import os

class FlatData:
	def __init__(self):
		env_path = Path('.') / '.env'
    		load_dotenv(dotenv_path=env_path)
    		DATABASE_URI = os.getenv("DATABASE_URI")
		self.engine = create_engine(DATABASE_URI)
		self.connection = self.engine.connect()
		self.flat_table = 'flat_info'
		self.announcement_table = 'announcement_info'
		self.number_of_flats = self.get_count_flats()
		self.avg_price_uah = self.get_price_UAH()
		self.avg_price_usd = self.get_price_USD()


	def get_count_flats(self):
		counter = self.engine.execute(f'SELECT COUNT(flat_id) FROM {self.announcement_table}')
		print('counter assigned')
		return int(counter.fetchall()[0][0])
	
	def get_price_UAH(self):
		counter_avg = self.engine.execute(f'SELECT AVG(price_UAH) FROM {self.announcement_table}')
		counter_max = self.engine.execute(f'SELECT MAX(price_UAH) FROM {self.announcement_table}')
		counter_min = self.engine.execute(f'SELECT MIN(price_UAH) FROM {self.announcement_table}')
		return int(counter_avg.fetchall()[0][0])

	def get_price_USD(self):
		counter_avg = self.engine.execute(f'SELECT AVG(price_USD) FROM {self.announcement_table}')
		counter_max = self.engine.execute(f'SELECT MAX(price_USD) FROM {self.announcement_table}')
		counter_min = self.engine.execute(f'SELECT MIN(price_USD) FROM {self.announcement_table}')
		return int(counter_avg.fetchall()[0][0])

	def get_values(self, limit, offset):
		return self.values(limit, offset)

	def values(limit, offset):
		res = self.engine.execute(f'SELECT * FROM {self.flat_table} JOIN  {self.announcement_table} ON {self.flat_table}.flat_id = {self.announcement_table}.flat_id ORDER BY date_created LIMIT {limit} OFFSET {offset}')
		return res.fetchall()


