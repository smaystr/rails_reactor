from sqlalchemy import create_engine
from .. import config
import string
import nltk


class ApartmentOverview():
    def __init__(self, number_of_words=100):
        self.engine = create_engine(
            f'postgresql://{config.DB_USER}:{config.DB_USER}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}'
        )
        self.connection = self.engine.connect()
        self.table = config.DB_TABLE
        self.number_of_words = number_of_words
        self.most_common_words = self.get_most_common_words()
        self.mean_uah = self.get_mean_uah()
        self.mean_ush = self.get_mean_usd()
        self.count = self.get_count()

    def get_count(self):
        count = self.engine.execute(f"SELECT COUNT(id) FROM {self.table}")
        return int(count.fetchall()[0][0])

    def get_mean_usd(self):
        mean_price = self.engine.execute(f"SELECT AVG(uah_price) FROM {self.table}")
        return float(mean_price.fetchall()[0][0])

    def get_mean_uah(self):
        mean_price = self.engine.execute(f"SELECT AVG(usd_price) FROM {self.table}")
        return float(mean_price.fetchall()[0][0])

    def get_most_common_words(self):
        query = self.engine.execute(f"SELECT description FROM {self.table}")
        text = ' '.join([sentence[0] for sentence in query.fetchall()])
        text = text.translate(str.maketrans('', '', string.punctuation))
        all_words = nltk.tokenize.word_tokenize(text)
        frequency = nltk.FreqDist(word.lower() for word in all_words)
        return frequency.most_common(self.number_of_words)
