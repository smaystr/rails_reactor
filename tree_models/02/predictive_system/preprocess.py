import pandas as pd
import numpy as np
from scraper.config import DB_USER, DB_HOST, DB_NAME, DB_TABLE, DB_PORT
from sqlalchemy import create_engine
import re
from predictive_system.utils import SafeLabelEncoder, SafeOneHot
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class Dataset:
    def __init__(self, to_save_encoder=True, db=True):
        self.data = None
        if db:
            self.data = self.fetch_data()
        self.text_features = None
        self.scale = None
        self.to_save_encoder = to_save_encoder
        if self.to_save_encoder:
            self.path = Path('encoders/')
            self.path.mkdir(parents=True, exist_ok=True)

    def fetch_data(self):
        try:
            engine = create_engine(
                f"postgresql://{DB_USER}:{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            )
            connection = engine.connect()
            result = pd.read_sql("SELECT * FROM apartment;", engine)

            engine.dispose()
            return result
        except Exception:
            print("Could not connect to databse. Check your config.")
            print(
                f"Received user {DB_USER}; host {DB_HOST}; port {DB_PORT}; database {DB_NAME}"
            )

    def get_text_features(self):
        def number_of_words(text):
            try:
                return len(text.split(" "))
            except Exception:
                return 0

        def number_of_numbers(text):
            try:
                return len(re.findall("\d+", text))
            except Exception:
                return 0

        def get_year(construction):
            try:
                return re.findall("\d+", construction)[0]
            except Exception:
                return 0

        self.data["description_words"] = self.data["description"].apply(
            lambda x: number_of_words(x)
        )
        self.data["title_words"] = self.data["title"].apply(
            lambda x: number_of_words(x)
        )

        self.data["description_numbers"] = self.data["description"].apply(
            lambda x: number_of_numbers(x)
        )
        self.data["title_numbers"] = self.data["title"].apply(
            lambda x: number_of_numbers(x)
        )

        self.data["construction_year"] = self.data["construction_year"].apply(
            lambda x: get_year(x)
        )
        self.data["construction_year"] = self.data["construction_year"].astype(int)

    def fill_missing_coords(self):
        def fill_latitude(row):
            same_city = self.data.loc[df["city"] == row["city"]]
            if len(same_city) > 1:
                return same_city.latitude.mean()
            return None

        def fill_longitude(row):
            same_city = self.data.loc[df["city"] == row["city"]]
            if len(same_city) > 1:
                return same_city.longitude.mean()
            return None

            self.data["latitude"] = self.data.apply(lambda x: fill_latitude(x), axis=1)
            self.data["longitude"] = self.data.apply(
                lambda x: fill_longitude(x), axis=1
            )

    def encode_labels(self, columns=["street", "region", "city"]):
        if self.to_save_encoder:
            for column in columns:
                encoder = SafeLabelEncoder()
                encoder.fit(self.data[column])
                self.data[column] = encoder.transform(self.data[column]).astype(int)

                self.save_encoder(encoder.classes_, column, "encoder")
        else:
            for column in columns:
                self.data[column] = self.label_encoders[column].transform(self.data[column]).astype(int)

    def remove_outliers(self, columns=["uah_price", "total_area", "room_count"]):
        for column in columns:
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1

            self.data = self.data.query(
                f"(@Q1 - 1.5 * @IQR) <= {column} <= (@Q3 + 1.5 * @IQR)"
            )
        self.data.reset_index(drop=True, inplace=True)

    def replace_nans(self, to_replace=0):
        self.data = self.data.replace(np.nan, to_replace)

    def encode_text(self, max_features=500, columns=["title", "description"]):
        column_length = len(columns)
        self.text_features = np.zeros(
            (self.data.shape[0], max_features * column_length)
        )
        if self.to_save_encoder:
            for key, column in enumerate(columns):
                self.data[column] = self.data[column].replace(np.nan, "")
                vectorizer = CountVectorizer(max_features=max_features)
                vectorizer.fit(self.data[column])

                self.text_features[
                    :, key * max_features: (key + 1) * max_features
                ] = vectorizer.transform(self.data[column]).toarray()

                self.save_vectorizer(vectorizer, column)
        else:
            for key, column in enumerate(columns):
                self.data[column] = self.data[column].replace(np.nan, "")

                self.text_features[
                    :, key * max_features: (key + 1) * max_features
                ] = self.vectorizers[column].transform(self.data[column]).toarray()

    def drop_columns(
        self, columns=["id", "pictures", "title", "description", "usd_price"]
    ):
        self.data.drop(columns, axis=1, inplace=True)

    def one_hot_encode(self, columns=["heating", "seller", "wall_material"]):
        if self.to_save_encoder:
            for column in columns:
                self.data[column] = self.data[column].replace(np.nan, "Missing")
                safe_one_hot = SafeOneHot()
                safe_one_hot.fit(self.data[column])

                new_data = pd.DataFrame(safe_one_hot.transform(self.data[column]))
                new_data.columns = [f"{i}_{column}" for i in range(new_data.shape[1])]

                self.data = self.data.join(new_data, how="left")

                self.save_encoder(safe_one_hot.classes_, column, "onehot")
        else:
            for column in columns:
                self.data[column] = self.data[column].replace(np.nan, "Missing")

                new_data = pd.DataFrame(self.onehot_encoders[column].transform(self.data[column]))
                new_data.columns = [f"{i}_{column}" for i in range(new_data.shape[1])]

                self.data = self.data.join(new_data, how="left")

    def to_numpy(self, with_target=True, target="uah_price"):
        if with_target:
            self.target = self.data[target].values.astype(float)
            self.data = self.data.drop([target], axis=1)
        self.data = self.data.select_dtypes(exclude=object).astype(float).values
        if self.text_features is not None:
            self.data = np.hstack((self.data, self.text_features))

    def scale_data(self):
        if self.to_save_encoder:
            self.scaler = StandardScaler()
            self.scaler.fit(self.data)
            self.save_scaler()
        self.data = self.scaler.transform(self.data)

    def get_data(self):
        return self.data

    def save_scaler(self):
        with open(self.path / "scaler.pkl", "wb") as output:
            pickle.dump(self.scaler, output)

    def save_encoder(self, classes, column, encoder_name):
        np.save(self.path / f"{column}_{encoder_name}.npy", classes)

    def save_vectorizer(self, vectorizer, column):
        with open(self.path / f"{column}_vectorizer.pkl", "wb") as output:
            pickle.dump(vectorizer, output)

    def load_encoders_and_stuff(
        self, path,
        one_hot_cols=["heating", "seller", "wall_material"],
        label_enc_cols=["street", "region", "city"],
        vec_cols=["title", "description"]
    ):

        self.onehot_encoders = {}
        self.vectorizers = {}
        self.label_encoders = {}
        self.scaler = None

        path = Path(path)
        if not path.exists:
            raise "Path doesn't exist."

        for column in one_hot_cols:
            try:
                path_to_encoder = next(path.glob(f'{column}_*.npy'))
                one_hot = SafeOneHot()
                one_hot.classes_ = np.load(path_to_encoder)
                self.onehot_encoders[column] = one_hot
            except StopIteration:
                print(f"Encoder for {column} not found.")

        for column in label_enc_cols:
            try:
                path_to_encoder = next(path.glob(f'{column}_*.npy'))
                label_encoder = SafeLabelEncoder()
                label_encoder.classes_ = np.load(path_to_encoder)
                self.label_encoders[column] = label_encoder
            except StopIteration:
                print(f"Encoder for {column} not found.")

        for column in vec_cols:
            try:
                path_to_encoder = next(path.glob(f'{column}_*.pkl'))
                with open(path_to_encoder, 'rb') as vectorizer:
                    vectorizer = pickle.load(vectorizer)

                self.vectorizers[column] = vectorizer
            except StopIteration:
                print(f"Vectorizer for {column} not found.")

        try:
            path_to_scaler = next(path.glob('scaler.pkl'))
            with open(path_to_scaler, 'rb') as scaler:
                scaler = pickle.load(scaler)

            self.scaler = scaler
        except StopIteration:
            print(f"Scaler for {column} not found.")
