import sqlalchemy as db
import pandas as pd
import numpy as np
from catboost import Pool, CatBoostRegressor


engine = db.create_engine(f'postgresql://bogdanivanyuk:bogdanivanyuk@localhost:5431/flats_data')
connection = engine.connect()
metadata = db.MetaData()
flat_info = db.Table('flat_info', metadata, autoload=True, autoload_with=engine)
announcement_info = db.Table('announcement_info', metadata, autoload=True, autoload_with=engine)

#Equivalent to 'SELECT * FROM census'
query_flat_info = connection.execute(db.select([flat_info]))
df_flat_info = pd.DataFrame(query_flat_info)
df_flat_info.columns = query_flat_info.keys()

query_announcement_info = connection.execute(db.select([announcement_info]))
df_announcement_info = pd.DataFrame(query_announcement_info)
df_announcement_info.columns = query_announcement_info.keys()

data = pd.merge(df_announcement_info, df_flat_info, on='flat_id')
data = data.drop(['page_url', 'image_urls','verified', 'title', 'street_name', 'price_uah', 'date_created', 'flat_id'], axis = 1)
data = data.fillna('NA')

from joblib import load, dump
tfidf = load('text_representation_tfidf.joblib')
svd = load('text_representation_svd.joblib')

from stop_words import get_stop_words
stop_words_russian = get_stop_words('russian')
stop_words_ukr = get_stop_words('ukrainian')
from pymystem3 import Mystem
mystem = Mystem() 
data['description'] = data['description'].apply(lambda x: ' '.join([t for t in mystem.lemmatize(x.lower()) if 
                                                                    (t not in stop_words_russian and t not in stop_words_ukr and t.isalpha() and len(t) > 2)]))
X = tfidf.fit_transform(data.description)
X = svd.fit_transform(X)
X = pd.DataFrame(X)
data = pd.concat([data, X], axis=1)

categorical_cols = ['type_of_proposal', 'city_name', 'heating_type', 'walls_type'] 
data = data.drop(data[(data['price_usd'] > 1000000) | (data['total_area'] > 600) | (data['living_area'] > 200) | (data['kitchen_area'] > 100) | (data['floor'] > 40) | 
                          (data['number_of_rooms'] > 6)].index)
target = data['price_usd']
data = data.drop('price_usd', axis = 1)
# preprocessing steps
data['year_of_construction'] = data['year_of_construction'].apply(lambda x: re.findall(r'\b\d+\b',str(x))[0] 
                                                                  if len(re.findall(r'\b\d+\b',str(x))) != 0 else -1)
data = data.drop('description', axis =1)

train_data, test_data,train_label, test_label = train_test_split(data, target, test_size=0.2, shuffle=True, random_state=42)

train_pool_preds = Pool(train_data,  
                  cat_features=['type_of_proposal','city_name', 'heating_type', 'walls_type'])

test_pool = Pool(test_data, 
                 cat_features=['type_of_proposal', 'city_name', 'heating_type', 'walls_type']) 

# specify the training parameters 
model = CatBoostRegressor(iterations=10, 
                          depth=16, 
                          learning_rate=1, 
                          loss_function='RMSE')
#train the model
model.fit(train_pool)
# make the prediction using the resulting model
#model.save_model('cat_boost_model_with_nlp')


