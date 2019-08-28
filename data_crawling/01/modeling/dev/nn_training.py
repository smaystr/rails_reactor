import sqlalchemy as db
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data


class NeuralNet(torch.nn.Module):
    def __init__(self, number_features, dimensions_hidden, number_output = 1):
        super(NeuralNet, self).__init__()
        self.hidden_1 = torch.nn.Linear(number_features, dimensions_hidden)
        self.relu = torch.nn.ReLU()
        self.hidden_2 = torch.nn.Linear(dimensions_hidden, dimensions_hidden)
        self.relu_2 = torch.nn.ReLU()
        self.hidden_3 = torch.nn.Linear(dimensions_hidden, dimensions_hidden)
        self.relu_3 = torch.nn.ReLU()
        self.hidden_4 = torch.nn.Linear(dimensions_hidden, dimensions_hidden)
        self.relu_4 = torch.nn.ReLU()
        self.predict = torch.nn.Linear(dimensions_hidden, number_output)
        
    def forward(self, x):
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.hidden_2(x)
        x = self.relu_2(x)
        x = self.hidden_3(x)
        x = self.relu_3(x)
        x = self.hidden_4(x)
        x = self.relu_4(x)
        x = self.predict(x)
        return x

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
data = data.drop(['page_url', 'image_urls','verified', 'title', 'street_name'], axis = 1)

# outlier detection
data = data.drop(data[(data['price_usd'] > 1000000) | (data['total_area'] > 600) | (data['living_area'] > 200) | (data['kitchen_area'] > 100) | (data['floor'] > 40) | 
                          (data['number_of_rooms'] > 6)].index)
# preprocessing steps
data['year_of_construction'] = data['year_of_construction'].apply(lambda x: re.findall(r'\b\d+\b',str(x))[0] if len(re.findall(r'\b\d+\b',str(x))) != 0 else -1)
data['type_of_proposal'] = data['type_of_proposal'].replace(r'^\s*$', 'NA_proposal', regex=True)
data['heating_type'] = data['heating_type'].replace(r'^\s*$', 'NA_heating', regex=True)
data['year_of_construction'] = data['year_of_construction'].astype(int)

from stop_words import get_stop_words
stop_words_russian = get_stop_words('russian')
stop_words_ukr = get_stop_words('ukrainian')
from pymystem3 import Mystem
mystem = Mystem() 
data['description'] = data['description'].apply(lambda x: ' '.join([t for t in mystem.lemmatize(x.lower()) if 
                                                                    (t not in stop_words_russian and t not in stop_words_ukr and t.isalpha())]))

corpus_text = '\n'.join(data['description'])
sentences = corpus_text.split('\n')

from gensim.models import FastText
model_fastText = FastText(sentences, size=100, window=5, min_count=5, workers=4)
data['description'] = data['description'].apply(lambda x: np.mean(model_fastText[x], axis =0))

ohe = OneHotEncoder(categories='auto')
feature_arr = ohe.fit_transform(data[['type_of_proposal','city_name', 'heating_type', 'walls_type']]).astype(int).toarray()
feature_labels = ohe.categories_
feature_labels = np.concatenate(feature_labels).ravel()
data = data.drop(['type_of_proposal', 'city_name', 'heating_type', 'walls_type'], axis=1)
data[feature_labels] = pd.DataFrame(feature_arr, columns=feature_labels)

target = data['price_usd']
data = data.drop(['price_usd', 'price_uah', 'date_created', 'flat_id'], axis=1)

data = data.fillna(0)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True, random_state=42)

torch.manual_seed(1)    # reproducible

scaler = StandardScaler()
x_train = scaler.fit_transform(data)

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_test = torch.Tensor(x_test.values)
y_test = torch.Tensor(y_test.values)

net = NeuralNet(number_features=x_train.shape[1], dimensions_hidden = 512)
print(net)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
epochs = 10
loss_function = torch.nn.MSELoss()

plt.ion()
for i in range(epochs):
    preds = net(x_train)
    loss = loss_function(preds, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    preds_valid = net(x_test)
    loss_valid = loss_function(preds_valid, y_test)
    print(f'Epochs RMSE {i}: train - {torch.sqrt(loss)}; validation - {torch.sqrt(loss_valid)}')


#net = torch.save(net.state_dict(), 'neural_network_state_dict_with_nlp')



































