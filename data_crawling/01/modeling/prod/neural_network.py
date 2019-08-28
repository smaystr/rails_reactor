import numpy as np
from sklearn.externals import joblib
import torch
from stop_words import get_stop_words
from pymystem3 import Mystem
from gensim.models import FastText

class NN():
	def __init__(self):
		self.the_model = NeuralNet(number_features = 265, dimensions_hidden = 512)
		self.the_model.load_state_dict(torch.load('modeling/prod/model_weights_with_nlp/neural_network_state_dict_with_nlp'))

	def predict(self, sample):
		real_sample = self.sample_preproces(sample)
		prediction = self.the_model(real_sample)
		return prediction

	def sample_preproces(self, sample):
		cat = self.encode_categorical(sample['type_of_proposal'], sample['city_name'], sample['heating_type'], sample['walls_type'])[0]
		res = np.array([])
		res = np.append(res, cat[:6])
		res = np.append(res, cat[6:226])
		res = np.append(res, sample['total_area'])
		res = np.append(res, sample['living_area'])
		res = np.append(res, sample['kitchen_area'])
		res = np.append(res, sample['floor'])
		res = np.append(res, sample['total_number_of_floors'])
		res = np.append(res, sample['number_of_rooms'])
		res = np.append(res, sample['year_of_construction'])
		res = np.append(res, cat[226:230])
		res = np.append(res, cat[230:])
		res = np.append(res, sample['latitude'])
		res = np.append(res, sample['longitude'])
		res = np.append(res, self.text_preprocessing(sample['description']))
		res = self.scale(res.reshape(1, -1))
		res = torch.Tensor(res)
		return res

	def text_preprocessing(self, text):
		stopwords_russian = get_stop_words('russian')
		stopwords_ukrainian = get_stop_words('ukrainian')
		mystem = Mystem() 
		model_fastText = FastText.load('modeling/prod/nn_preprocessing/fast_text_model') 
		text = ' '.join([t for t in mystem.lemmatize(text.lower()) if t not in stopwords_russian and t not in stopwords_ukrainian and t.isalpha()])
		text = np.mean(model_fastText[text])
		return text

	def scale(self, data):
		scaler = joblib.load(r'modeling/prod/nn_preprocessing/std_scaler.joblib')
		data = scaler.transform(data)
		return data 

	def encode_categorical(self, type_of_proposal, city_name, heating_type, walls_type):
		encoder = joblib.load(r"modeling/prod/nn_preprocessing/one_hot_encoder.joblib")
		sample = encoder.transform(np.array([type_of_proposal, city_name, heating_type, walls_type]).reshape(1, -1)).toarray()
		return sample

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
