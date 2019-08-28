from joblib import load
import pprint
from sklearn.preprocessing import LabelEncoder
import numpy 
import modeling.prod.nlp_preprocessing.text_preprocessing as nlp

class DecisionTree:
	def __init__(self):
		#self.model = load('modeling/prod/model_weights/decision_tree_model.joblib') 
		self.model = load('modeling/prod/model_weights_with_nlp/decision_tree_model_with_nlp.joblib')

	def predict(self, sample):
		if 'description' in sample.keys():
			sample['description'] = nlp.preprocess_text_trees(sample['description'])
		real_sample = self.sample_preproces(sample)
		prediction = self.model.predict(real_sample)
		return prediction

	def sample_preproces(self, sample):
		return numpy.array([self.encode_proposal(sample['type_of_proposal']),
			self.encode_city(sample['city_name']), 
			sample['total_area'],
			sample['living_area'], 
			sample['kitchen_area'], 
			sample['floor'], 
			sample['total_number_of_floors'],
			sample['number_of_rooms'],
			sample['year_of_construction'],
			self.encode_heating(sample['heating_type']),
			self.encode_walls(sample['walls_type']), 
			sample['latitude'], 
			sample['longitude'],
			sample['description'],
			sample['description']]).reshape(1, -1)

	def encode_proposal(self, sample):
		encoder = LabelEncoder()
		encoder.classes_ = numpy.load('modeling/prod/label_encoder/type_of_proposal_classes.npy',allow_pickle=True)
		sample = encoder.transform([sample])
		return sample

	def encode_city(self, sample):
		encoder = LabelEncoder()
		encoder.classes_ = numpy.load('modeling/prod/label_encoder/city_classes.npy',allow_pickle=True)
		sample = encoder.transform([sample])
		return sample

	def encode_heating(self, sample):
		encoder = LabelEncoder()
		encoder.classes_ = numpy.load('modeling/prod/label_encoder/heating_classes.npy',allow_pickle=True)
		sample = encoder.transform([sample])
		return sample

	def encode_walls(self, sample):
		encoder = LabelEncoder()
		encoder.classes_ = numpy.load('modeling/prod/label_encoder/walls_classes.npy',allow_pickle=True)
		sample = encoder.transform([sample])
		return sample