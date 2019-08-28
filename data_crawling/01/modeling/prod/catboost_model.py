from catboost import Pool, CatBoostRegressor
import numpy

class CatBoost:
	def __init__(self):
		self.model = CatBoostRegressor().load_model('modeling/prod/model_weights_with_nlp/cat_boost_model')

	def predict(self, sample):
		real_sample = self.sample_preproces(sample)
		real_sample = Pool(real_sample, cat_features=[0, 1, 9, 10])
		prediction = self.model.predict(real_sample)
		return prediction

	def sample_preproces(self, sample):
		return numpy.array([sample['type_of_proposal'],
			sample['city_name'], 
			sample['total_area'],
			sample['living_area'], 
			sample['kitchen_area'], 
			sample['floor'], 
			sample['total_number_of_floors'],
			sample['number_of_rooms'],
			sample['year_of_construction'],
			sample['heating_type'],
			sample['walls_type'], 
			sample['latitude'], 
			sample['longitude'], 
			sample['description']]).reshape(1, -1)
