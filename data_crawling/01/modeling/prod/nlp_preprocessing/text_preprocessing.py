from sklearn.decomposition import TruncatedSVD
from stop_words import get_stop_words
from joblib import load
from pymystem3 import Mystem

def preprocess_text_trees(text):
	text = text_cleaning(text)
	text = [text]
	text = text_vectorization(text)
	text = text_svd(text)
	return text

def text_cleaning(text):
	stopwords_russian = get_stop_words('russian')
	stopwords_ukrainian = get_stop_words('ukrainian')
	lemmatizer = Mystem()
	text = ' '.join([t for t in lemmatizer.lemmatize(text.lower()) if t not in stopwords_russian and t not in stopwords_ukrainian and t.isalpha()])
	return text

def text_vectorization(text):
	tfidf = load('text_representation_tfidf.joblib')
	text = tfidf.transform(text)
	return text

def text_svd(text):
	svd = load('text_representation_svd.joblib')
	text = svd.transform(text)
	return text
