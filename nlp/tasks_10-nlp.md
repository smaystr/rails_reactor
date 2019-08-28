# Homework 10

In this homework you are going to work with text available in apartment descriptions crawled in HW7.

1. Look at and describe your data: extend your statistics endpoint with information on most frequent words that are used to describe an appartment. Put it in the report.

2. Clean and preprocess apartment descriptions according to the previous step. Please, describe every processing step and motivation behind it in the report. Think about which words might not be relevant to apartment price prediction.

3. Build TF-IDF representations of apartment descriptions using sklearn's vectorizer. If the vocabulary is too big for your RAM, limit it to N most frequent words, choose N to fit in your memory.

4. Use SVD to make the representations dense(equivalent to doing LSA) - output dimension = 100.

5. Add these representations to your price prediction model from HW 8 and report the results.

6. Now, instead of doing LSA, use [gensim](https://radimrehurek.com/gensim/models/fasttext.html) to train FastText for words in your apartment descriptions. Try playing with word similarities while you're at it. Average word vectors in a description to obtain document representation.

7. Add these representations to your price prediction model from HW 8 and report the results. Compare with LSA.

8. Don't forget to update your prediction endpoint with the new feature - apartment description.

# Deadline

**Due on 25.08.2019 23:59**
