# https://www.kaggle.com/ndrewgele/omg-nlp-with-the-djia-and-reddit

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('../../data/Combined_News_DJIA.csv')

q1 = 1600  # originally 1600
q2 = 1800  # 1800
q3 = 1980  # 1980

train = data[:q1]
val = data[q1+1:q2]
test = data[q2+1:q3]
trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
valheadlines = []
for row in range(0,len(val.index)):
	valheadlines.append(' '.join(str(x) for x in val.iloc[row,2:27]))
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

def run_lr(use_ngram):
	vectorizer = CountVectorizer()
	if use_ngram:
		vectorizer = CountVectorizer(ngram_range=(2,2))

	model = LogisticRegression()
	train_input = vectorizer.fit_transform(trainheadlines)
	model = model.fit(train_input, train["Label"])

	train_pred = model.predict(train_input)
	train_labels = [entry for entry in train["Label"]]
	print("Train Accuracy: ", np.mean(train_pred == train_labels))

	val_input = vectorizer.transform(valheadlines)
	val_predictions = model.predict(val_input)
	val_labels = [entry for entry in val["Label"]]
	print("Val Accuracy: ", np.mean(val_predictions == val_labels))
	
	test_input = vectorizer.transform(testheadlines)
	test_predictions = model.predict(test_input)
	labels = [entry for entry in test["Label"]]
	print("Test Accuracy: ", np.mean(test_predictions == labels))

if __name__ == "__main__":
	print("\n==== BAG OF WORDS ====")
	run_lr(False)
	print("\n\n==== NGRAMS ====")
	run_lr(True)
	print("\n")