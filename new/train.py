# -*- coding: utf-8 -*-
import sklearn_crfsuite
from emoji import UNICODE_EMOJI
from sklearn_crfsuite import scorers,metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate,train_test_split
import dill
from predata import get_conll
from pythainlp.corpus import  thai_stopwords
from string import punctuation
stopwords = list(thai_stopwords())
from data import *
def is_emoji(s):
	for i in s:
		if i in UNICODE_EMOJI:
			return True
	return False

def check_punctuation(text):
	for i in text:
		if i in list(set(punctuation)):
			return True
	return False

def punct_features(tokens, i):
    word = tokens[i][0]
    # Features from current word
    features={
        'word.word': word,
        'word.is_stopword': word in stopwords,
        'word.punctuation':check_punctuation(word),
        'word.is_space':word.isspace(),
        'word.is_digit': word.isdigit(),
		'word.is_conjunctions': word in conjunctions,
		'word.is_emoji':is_emoji(word)
    }
    if i > 0:
        prevword = tokens[i-1][0]
        features['word.prevword'] = prevword
        features['word.previsspace']=prevword.isspace()
        features['word.prev_punctuation']=check_punctuation(prevword)
        features['word.prevstopword']=prevword in stopwords
        features['word.prevwordisdigit'] = prevword.isdigit()
        features['word.prevconjunctions']=prevword in conjunctions
    else:
        features['BOS'] = True # Special "Beginning of Sequence" tag
    # Features from next word
    if i < len(tokens)-1:
        nextword = tokens[i+1][0]
        features['word.nextword'] = nextword
        features['word.nextisspace']=nextword.isspace()
        features['word.next_punctuation']=check_punctuation(nextword)
        features['word.nextstopword']=nextword in stopwords
        features['word.nextwordisdigit'] = nextword.isdigit()
        features['word.nextwordisdigit'] = nextword in conjunctions
    else:
        features['EOS'] = True # Special "End of Sequence" tag
    return features
#classifier = nltk.NaiveBayesClassifier.train(train_set)
def extract_features(doc):
    return [punct_features(doc, i) for i in range(len(doc))]

def get_labels(doc):
    return [tag for (token,tag) in doc]

data=get_conll("data.txt")
X_data = [extract_features(doc) for doc in data]
y_data = [get_labels(doc) for doc in data]

X, X_test, y, y_test = train_test_split(X_data, y_data, test_size=0.1)
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=500,
    all_possible_transitions=True,
    model_filename="test.model0"
)
crf.fit(X, y);

labels = list(crf.classes_)
labels.remove('O')
y_pred = crf.predict(X_test)
e=metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)
print(e)
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))