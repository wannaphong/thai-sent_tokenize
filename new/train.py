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
from tokenizeword import wordcut as word_tokenize
stopwords = list(thai_stopwords())

poson=True
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
    if poson:
        pos = tokens[i][1]
        features['word.pos'] = pos
    if i > 0:
        prevword = tokens[i-1][0]
        
        features['word.prevword'] = prevword
        if poson:
            pos = tokens[i-1][1]
            features['word.prevpos'] =  pos
        features['word.previsspace']=prevword.isspace()
        features['word.prev_punctuation']=check_punctuation(prevword)
        features['word.prevstopword']=prevword in stopwords
        features['word.prevwordisdigit'] = prevword.isdigit()
        features['word.prevconjunctions']=prevword in conjunctions
    else:
        features['BOS'] = True # Special "Beginning of Sequence" tag
    if i > 1:
        prevword = tokens[i-2][0]
        if poson:
            pos = tokens[i-2][1]
            features['word.prevpos2'] =  pos
        features['word.prevword2'] = prevword
        
        features['word.previsspace2']=prevword.isspace()
        features['word.prev_punctuation2']=check_punctuation(prevword)
        features['word.prevstopword2']=prevword in stopwords
        features['word.prevwordisdigit2'] = prevword.isdigit()
        features['word.prevconjunctions2']=prevword in conjunctions
    # Features from next word
    if i < len(tokens)-1:
        nextword = tokens[i+1][0]
        if poson:
            pos=tokens[i+1][1]
            features['word.nextpos'] = pos
        features['word.nextword'] = nextword
        
        features['word.nextisspace']=nextword.isspace()
        features['word.next_punctuation']=check_punctuation(nextword)
        features['word.nextstopword']=nextword in stopwords
        features['word.nextwordisdigit'] = nextword.isdigit()
        features['word.nextwordisdigit'] = nextword in conjunctions
    else:
        features['EOS'] = True # Special "End of Sequence" tag
    if i < len(tokens)-2:
        nextword = tokens[i+2][0]
        if poson:
            pos=tokens[i+2][1]
            features['word.nextpos2'] = pos
        features['word.nextword2'] = nextword
        
        features['word.nextisspace2']=nextword.isspace()
        features['word.next_punctuation2']=check_punctuation(nextword)
        features['word.nextstopword2']=nextword in stopwords
        features['word.nextwordisdigit2'] = nextword.isdigit()
        features['word.nextwordisdigit2'] = nextword in conjunctions
    return features
#classifier = nltk.NaiveBayesClassifier.train(train_set)
def extract_features(doc):
    return [punct_features(doc, i) for i in range(len(doc))]

def get_labels(doc):
    if poson:
        return [tag for (token,pos,tag) in doc]
    return [tag for (token,tag) in doc]

data=get_conll("data.txt",poson)
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
e=metrics.flat_f1_score(y, crf.predict(X),
                      average='weighted', labels=labels)
print(e)

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
#print(X[0])
from pythainlp.tag import pos_tag
def get_sent(text):
    global poson
    if poson:
        word_cut=[(i,pos) for i,pos in pos_tag(word_tokenize(text),engine="perceptron", corpus="orchid_ud")]
    else:
        word_cut=[(i,) for i in word_tokenize(text)]
    X_test =[punct_features(word_cut, i) for i in range(len(word_cut))]
    #print(X_test)
    #print(word_cut)
    y_=crf.predict_single(X_test)
    sent= [(word_cut[i][0],data) for i,data in enumerate(y_)]
    textsent=""
    for i,data in enumerate(sent):
        if i>0 and data[1]=="B-S":
            textsent+="|"
        textsent+=data[0]
    return textsent

print(get_sent("การแข่งขันสตาร์ทอัพโดยทั่วไปมีเป้าหมายเพื่อส่งเสริมและเพิ่มขีดความสามารถของธุรกิจที่มีไอเดียเจ๋งๆ ให้มีเงินทุนเพื่อจะสานต่อธุรกิจของตนต่อไปได้ แต่ U.REKA โครงการส่งเสริมนักวิจัยและผู้ประกอบการนวัตกรรม มีเป้าหมายไกลถึงระดับที่ให้ประเทศไทยมีความแข็งแรงทางเทคโนโลยี ทัดเทียมนานาชาติ"))