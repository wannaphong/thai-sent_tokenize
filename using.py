# -*- coding: utf-8 -*-
import sklearn_crfsuite
from sklearn_crfsuite import scorers,metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate,train_test_split
import dill
import random
from predata import get_conll

from tokenizeword import wordcut as word_tokenize
from features import *


poson=True
#random.shuffle(data)
#random.shuffle(data)
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    #max_iterations=500,
    all_possible_transitions=True,
    model_filename="test.model0"
)
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

while True:
    text=input("text : ")
    print(get_sent(text))