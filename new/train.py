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
from data import *

poson=True
data=get_conll("data.txt",poson)
#random.shuffle(data)
#random.shuffle(data)
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
'''
e=metrics.flat_f1_score(y, crf.predict(X),
                      average='weighted', labels=labels)
print(e)
'''
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

print(get_sent("การแข่งขันสตาร์ทอัพโดยทั่วไปมีเป้าหมายเพื่อส่งเสริมและเพิ่มขีดความสามารถของธุรกิจที่มีไอเดียเจ๋งๆ ให้มีเงินทุนเพื่อจะสานต่อธุรกิจของตนต่อไปได้"))
print(get_sent("'ศรีสุวรรณ'ยื่นศาลปกครองไต่สวนฉุกเฉิน ระงับการขึ้นค่ารถเมล์ ซัดสร้างภาระให้ประชาชน"))