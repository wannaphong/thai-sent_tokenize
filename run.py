# -*- coding: utf-8 -*-
import codecs
import nltk
from pythainlp.tokenize import word_tokenize
thai_tokenize="icu"
conjunctions="""ก็
กว่า
ก่อน
กับ
เกลือก
ครั้น
ค่าที่
คือ
จน
จนกว่า
จนถึง
จึง
ฐาน
ด้วย
ได้แก่
ตราบ
แต่
แต่ว่า
ถ้า
ถึง
ทว่า
ทั้งนี้
เท่ากับ
เนื่องจาก
เนื่องด้วย
เนื่องแต่
เผื่อ
เพราะ
เมื่อ
แม้
แม้ว่า
รึ
เลย
แล
และ
หรือ
ว่า
เว้นแต่
ส่วน
หาก
หากว่า
เหตุ
เหมือน
อย่างไรก็ดี
อย่างไรก็ตาม""".split("\n")
with codecs.open("corpus.txt", 'r',encoding='utf8') as f:
	lines1 = f.read().splitlines()
f.close()
data_all=[]
print("จำนวนประโยค : "+str(len(lines1)))
for lines in lines1:
	text=word_tokenize(lines,thai_tokenize)
	data_all.append(text)
sents=data_all
tokens = []
boundaries = set()
offset = 0
for sent in sents:
	tokens.extend(sent)
	offset += len(sent)
	boundaries.add(offset-1)
def punct_features(tokens, i):
	if len(tokens)-(i+1)>0 and len(tokens)-(i-1)>0:
		return {'conjunctions':tokens[i] in conjunctions,'next-word-capitalized': tokens[i+1][0],'prev-word': tokens[i-1],'punct': tokens[i],'prev-word-is-one-char': len(tokens[i-1]) == 1}
	elif len(tokens)-(i+1)>0:
		return {'conjunctions':tokens[i] in conjunctions,'next-word-capitalized': tokens[i+1][0],'prev-word': None,'punct': tokens[i],'prev-word-is-one-char': False}
	elif len(tokens)-(i-1)>0:
		return {'conjunctions':tokens[i] in conjunctions,'next-word-capitalized': None,'prev-word': tokens[i-1],'punct': tokens[i],'prev-word-is-one-char': len(tokens[i-1]) == 1}
	else:
		return {'conjunctions':tokens[i] in conjunctions,'next-word-capitalized': None,'prev-word': None,'punct': tokens[i],'prev-word-is-one-char': False}

featuresets = [(punct_features(tokens, i), (i in boundaries)) for i in range(1, len(tokens)-1)]
#print(featuresets)
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
#print(train_set)
classifier = nltk.NaiveBayesClassifier.train(train_set)
t=nltk.classify.accuracy(classifier, test_set)
print(t)
def segment_sentences(words):
	start = 0
	sents = []
	for i, word in enumerate(words):
		try:
			if classifier.classify(punct_features(words, i)) == True:
				sents.append(words[start:i+1])
				start = i+1
		except:
			pass
	if start < len(words):
		sents.append(words[start:])
	return sents
while True:
	t=input("Text : ")
	v=word_tokenize(t,thai_tokenize)
	#print(v)
	b=segment_sentences(v)
	print('/'.join([''.join(i) for i in b]))