# -*- coding: utf-8 -*-
import codecs
import nltk
from random import shuffle
from pythainlp.tokenize import word_tokenize,dict_word_tokenize,create_custom_dict_trie
from pythainlp.corpus import stopwords
stopwords = stopwords.words('thai')
thai_tokenize="newmm"
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
with codecs.open("thai.txt", 'r',encoding='utf8') as f:
	lines2 = f.read().splitlines()
data_all=[]
thaiword=create_custom_dict_trie(list(set(lines2+stopwords+conjunctions)))
print("จำนวนประโยค : "+str(len(lines1)))
for lines in lines1:
	text=dict_word_tokenize(lines,thaiword)
	#text=word_tokenize(lines,thai_tokenize)
	data_all.append(text)
sents=data_all
tokens = []
boundaries = set()
offset = 0
def num_there(s):
    return any(i.isdigit() for i in s)
for sent in sents:
	tokens.extend(sent)
	offset += len(sent)
	boundaries.add(offset-1)
def punct_features(tokens, i):
	if len(tokens)-(i+1)>0 and len(tokens)-(i-1)>0:
		return {'conjunctions':tokens[i] in conjunctions,'next-word-capitalized': tokens[i+1],'prev-word': tokens[i-1],'word': tokens[i],'is_space' :' ' in tokens[i],'is_num':num_there(tokens[i]),'is_stopword':tokens[i] in stopwords}
	elif len(tokens)-(i+1)>0:
		return {'conjunctions':tokens[i] in conjunctions,'next-word-capitalized': tokens[i+1],'prev-word': None,'word': tokens[i],'is_space' :' ' in tokens[i],'is_num':num_there(tokens[i]),'is_stopword':tokens[i] in stopwords}
	elif len(tokens)-(i-1)>0:
		return {'conjunctions':tokens[i] in conjunctions,'next-word-capitalized': None,'prev-word': tokens[i-1],'word': tokens[i],'is_space' :' ' in tokens[i],'is_num':num_there(tokens[i]),'is_stopword':tokens[i] in stopwords}
	else:
		return {'conjunctions':tokens[i] in conjunctions,'next-word-capitalized': None,'prev-word': None,'word': tokens[i],'is_space' :' ' in tokens[i],'is_num':num_there(tokens[i]),'is_stopword':tokens[i] in stopwords}
	#return {'next-word-capitalized': tokens[i+1][0],'prev-word': tokens[i-1],'punct': tokens[i],'prev-word-is-one-char': len(tokens[i-1]) == 1}

test=True#False#True
featuresets = [(punct_features(tokens, i), (i in boundaries)) for i in range(1, len(tokens)-1)]
shuffle(featuresets)
if test:
	size = int(len(featuresets) * 0.2)
	#train_set=featuresets
	train_set, test_set = featuresets[size:], featuresets[:size]
	#print(train_set)
else:
	train_set=featuresets
classifier = nltk.NaiveBayesClassifier.train(train_set)
if test:
	t=nltk.classify.accuracy(classifier, test_set)
	print(t)
def segment_sentences(words):
	start = 0
	sents = []
	for i, word in enumerate(words):
		try:
			dist = classifier.prob_classify(punct_features(words, i))
			num_true=0.0
			for label in dist.samples():
				if label==True:
					num_true=dist.prob(label)
			if classifier.classify(punct_features(words, i)) == True and num_true>0.60:
				sents.append(words[start:i+1])
				start = i+1
		except:
			pass
	if start < len(words):
		sents.append(words[start:])
	return sents
while True:
	thai_sent=input("Text : ")
	#thai_word=word_tokenize(thai_sent,thai_tokenize)#
	text_all=[]
	temp=thai_sent.split(' ')
	for data in temp:
		thai_word=dict_word_tokenize(data,thaiword)
		text_all.extend(thai_word)
	#print(v)
	thai_sents=segment_sentences(text_all)
	print('sent : '+'/'.join([''.join(i) for i in thai_sents]))