# -*- coding: utf-8 -*-
from pythainlp.util import normalize
import codecs
import nltk
from string import punctuation
from emoji import UNICODE_EMOJI
import dill
from pythainlp.tokenize import dict_word_tokenize
from pythainlp.corpus import stopwords
stopwords = stopwords.words('thai')
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
with open('thaiword.data', 'rb') as in_strm:
    thaiword = dill.load(in_strm)
with open('classifier.data', 'rb') as in_strm:
    classifier = dill.load(in_strm)
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
def num_there(s):
    return any(i.isdigit() for i in s)
def punct_features(tokens, i):
	if i<len(tokens)-1 and i!=0:
		return {'conjunctions':tokens[i] in conjunctions,'next-word-capitalized': tokens[i+1],'prev-word': tokens[i-1],'word': tokens[i],'is_space' :' ' in tokens[i],'is_num':num_there(tokens[i]),'is_stopword':tokens[i] in stopwords,'is_punctuation':check_punctuation(tokens[i]),'is_emoji':is_emoji(tokens[i])}
	elif i>0 and len(tokens)>1 and i<len(tokens)-1:
		return {'conjunctions':tokens[i] in conjunctions,'next-word-capitalized': tokens[i+1],'prev-word': tokens[i-1],'word': tokens[i],'is_space' :' ' in tokens[i],'is_num':num_there(tokens[i]),'is_stopword':tokens[i] in stopwords,'is_punctuation':check_punctuation(tokens[i]),'is_emoji':is_emoji(tokens[i])}
	elif i==len(tokens)-1:
		return {'conjunctions':tokens[i] in conjunctions,'next-word-capitalized': '','prev-word': tokens[i-1],'word': tokens[i],'is_space' :' ' in tokens[i],'is_num':num_there(tokens[i]),'is_stopword':tokens[i] in stopwords,'is_punctuation':check_punctuation(tokens[i]),'is_emoji':is_emoji(tokens[i])}
	elif i==0 and len(tokens)>1:
		return {'conjunctions':tokens[i] in conjunctions,'next-word-capitalized': tokens[i+1],'prev-word': '','word': tokens[i],'is_space' :' ' in tokens[i],'is_num':num_there(tokens[i]),'is_stopword':tokens[i] in stopwords,'is_punctuation':check_punctuation(tokens[i]),'is_emoji':is_emoji(tokens[i])}
	else:
		return {'conjunctions':tokens[i] in conjunctions,'next-word-capitalized': '','prev-word': '','word': tokens[i],'is_space' :' ' in tokens[i],'is_num':num_there(tokens[i]),'is_stopword':tokens[i] in stopwords,'is_punctuation':check_punctuation(tokens[i]),'is_emoji':is_emoji(tokens[i])}
def segment_sentences(words):
	start = 0
	sents = []
	num_true=0.0
	num_all=0
	for i, word in enumerate(words):
		dist = classifier.prob_classify(punct_features(words, i))
		for label in dist.samples():
			if label==True:
				num_true+=dist.prob(label)
		if classifier.classify(punct_features(words, i)) == True and num_true>0.60:
			sents.append(words[start:i+1])
			start = i+1
	if start < len(words):
		sents.append(words[start:])
	#print(num_true/num_all)
	return sents
while True:
	thai_sent=normalize(input("Text : "))
	#thai_word=word_tokenize(thai_sent,thai_tokenize)#
	text_all=dict_word_tokenize(thai_sent,thaiword)#[]
	"""temp=thai_sent.split(' ')
	for data in temp:
		thai_word=dict_word_tokenize(data,thaiword)
		text_all.extend(thai_word)"""
	#print(text_all)
	thai_sents=segment_sentences(text_all)
	print('sent : '+'/'.join([''.join(i) for i in thai_sents]))