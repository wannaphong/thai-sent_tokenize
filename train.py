# -*- coding: utf-8 -*-
'''
ตัวตัดประโยคภาษาไทย
MARK 8
พัฒนาโดย นาย วรรณพงษ์  ภัททิยไพบูลย์

ก่อนใช้งาน
- ใช้คำสั่ง pip install emoji
- ลอง pythainlp เวชั่นใน GitHub โดยใช้คำสั่ง pip install --ignore-installed https://github.com/PyThaiNLP/pythainlp/archive/dev.zip
'''
import codecs
import nltk
from random import shuffle
from pythainlp.tokenize import dict_word_tokenize,create_custom_dict_trie
from pythainlp.corpus import stopwords
from pythainlp.util import normalize
from string import punctuation
from emoji import UNICODE_EMOJI
import dill
def is_emoji(s):
	for i in s:
		if i in UNICODE_EMOJI:
			return True
	return False
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
ccc="""ก็
เกลือก 
กว่า
กับ
ครั้น 
คือ 
จน 
จึง 
ฉะนั้น
เช่น
ด้วย
ดัง
โดย
ต่อ
ถ้า
ถึง
ทั้ง
ท านอง
เท่า
เผื่อ
เพราะ
เพื่อ
แม้น
แม้
เมื่อ
แล
และ
แล้ว
สำหรับ
เหมือน
หรือ
ฤา
หาก
เหตุ
อนึ่ง
กระนั้นถ้า
ก็
เพราะ
ก็เพื่อ
กับทั้ง
กับอนึ่ง
ครั้นเมื่อ
จนกระทั่ง
จนกว่า
จนถึง
ต่อเมื่อ
แต่ก็
แต่ทว่า
แต่
อย่างไรก็ดี
แต่อย่างไรก็ตาม
ถ้าแล
ถ้าหาก
ถึงกระนั้น
ถึงกระนั้นก็ดี
ถึง
กระนั้นถ้า
ถึงมาทว่า
ถึงแม้
เท่าเมื่อ
เพราะฉะนั้น
เพราะฉะนั้นก็
เพราะ
ด้วย
เพราะเหตุ
เพื่อสำหรับ
แม้
กระนั้น
แล้วก็
แล้วจึง
หรืออย่างไรก็ดี
หากแต่
หากทว่า
เหตุฉะนั้น
เหตุ
ฉะนี้
เหตุด้วย
เหมือนเช่น
เหมือน
อย่าง
อีกทั้ง
กระนั้นก็ตาม
กล่าวคือ
ก็เพราะเหตุว่า
ขณะที่
ขณะนั้น
ครั้ง
ต่อมา
คือว่า
ด้วยเป็นเพราะ
ด้วยเหตุ
ที่
ด้วยเหตุว่า
ดุจหนึ่ง
โดยที่ใน
ขณะเดียวกันก็
โดยที่ในขณะเดียวกันก็
ตรงกันข้ามด้วยซ้ำ
ต่อนั้นมา
ต่อมา
ตั้งแต่
ตัวอย่างเช่น
แต่ตรงกันข้าม
แต่
ทั้งนี้ทั้งนั้นก็
แต่ที่จริง
แต่ที่แท้ก็
แต่ที่
แท้จริง
แต่ว่า
ถึงอย่างไร
ทั้งนี้
ทั้ง ๆ 
ที่่
ทั้งนี้คงเนื่องมาจาก
ทั้งนี้ด้วยเหตุผล
ที่ว่า
ทั้งนี้เนื่องจาก
ทำนองเดียวกันกับ
ที่
ที่จริง
ที่จริงก็
เท่าที่
นอกจากนี้
นั่นก็คือ
เนื่องจาก
เนื่องด้วย
เนื่อง
ด้วยเหตุผลที่ว่า
เนื่องเพราะ
เนื่องมาจาก
เนื่องมาแต่เมื่อ
ใน
ขณะเดียวกัน
ในขณะที่
ในทางตรงกัน
ข้าม
ในทำนองเดียวกัน
ในทำนอง
เดียวกันกับ
ในที่สุด
ในระหว่างนั้น
บัดนี้
ประการหนึ่ง
เป็นต้นแต่
เป็นต้น
ว่า
เพราะว่า
เพราะเหตุที่
เพราะเหตุ
ว่า
เพราะอย่างน้อยก็
แม้ว่า
ระหว่าง
นี้
ราวกับว่า
หรือกล่าวอีกนัยหนึ่ง
หรือที่ถูกก็
หรือมิฉะนั้นอย่างสูงกว่านั้น
ขึ้นไป
หรือไม่ก็
หรือว่า
หรือว่าอีก
อย่างหนึ่ง
หรืออีกนัยหนึ่ง
หรืออีก
อย่างหนึ่ง
หากแต่ว่า
เหตุดังนั้น
เหตุ
นี้
เหมือนดังว่า
อย่างไรก็ดี
อย่างไรก็
ตาม
อนึ่งคือว่า
อีกประการหนึ่ง
อีก
อย่างหนึ่ง""".split("\n") # หน้า 64 http://www.arts.chula.ac.th/~ling/thesis/2556MA-LING-Nalinee.pdf
with codecs.open("corpus.txt", 'r',encoding='utf8') as f:
	lines1 = list(set(normalize(f.read()).splitlines()))
f.close()
test=False#True##เปิด/ปิดการ test
#'''
with codecs.open("thai.txt", "r",encoding="utf8") as f:
	lines2 = f.read().splitlines()#'''
'''
from pythainlp.corpus.thaiword import get_data	
lines2 =get_data()'''
data_all=[]
thaiword=create_custom_dict_trie(list(set(ccc+lines2+stopwords+conjunctions)))
print("จำนวนประโยค : "+str(len(lines1)))
for lines in lines1:
	text=dict_word_tokenize(lines,thaiword)
	#text=word_tokenize(lines,thai_tokenize)
	data_all.append(text)
sents=data_all
tokens = []
boundaries = set()
offset = 0
def check_punctuation(text):
	for i in text:
		if i in list(set(punctuation)):
			return True
	return False
def num_there(s):
    return any(i.isdigit() for i in s)
for sent in sents:
	tokens.extend(sent)
	offset += len(sent)
	boundaries.add(offset-1)
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
	#return {'next-word-capitalized': tokens[i+1][0],'prev-word': tokens[i-1],'punct': tokens[i],'prev-word-is-one-char': len(tokens[i-1]) == 1}

featuresets = [(punct_features(tokens, i), (i in boundaries)) for i in range(1, len(tokens)-1)]
shuffle(featuresets)
if test:
	size = int(len(featuresets) * 0.1)
	train_set, test_set = featuresets[size:], featuresets[:size]
else:
	train_set=featuresets
classifier = nltk.NaiveBayesClassifier.train(train_set)
'''t=nltk.classify.accuracy(classifier, train_set)
print(t)'''
if test:
	t=nltk.classify.accuracy(classifier, test_set)
	print(t)
with open("thaiword.data", "wb") as dill_file:
    dill.dump(thaiword, dill_file)
with open("classifier.data", "wb") as dill_file:
    dill.dump(classifier, dill_file)