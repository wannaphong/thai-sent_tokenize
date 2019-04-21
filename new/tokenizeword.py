# -*- coding: utf-8 -*-
from pythainlp.tokenize import word_tokenize,dict_trie
from pythainlp.corpus import  thai_stopwords,thai_words,tnc
from pythainlp.util import normalize
import data
stopwords = list(thai_stopwords())
thaiword=list(thai_words())
tnc1=[word for word,i in tnc.word_freqs()]
thaiword.remove("กินข้าว")
datadict=dict_trie(list(set(data.ccc+thaiword+stopwords+data.conjunctions+tnc1)))
def wordcut(word):
    global datadict
    return word_tokenize(word,custom_dict=datadict)