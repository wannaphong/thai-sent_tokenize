# -*- coding: utf-8 -*-
from pythainlp.tokenize import dict_word_tokenize,dict_trie
from pythainlp.corpus import  thai_stopwords,thai_words
from pythainlp.util import normalize
import data
stopwords = list(thai_stopwords())
thaiword=list(thai_words())
datadict=dict_trie(list(set(data.ccc+thaiword+stopwords+data.conjunctions)))
def wordcut(word):
    return dict_word_tokenize(word,datadict)