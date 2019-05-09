from emoji import UNICODE_EMOJI
from pythainlp.corpus import  thai_stopwords
from data import *
stopwords = list(thai_stopwords())
from string import punctuation
poson=True
def have_particles(s):
    for i in particles:
        if i in s:
            return True
    return False
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

def is_thai(text):
    for ch in text:
        ch_val = ord(ch)
        if ch_val >= 3584 and ch_val <= 3711:
            pass
        else:
            return False
    return True

def get_features(name,word,pos=None):
    features={
        name+'.word': word,
        name+'.is_stopword': word in stopwords,
        name+'.punctuation':check_punctuation(word),
        name+'.is_space':word.isspace(),
        name+'.is_digit': word.isdigit(),
        name+'.is_conjunctions': word in conjunctions+ccc,
        name+'.is_emoji':is_emoji(word),
        name+'.has_t1': word.startswith('การ'),
        name+'.has_t2': word.startswith('ความ'),
        name+".has_particles":have_particles(word)
    }
    if word.isspace():
        features[name+'.len_space']=len(word)
    if word.isdigit()==False and word.isspace()==False:
        features[name+'.is_thai']=is_thai(word)
    if pos!=None:
        features[name+'.pos'] = pos
    return features

def punct_features(tokens, i):
    if poson:
        features=get_features("word",tokens[i][0],tokens[i][1])
    else:
        features=get_features("word",tokens[i][0])
    if i > 0:
        if poson:
            features.update(get_features("prevword",tokens[i-1][0],tokens[i-1][1]))
        else:
            features.update(get_features("prevword",tokens[i-1][0]))
    else:
        features['BOS'] = True # Special "Beginning of Sequence" tag
    if i > 1:
        if poson:
            features.update(get_features("prevword2",tokens[i-2][0],tokens[i-2][1]))
        else:
            features.update(get_features("prevword2",tokens[i-2][0]))
    if i > 2:
        if poson:
            features.update(get_features("prevword3",tokens[i-3][0],tokens[i-3][1]))
        else:
            features.update(get_features("prevword3",tokens[i-3][0]))
    # Features from next word
    if i < len(tokens)-1:
        if poson:
            features.update(get_features("nextword",tokens[i+1][0],tokens[i+1][1]))
        else:
            features.update(get_features("nextword",tokens[i+1][0]))
    else:
        features['EOS'] = True # Special "End of Sequence" tag
    if i < len(tokens)-2:
        if poson:
            features.update(get_features("nextword2",tokens[i+2][0],tokens[i+2][1]))
        else:
            features.update(get_features("nextword2",tokens[i+2][0]))
    if i < len(tokens)-3:
        if poson:
            features.update(get_features("nextword3",tokens[i+3][0],tokens[i+3][1]))
        else:
            features.update(get_features("nextword3",tokens[i+3][0]))
    return features
def extract_features(doc):
    return [punct_features(doc, i) for i in range(len(doc))]

def get_labels(doc):
    if poson:
        return [tag for (token,pos,tag) in doc]
    return [tag for (token,tag) in doc]