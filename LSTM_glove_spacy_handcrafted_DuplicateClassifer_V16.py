# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 20:58:34 2017

@author: Giancarlo
"""

#=========================== import packages ==========================#

import os
import re
import csv
import pickle
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from keras.layers import Merge, merge #Concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


#======================== set directories and parameters ==================#

BASE_DIR = '/home/cuda/Desktop/Quora/input/'
EMBEDDING_FILE = BASE_DIR + 'w2v/GoogleNews-vectors-negative300.bin'
GLOVE_EMBEDDING = BASE_DIR + 'glove.840B/glove.840B.300d.txt'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.15

num_lstm = 96#32#np.random.randint(175, 275)
num_dense = 48#16#np.random.randint(100, 150)
rate_drop_lstm = 0.25#0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.25#0.15 + np.random.rand() * 0.25
nb_words = 120259

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_glove840_spacy_magicfeat%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)


#========================= process texts in datasets =====================#

print('Processing text dataset')

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=True, stem_words=True):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r"\0k ", "0000 ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

texts_1 = [] 
texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[3]))
        texts_2.append(text_to_wordlist(values[4]))
        labels.append(int(values[5]))
print('Found %s texts in train.csv' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_ids = []
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[1]))
        test_texts_2.append(text_to_wordlist(values[2]))
        test_ids.append(values[0])
print('Found %s texts in test.csv' % len(test_texts_1))

#========================= Pickle texts lists ============================#
picklepath = '/home/cuda/Desktop/Quora/input/text_lists/normal/'
with open(picklepath+"labels.txt", "wb") as fout:
    pickle.dump(labels, fout)
with open(picklepath+"test_ids.txt", "wb") as fout:
    pickle.dump(test_ids, fout)

picklepath = '/home/cuda/Desktop/Quora/input/text_lists/no_stopwords_stemmed/'   #change folder based on text_to_list features
with open(picklepath+"texts_1.txt", "wb") as fout:
    pickle.dump(texts_1, fout)
with open(picklepath+"texts_2.txt", "wb") as fout:
    pickle.dump(texts_2, fout)   
with open(picklepath+"test_texts_1.txt", "wb") as fout:
    pickle.dump(test_texts_1, fout)    
with open(picklepath+"test_texts_2.txt", "wb") as fout:
    pickle.dump(test_texts_2, fout)


#========================= Unpickle texts lists ============================#    
picklepath = '/home/cuda/Desktop/Quora/input/text_lists/normal/'
with open(picklepath+"texts_1.txt", "rb") as fin:
    texts_1 = pickle.load(fin)
with open(picklepath+"texts_2.txt", "rb") as fin:
    texts_2 = pickle.load(fin)
with open(picklepath+"test_texts_1.txt", "rb") as fin:
    test_texts_1 = pickle.load(fin)
with open(picklepath+"test_texts_2.txt", "rb") as fin:
    test_texts_2 = pickle.load(fin)
with open(picklepath+"labels.txt", "rb") as fin:
    labels = pickle.load(fin)
with open(picklepath+"test_ids.txt", "rb") as fin:
    test_ids = pickle.load(fin)

    
picklepath = '/home/cuda/Desktop/Quora/input/text_lists/no_stopwords/'
with open(picklepath+"texts_1.txt", "rb") as fin:
    texts_1_no_stopwords = pickle.load(fin)
with open(picklepath+"texts_2.txt", "rb") as fin:
    texts_2_no_stopwords = pickle.load(fin)
with open(picklepath+"test_texts_1.txt", "rb") as fin:
    test_texts_1_no_stopwords = pickle.load(fin)
with open(picklepath+"test_texts_2.txt", "rb") as fin:
    test_texts_2_no_stopwords = pickle.load(fin)
    
picklepath = '/home/cuda/Desktop/Quora/input/text_lists/stemmed/'
with open(picklepath+"texts_1.txt", "rb") as fin:
    texts_1_stemmed = pickle.load(fin)
with open(picklepath+"texts_2.txt", "rb") as fin:
    texts_2_stemmed = pickle.load(fin)
with open(picklepath+"test_texts_1.txt", "rb") as fin:
    test_texts_1_stemmed = pickle.load(fin)
with open(picklepath+"test_texts_2.txt", "rb") as fin:
    test_texts_2_stemmed = pickle.load(fin)
    
picklepath = '/home/cuda/Desktop/Quora/input/text_lists/no_stopwords_stemmed/'
with open(picklepath+"texts_1.txt", "rb") as fin:
    texts_1_no_stopwords_stemmed = pickle.load(fin)
with open(picklepath+"texts_2.txt", "rb") as fin:
    texts_2_no_stopwords_stemmed = pickle.load(fin)
with open(picklepath+"test_texts_1.txt", "rb") as fin:
    test_texts_1_no_stopwords_stemmed = pickle.load(fin)
with open(picklepath+"test_texts_2.txt", "rb") as fin:
    test_texts_2_no_stopwords_stemmed = pickle.load(fin)
    
#======================== Tokenize and pad texts lists ===================#
tokenizer = Tokenizer(MAX_NB_WORDS)#num_words=
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

sequences_1_no_stopwords = tokenizer.texts_to_sequences(texts_1_no_stopwords)
sequences_2_no_stopwords = tokenizer.texts_to_sequences(texts_2_no_stopwords)
test_sequences_1_no_stopwords = tokenizer.texts_to_sequences(test_texts_1_no_stopwords)
test_sequences_2_no_stopwords = tokenizer.texts_to_sequences(test_texts_2_no_stopwords)

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)


#======================== text length =====================================#
sequences_1_len = map(len, sequences_1)
sequences_1_len = np.array([x /float(MAX_SEQUENCE_LENGTH) for x in sequences_1_len])
sequences_2_len = map(len, sequences_2)
sequences_2_len = np.array([x /float(MAX_SEQUENCE_LENGTH) for x in sequences_2_len])
test_sequences_1_len = map(len, test_sequences_1)
test_sequences_1_len = np.array([x /float(MAX_SEQUENCE_LENGTH) for x in test_sequences_1_len])
test_sequences_2_len = map(len, test_sequences_2)
test_sequences_2_len = np.array([x /float(MAX_SEQUENCE_LENGTH) for x in test_sequences_2_len])

sequences_1_no_stopwords_len = map(len, sequences_1_no_stopwords)
sequences_1_no_stopwords_len = np.array([x /float(MAX_SEQUENCE_LENGTH) for x in sequences_1_no_stopwords_len])
sequences_2_no_stopwords_len = map(len, sequences_2_no_stopwords)
sequences_2_no_stopwords_len = np.array([x /float(MAX_SEQUENCE_LENGTH) for x in sequences_2_no_stopwords_len])
test_sequences_1_no_stopwords_len = map(len, test_sequences_1_no_stopwords)
test_sequences_1_no_stopwords_len = np.array([x /float(MAX_SEQUENCE_LENGTH) for x in test_sequences_1_no_stopwords_len])
test_sequences_2_no_stopwords_len = map(len, test_sequences_2_no_stopwords)
test_sequences_2_no_stopwords_len = np.array([x /float(MAX_SEQUENCE_LENGTH) for x in test_sequences_2_no_stopwords_len])

#======================== word match =====================================#
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))

def word_match_share(zipped_texts):
    q1words = {}
    q2words = {}
    for word in str(zipped_texts[0]).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(zipped_texts[1]).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

word_match_train = np.array(map(word_match_share, zip(texts_1,texts_2)))
word_match_test = np.array(map(word_match_share, zip(test_texts_1,test_texts_2)))
word_match_train_no_stopwords = np.array(map(word_match_share, zip(texts_1_no_stopwords,texts_2_no_stopwords)))
word_match_test_no_stopwords = np.array(map(word_match_share, zip(test_texts_1_no_stopwords,test_texts_2_no_stopwords)))
word_match_train_stemmed = np.array(map(word_match_share, zip(texts_1_stemmed,texts_2_stemmed)))
word_match_test_stemmed = np.array(map(word_match_share, zip(test_texts_1_stemmed,test_texts_2_stemmed)))
word_match_train_no_stopwords_stemmed = np.array(map(word_match_share, zip(texts_1_no_stopwords_stemmed,texts_2_no_stopwords_stemmed)))
word_match_test_no_stopwords_stemmed = np.array(map(word_match_share, zip(test_texts_1_no_stopwords_stemmed,test_texts_2_no_stopwords_stemmed)))

#======================== TF-IDF word match =====================================#
from collections import Counter

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / float(count + eps)

eps = 5000 
words = (" ".join(texts_1)).lower().split()
words.extend((" ".join(texts_1)).lower().split())
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

def tfidf_word_match_share(zipped_texts):
    q1words = {}
    q2words = {}
    for word in str(zipped_texts[0]).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(zipped_texts[1]).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0   
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]  
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

tfidf_word_match_train = np.array(map(tfidf_word_match_share, zip(texts_1,texts_2)))
tfidf_word_match_test = np.array(map(tfidf_word_match_share, zip(test_texts_1,test_texts_2)))

#to use this update counts and weights for each text
#tfidf_word_match_train = np.array(map(tfidf_word_match_share, zip(texts_1,texts_2)))
#tfidf_word_match_test = np.array(map(tfidf_word_match_share, zip(test_texts_1,test_texts_2)))
#tfidf_word_match_train_no_stopwords = np.array(map(tfidf_word_match_share, zip(texts_1_no_stopwords,texts_2_no_stopwords)))
#tfidf_word_match_test_no_stopwords = np.array(map(tfidf_word_match_share, zip(test_texts_1_no_stopwords,test_texts_2_no_stopwords)))
#tfidf_word_match_train_stemmed = np.array(map(tfidf_word_match_share, zip(texts_1_stemmed,texts_2_stemmed)))
#tfidf_word_match_test_stemmed = np.array(map(tfidf_word_match_share, zip(test_texts_1_stemmed,test_texts_2_stemmed)))
#tfidf_word_match_train_no_stopwords_stemmed = np.array(map(tfidf_word_match_share, zip(texts_1_no_stopwords_stemmed,texts_2_no_stopwords_stemmed)))
#tfidf_word_match_test_no_stopwords_stemmed = np.array(map(tfidf_word_match_share, zip(test_texts_1_no_stopwords_stemmed,test_texts_2_no_stopwords_stemmed)))



#============================= generate leaky features =========================#
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

ques = pd.concat([train_df[['question1', 'question2']], \
        test_df[['question1', 'question2']]], axis=0).reset_index(drop='index')
q_dict = defaultdict(set)
for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

def q1_freq(row):
    return(len(q_dict[row['question1']]))
    
def q2_freq(row):
    return(len(q_dict[row['question2']]))
    
def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

train_df['q1_q2_intersect'] = train_df.apply(q1_q2_intersect, axis=1, raw=True)
train_df['q1_freq'] = train_df.apply(q1_freq, axis=1, raw=True)
train_df['q2_freq'] = train_df.apply(q2_freq, axis=1, raw=True)

test_df['q1_q2_intersect'] = test_df.apply(q1_q2_intersect, axis=1, raw=True)
test_df['q1_freq'] = test_df.apply(q1_freq, axis=1, raw=True)
test_df['q2_freq'] = test_df.apply(q2_freq, axis=1, raw=True)

leaks = train_df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]
test_leaks = test_df[['q1_q2_intersect', 'q1_freq', 'q2_freq']]

ss = StandardScaler()
ss.fit(np.vstack((leaks, test_leaks)))
leaks = ss.transform(leaks)
test_leaks = ss.transform(test_leaks)


#============================= Spacy NLP features =========================#
import spacy
nlp = spacy.load('en')


def spacy_Lemma(text):
	doc = nlp(unicode(text))
	wordsLemma = []
	for word in doc:
		wordsLemma.append(int(word.lemma))
	return wordsLemma
	
def spacy_Tag(text):
	doc = nlp(unicode(text))
	wordsTag = []
	for word in doc:
		wordsTag.append(word.tag)
	return  wordsTag
	
def spacy_POS(text):
	doc = nlp(unicode(text))
	wordsPOS = []
	for word in doc:
		wordsPOS.append(word.pos)
	return wordsPOS
	
def spacy_NER(text):
	doc = nlp(unicode(text))
	wordsNER = []
	for word in doc:
		wordsNER.append(word.ent_type)
	return wordsNER	
	
import time
from multiprocessing import Pool
pool = Pool(7)
start_time = time.time()	

q1Lemma = pool.map(spacy_Lemma, texts_1)
print(time.time() - start_time)
start_time = time.time()	
q1Tag = pool.map(spacy_Tag, texts_1)
print(time.time() - start_time)
q1POS = pool.map(spacy_POS, texts_1)
q1NER = pool.map(spacy_NER, texts_1)
q2Lemma = pool.map(spacy_Lemma, texts_2)
q2Tag = pool.map(spacy_Tag, texts_2)
q2POS = pool.map(spacy_POS, texts_2)
q2NER = pool.map(spacy_NER, texts_2)
q1testLemma = pool.map(spacy_Lemma, test_texts_1)
q1testTag = pool.map(spacy_Tag, test_texts_1)
q1testPOS = pool.map(spacy_POS, test_texts_1)
q1testNER = pool.map(spacy_NER, test_texts_1)
q2testLemma = pool.map(spacy_Lemma, test_texts_2)
q2testTag = pool.map(spacy_Tag, test_texts_2)
q2testPOS = pool.map(spacy_POS, test_texts_2)
q2testNER = pool.map(spacy_NER, test_texts_2)
elapsed_time = time.time() - start_time
	

picklepath = '/home/cuda/Desktop/Quora/input/text_lists/normal/'
with open(picklepath+"q1Lemma.txt", "wb") as fout:
    pickle.dump(q1Lemma, fout)
with open(picklepath+"q1Tag.txt", "wb") as fout:
    pickle.dump(q1Tag, fout)
with open(picklepath+"q1POS.txt", "wb") as fout:
    pickle.dump(q1POS, fout)
with open(picklepath+"q1NER.txt", "wb") as fout:
    pickle.dump(q1NER, fout)

with open(picklepath+"q2Lemma.txt", "wb") as fout:
    pickle.dump(q2Lemma, fout)
with open(picklepath+"q2Tag.txt", "wb") as fout:
    pickle.dump(q2Tag, fout)
with open(picklepath+"q2POS.txt", "wb") as fout:
    pickle.dump(q2POS, fout)
with open(picklepath+"q2NER.txt", "wb") as fout:
    pickle.dump(q2NER, fout)
	
with open(picklepath+"q1testLemma.txt", "wb") as fout:
    pickle.dump(q1testLemma, fout)
with open(picklepath+"q1testTag.txt", "wb") as fout:
    pickle.dump(q1testTag, fout)
with open(picklepath+"q1testPOS.txt", "wb") as fout:
    pickle.dump(q1testPOS, fout)
with open(picklepath+"q1testNER.txt", "wb") as fout:
    pickle.dump(q1testNER, fout)

with open(picklepath+"q2testLemma.txt", "wb") as fout:
    pickle.dump(q2testLemma, fout)
with open(picklepath+"q2testTag.txt", "wb") as fout:
    pickle.dump(q2testTag, fout)
with open(picklepath+"q2testPOS.txt", "wb") as fout:
    pickle.dump(q2testPOS, fout)
with open(picklepath+"q2testNER.txt", "wb") as fout:
    pickle.dump(q2testNER, fout)

    

picklepath = '/home/cuda/Desktop/Quora/input/text_lists/normal/'

with open(picklepath+"q1Lemma.txt", "rb") as fin:
    q1Lemma = pickle.load(fin)
with open(picklepath+"q1Tag.txt", "rb") as fin:
    q1Tag = pickle.load(fin)
with open(picklepath+"q1POS.txt", "rb") as fin:
    q1POS = pickle.load(fin)
with open(picklepath+"q1NER.txt", "rb") as fin:
    q1NER = pickle.load(fin)

with open(picklepath+"q2Lemma.txt", "rb") as fin:
    q2Lemma = pickle.load(fin)
with open(picklepath+"q2Tag.txt", "rb") as fin:
    q2Tag = pickle.load(fin)
with open(picklepath+"q2POS.txt", "rb") as fin:
    q2POS = pickle.load(fin)
with open(picklepath+"q2NER.txt", "rb") as fin:
    q2NER = pickle.load(fin)

with open(picklepath+"q1testLemma.txt", "rb") as fin:
    q1testLemma = pickle.load(fin)
with open(picklepath+"q1testTag.txt", "rb") as fin:
    q1testTag = pickle.load(fin)
with open(picklepath+"q1testPOS.txt", "rb") as fin:
    q1testPOS = pickle.load(fin)
with open(picklepath+"q1testNER.txt", "rb") as fin:
    q1testNER = pickle.load(fin)
    
with open(picklepath+"q2testLemma.txt", "rb") as fin:
    q2testLemma = pickle.load(fin)
with open(picklepath+"q2testTag.txt", "rb") as fin:
    q2testTag = pickle.load(fin)
with open(picklepath+"q2testPOS.txt", "rb") as fin:
    q2testPOS = pickle.load(fin)
with open(picklepath+"q2testNER.txt", "rb") as fin:
    q2testNER = pickle.load(fin)
    
	
q1LemmaPad = pad_sequences(q1Lemma, maxlen=MAX_SEQUENCE_LENGTH)
q1TagPad = pad_sequences(q1Tag, maxlen=MAX_SEQUENCE_LENGTH)
q1POSPad = pad_sequences(q1POS, maxlen=MAX_SEQUENCE_LENGTH)
q1NERPad = pad_sequences(q1NER, maxlen=MAX_SEQUENCE_LENGTH)
q2LemmaPad = pad_sequences(q2Lemma, maxlen=MAX_SEQUENCE_LENGTH)
q2TagPad = pad_sequences(q2Tag, maxlen=MAX_SEQUENCE_LENGTH)
q2POSPad = pad_sequences(q2POS, maxlen=MAX_SEQUENCE_LENGTH)
q2NERPad = pad_sequences(q2NER, maxlen=MAX_SEQUENCE_LENGTH)
q1testLemmaPad = pad_sequences(q1testLemma, maxlen=MAX_SEQUENCE_LENGTH)
q1testTagPad = pad_sequences(q1testTag, maxlen=MAX_SEQUENCE_LENGTH)
q1testPOSPad = pad_sequences(q1testPOS, maxlen=MAX_SEQUENCE_LENGTH)
q1testNERPad = pad_sequences(q1testNER, maxlen=MAX_SEQUENCE_LENGTH)
q2testLemmaPad = pad_sequences(q2testLemma, maxlen=MAX_SEQUENCE_LENGTH)
q2testTagPad = pad_sequences(q2testTag, maxlen=MAX_SEQUENCE_LENGTH)
q2testPOSPad = pad_sequences(q2testPOS, maxlen=MAX_SEQUENCE_LENGTH)
q2testNERPad = pad_sequences(q2testNER, maxlen=MAX_SEQUENCE_LENGTH)

#======================== handcrafted v2 features =============================#
def jaccard(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))

def common_words(row):
    return len(set(row['question1']).intersection(set(row['question2'])))

def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))

def total_unq_words_stop(row, stops):
    return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])

def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))

def wc_ratio(row):
    l1 = len(row['question1'])*1.0 
    l2 = len(row['question2'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))

def wc_ratio_unique(row):
    l1 = len(set(row['question1'])) * 1.0
    l2 = len(set(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row['question1']) if x not in stops]) - len([x for x in set(row['question2']) if x not in stops]))

def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['question1']) if x not in stops])*1.0 
    l2 = len([x for x in set(row['question2']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])

def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))

def char_ratio(row):
    l1 = len(''.join(row['question1'])) 
    l2 = len(''.join(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(''.join([x for x in set(row['question2']) if x not in stops])))


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)
    
def tfidf_word_match_share_stops(row, stops=None, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def tfidf_word_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        q1words[word] = 1
    for word in row['question2']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R



#============================= index w2v word vectors =========================#
print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))


#======================== prepare embeddings w2v =============================#
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))



#======================== prepare GLOVE embeddings =============================#
embeddings_index = {}
f = open(GLOVE_EMBEDDING)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

glove_embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        glove_embedding_matrix[i] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(glove_embedding_matrix, axis=1) == 0))



#==================== sample train/validation data =====================#
#np.random.seed(1234)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

#train
data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
data_1_train_len = np.concatenate((sequences_1_len[idx_train], sequences_2_len[idx_train]))
data_2_train_len = np.concatenate((sequences_2_len[idx_train], sequences_1_len[idx_train]))
data_1_no_stopwords_train_len = np.concatenate((sequences_1_no_stopwords_len[idx_train], sequences_2_no_stopwords_len[idx_train]))
data_2_no_stopwords_train_len = np.concatenate((sequences_2_no_stopwords_len[idx_train], sequences_1_no_stopwords_len[idx_train]))
data_0_train_word_match = np.concatenate((word_match_train[idx_train], word_match_train[idx_train]))
data_0_no_stopwords_train_word_match = np.concatenate((word_match_train_no_stopwords[idx_train], word_match_train_no_stopwords[idx_train]))
data_0_stemmed_train_word_match = np.concatenate((word_match_train_stemmed[idx_train], word_match_train_stemmed[idx_train]))
data_0_no_stopwords_stemmed_train_word_match = np.concatenate((word_match_train_no_stopwords_stemmed[idx_train], word_match_train_no_stopwords_stemmed[idx_train]))
data_0_train_tfidf_word_match = np.concatenate((tfidf_word_match_train[idx_train], tfidf_word_match_train[idx_train]))
leaks_train = np.vstack((leaks[idx_train], leaks[idx_train]))
lemma_1_train = np.vstack((q1LemmaPad[idx_train], q2LemmaPad[idx_train]))
tag_1_train = np.vstack((q1TagPad[idx_train], q2TagPad[idx_train]))
pos_1_train = np.vstack((q1POSPad[idx_train], q2POSPad[idx_train]))
ner_1_train = np.vstack((q1NERPad[idx_train], q2NERPad[idx_train]))
lemma_2_train = np.vstack((q2LemmaPad[idx_train], q1LemmaPad[idx_train]))
tag_2_train = np.vstack((q2TagPad[idx_train], q1TagPad[idx_train]))
pos_2_train = np.vstack((q2POSPad[idx_train], q1POSPad[idx_train]))
ner_2_train = np.vstack((q2NERPad[idx_train], q1NERPad[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

#validation
data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
data_1_val_len = np.concatenate((sequences_1_len[idx_val], sequences_2_len[idx_val]))
data_2_val_len = np.concatenate((sequences_2_len[idx_val], sequences_1_len[idx_val]))
data_1_no_stopwords_val_len = np.concatenate((sequences_1_no_stopwords_len[idx_val], sequences_2_no_stopwords_len[idx_val]))
data_2_no_stopwords_val_len = np.concatenate((sequences_2_no_stopwords_len[idx_val], sequences_1_no_stopwords_len[idx_val]))
data_0_val_word_match = np.concatenate((word_match_train[idx_val], word_match_train[idx_val]))
data_0_no_stopwords_val_word_match = np.concatenate((word_match_train_no_stopwords[idx_val], word_match_train_no_stopwords[idx_val]))
data_0_stemmed_val_word_match = np.concatenate((word_match_train_stemmed[idx_val], word_match_train_stemmed[idx_val]))
data_0_no_stopwords_stemmed_val_word_match = np.concatenate((word_match_train_no_stopwords_stemmed[idx_val], word_match_train_no_stopwords_stemmed[idx_val]))
data_0_val_tfidf_word_match = np.concatenate((tfidf_word_match_train[idx_val], tfidf_word_match_train[idx_val]))
leaks_val = np.vstack((leaks[idx_val], leaks[idx_val]))
lemma_1_val = np.vstack((q1LemmaPad[idx_val], q2LemmaPad[idx_val]))
tag_1_val = np.vstack((q1TagPad[idx_val], q2TagPad[idx_val]))
pos_1_val = np.vstack((q1POSPad[idx_val], q2POSPad[idx_val]))
ner_1_val = np.vstack((q1NERPad[idx_val], q2NERPad[idx_val]))
lemma_2_val = np.vstack((q2LemmaPad[idx_val], q1LemmaPad[idx_val]))
tag_2_val = np.vstack((q2TagPad[idx_val], q1TagPad[idx_val]))
pos_2_val = np.vstack((q2POSPad[idx_val], q1POSPad[idx_val]))
ner_2_val = np.vstack((q2NERPad[idx_val], q1NERPad[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344


	
#======================== define the model structure =====================#

#embedding_layer = Embedding(nb_words,
#        EMBEDDING_DIM,
#        weights=[embedding_matrix],
#        input_length=MAX_SEQUENCE_LENGTH,
#        trainable=False)

glove_embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[glove_embedding_matrix840b],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)

NER_embedding = Embedding(10,
        3, input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)

POS_embedding = Embedding(45,
        5, input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)

Lemma_embedding = Embedding(100000,
        32, input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)


lstm_layer = LSTM(num_lstm, dropout_W=rate_drop_lstm, dropout_U=rate_drop_lstm)#recurrent_

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#embedded_sequences_1 = embedding_layer(sequence_1_input)
glove_embedded_sequences_1 = glove_embedding_layer(sequence_1_input)
spacy_Lemma_seq_1 =  Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
spacy_Tag_seq_1 =  Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#pacy_POS_seq_1 =  Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
spacy_NER_seq_1 =  Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
NER_embedded_1 = NER_embedding(spacy_NER_seq_1)
Lemma_embedded_1 = Lemma_embedding(spacy_Lemma_seq_1)
#POS_embedded_1 = POS_embedding(spacy_POS_seq_1)
Tag_embedded_1 = POS_embedding(spacy_Tag_seq_1)
#embedded_sequences_1 = merge([glove_embedded_sequences_1, spacy_Lemma_seq_1, spacy_Tag_seq_1, spacy_POS_seq_1, spacy_NER_seq_1], mode='concat', concat_axis=2)
embedded_sequences_1 = merge([glove_embedded_sequences_1, NER_embedded_1, Tag_embedded_1, Lemma_embedded_1], mode='concat', concat_axis=2)
#cnn_sequences_1 = Conv1D(filters = 32, kernel_size=1, padding='valid', activation='relu', strides=1)(embedded_sequences_1)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#embedded_sequences_2 = embedding_layer(sequence_2_input)
glove_embedded_sequences_2 = glove_embedding_layer(sequence_2_input)
spacy_Lemma_seq_2 =  Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
spacy_Tag_seq_2 =  Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#spacy_POS_seq_2 =  Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
spacy_NER_seq_2 =  Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
NER_embedded_2 = NER_embedding(spacy_NER_seq_2)
Lemma_embedded_2 = Lemma_embedding(spacy_Lemma_seq_2)
#POS_embedded_2 = POS_embedding(spacy_POS_seq_2)
Tag_embedded_2 = POS_embedding(spacy_Tag_seq_2)
#embedded_sequences_2 = merge([glove_embedded_sequences_2, spacy_Lemma_seq_2, spacy_Tag_seq_2, spacy_POS_seq_2, spacy_NER_seq_2], mode='concat', concat_axis=2)
embedded_sequences_2 = merge([glove_embedded_sequences_2, NER_embedded_2, Tag_embedded_2, Lemma_embedded_2], mode='concat', concat_axis=2)
#cnn_sequences_2 = Conv1D(filters = 32, kernel_size=1, padding='valid', activation='relu', strides=1)(embedded_sequences_2)
y1 = lstm_layer(embedded_sequences_2)

sequence_1_len_input = Input(shape=(1,), dtype='float32')
sequence_2_len_input = Input(shape=(1,), dtype='float32')
sequence_1_no_stopwords_len_input = Input(shape=(1,), dtype='float32')
sequence_2_no_stopwords_len_input = Input(shape=(1,), dtype='float32')
sequence_word_match_input = Input(shape=(1,), dtype='float32')
sequence_no_stopwords_word_match_input = Input(shape=(1,), dtype='float32')
sequence_stemmed_word_match_input = Input(shape=(1,), dtype='float32')
sequence_no_stopwords_stemmed_word_match_input = Input(shape=(1,), dtype='float32')
sequence_tfidf_word_match_input = Input(shape=(1,), dtype='float32')

leaks_input = Input(shape=(leaks.shape[1],))
leaks_dense = Dense(num_dense/2, activation=act)(leaks_input)

merged = merge([x1, y1], mode='concat')#concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
merged = merge([merged, sequence_1_len_input, sequence_2_len_input, sequence_1_no_stopwords_len_input, sequence_2_no_stopwords_len_input, sequence_word_match_input, sequence_no_stopwords_word_match_input, sequence_stemmed_word_match_input, sequence_no_stopwords_stemmed_word_match_input, sequence_tfidf_word_match_input, leaks_input], mode='concat')
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)


#=========================== add class weight =========================#
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None


#=========================== train the model =========================#

model = Model(input=[sequence_1_input, spacy_Tag_seq_1, spacy_Lemma_seq_1, spacy_NER_seq_1,\
		sequence_2_input, spacy_Tag_seq_2, spacy_Lemma_seq_2, spacy_NER_seq_2, \
		sequence_1_len_input, sequence_2_len_input, sequence_1_no_stopwords_len_input, sequence_2_no_stopwords_len_input, \
		sequence_word_match_input, sequence_no_stopwords_word_match_input, sequence_stemmed_word_match_input, sequence_no_stopwords_stemmed_word_match_input, sequence_tfidf_word_match_input, \
		leaks_input], output=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
model.summary()
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = BASE_DIR + 'models/' + STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)



hist = model.fit([data_1_train, tag_1_train, lemma_1_train, ner_1_train,\
		data_2_train, tag_2_train, lemma_2_train, ner_2_train,\
		data_1_train_len, data_2_train_len, data_1_no_stopwords_train_len, data_2_no_stopwords_train_len, \
		data_0_train_word_match, data_0_no_stopwords_train_word_match, data_0_stemmed_train_word_match, data_0_no_stopwords_stemmed_train_word_match, data_0_train_tfidf_word_match, \
		leaks_train], labels_train, \
		validation_data=([data_1_val, tag_1_val, lemma_1_val, ner_1_val,\
		data_2_val, tag_2_val, lemma_2_val, ner_2_val,\
		data_1_val_len, data_1_val_len, data_1_no_stopwords_val_len, data_2_no_stopwords_val_len, \
		data_0_val_word_match, data_0_no_stopwords_val_word_match, data_0_stemmed_val_word_match, data_0_no_stopwords_stemmed_val_word_match, data_0_val_tfidf_word_match, \
		leaks_val], labels_val, weight_val), \
		nb_epoch=200, batch_size=800, shuffle=True, \
		class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

#======================== make the submission ======================#

print('Start making the submission before fine-tuning')

preds = model.predict([test_data_1, q1testLemmaPad, q1testTagPad, q1testPOSPad, q1testNERPad, \
		test_data_2, q2testLemmaPad, q2testTagPad, q2testPOSPad, q2testNERPad, \
		test_sequences_1_len, test_sequences_2_len, test_sequences_1_no_stopwords_len, test_sequences_2_no_stopwords_len, \
		word_match_test, word_match_test_no_stopwords, word_match_test_stemmed, word_match_test_no_stopwords_stemmed, tfidf_word_match_test, \
		test_leaks], batch_size=1536, verbose=1)

		preds += model.predict([test_data_2, q2testLemmaPad, q2testTagPad, q2testPOSPad, q2testNERPad, \
		test_data_1, q1testLemmaPad, q1testTagPad, q1testPOSPad, q1testNERPad, \
		test_sequences_2_len, test_sequences_1_len, test_sequences_2_no_stopwords_len, test_sequences_1_no_stopwords_len, \
		word_match_test, word_match_test_no_stopwords, word_match_test_stemmed, word_match_test_no_stopwords_stemmed, tfidf_word_match_test, \
		test_leaks], batch_size=1536, verbose=1)

preds /= 2

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)