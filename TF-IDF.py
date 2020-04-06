from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import os
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize
from bs4 import BeautifulSoup
from os import listdir
from time import time
import json
from bs4 import BeautifulSoup
import math
import numpy as np
from collections import OrderedDict
import copy

nltk.download('stopwords')   # Download data for tokenizer.
stop_words = stopwords.words('english')

corpus_path = 'EB02-WordEmbeddings\clueweb09_50'  
topics_path = 'EB02-WordEmbeddings\\topics.txt'

doc_corpus = []
corpus = []
topics = {}
doc_tokens = []

def preprocess(doc):
  doc = doc.lower()  # Lower the text.
  doc = word_tokenize(doc)  # Split into words.
  doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
  doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
  return ' '.join(doc)

def cosine_sim(vec1, vec2):
  vec1 = [val for val in vec1.values()]
  vec2 = [val for val in vec2.values()]
  
  dot_prod = 0
  for i, v in enumerate(vec1):
    dot_prod += v * vec2[i]
  mag_1 = math.sqrt(sum([x**2 for x in vec1]))
  mag_2 = math.sqrt(sum([x**2 for x in vec2]))
  return dot_prod / (mag_1 * mag_2)

def queryVectorizer(query):
  query_vec = copy.copy(zero_vector)
  tokens = word_tokenize(query)
  token_counts = Counter(tokens)
  for key, value in token_counts.items():
    docs_containing_key = 0
    for _doc in corpus:
      if key in _doc:
        docs_containing_key += 1
    tf = value / len(tokens)
    if docs_containing_key > 0:
      idf = len(corpus) / docs_containing_key
    else:
      idf = 0
    query_vec[key] = tf * idf
  return query_vec

# Retrieve docs
for file in os.listdir(corpus_path):
  f = open(corpus_path + '/' + file)
  doc_corpus.append(file)
  # Pre-process document.
  html = f.read() 
  soup = BeautifulSoup(html, "html.parser")
  text = soup.get_text()
  text = preprocess(text)
  doc_tokens += [sorted(word_tokenize(text))]
  # Add to corpus for training Word2Vec.
  corpus.append(text)
  f.close()

lexicon = sorted(set(sum(doc_tokens, [])))
zero_vector = OrderedDict((token, 0) for token in lexicon)
document_tfidf_vectors = []

#TF-IDF vectorization
for doc in corpus:
  # Pre-process document.
  vec = copy.copy(zero_vector)
  token_counts = Counter(text)
  for key, value in token_counts.items():
    docs_containing_key = 0
    for _doc in corpus:
      if key in _doc:
        docs_containing_key += 1
    tf = value / len(lexicon)
    if docs_containing_key > 0:
      idf = len(corpus) / docs_containing_key
    else:
      idf = 0
    vec[key] = tf * idf
  document_tfidf_vectors.append(vec)

# Get topics
with open(topics_path) as f:
    lines = [line.rstrip('\n') for line in f]
for line in lines:
  content = line.split(':')
  tn = content[0]
  tc = content[1]
  topics[tn] = tc

globalRanking = []
for key, value in topics.items():
  rankings = {}
  query_vec = queryVectorizer(value)
  for i in range(len(corpus)):
    try:
      diff = cosine_sim(query_vec, document_tfidf_vectors[i])
    except ZeroDivisionError:
      diff = 0
    if diff > 0:
     rankings[doc_corpus[i]] = diff
  doc_count = 0
  for rankingsKey, rankingsValue in sorted(rankings.items(), key=lambda item: item[1], reverse = True):
    if doc_count == 20:
      break
    globalRanking.append(str(key) + " Q0 " + rankingsKey + " " + str(doc_count) + " " + str(rankingsValue))
    doc_count += 1

with open("TFIDF.txt", "w") as outfile:
    outfile.write("\n".join(globalRanking))

