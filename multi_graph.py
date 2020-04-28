import os
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize
import re
import string
import copy
from bs4 import BeautifulSoup
from os import listdir
from time import time
import networkx as nx
import matplotlib
from networkx.readwrite import json_graph
import json
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import spacy
from spacy.lang.en import English
from multiprocessing.dummy import Pool as ThreadPool

start_project = time()

nltk.download('wordnet')
nltk.download("punkt")   # Download data for tokenizer.
stop_words = stopwords.words('english')

porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

corpus = {}
path = 'clueweb09_50'  

def preprocess(doc):
  doc = doc.lower()  # Lower the text.
  doc = word_tokenize(doc)  # Split into words.
  doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
  doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation
  return ' '.join(doc)

def get_edge(curr_pairing):
  process_start = time()
  doc1_name = curr_pairing[0][0]
  doc2_name = curr_pairing[1][0]
  print("Process Start: " + doc1_name + " , " + doc2_name)
  doc1_vector = curr_pairing[0][1]
  doc2_vector = curr_pairing[1][1]
  sim = doc1_vector.similarity(doc2_vector)
  edge = (doc1_name, doc2_name, {'weight' : sim})
  print("Process End: " + doc1_name + " , " + doc2_name + ". Took " + str(time() - process_start) + "s")
  return edge

def get_doc_pairings(corpus):
  parings = []
  for doc_i in corpus:
    for doc_j in corpus:
      if doc_i == doc_j :
        continue
      parings.append(((doc_i, corpus[doc_i]), (doc_j, corpus[doc_j])))
  return parings

# train model
start_load = time()
nlp = spacy.load('en_core_web_md')
print("Model Done loading: " + str(time() - start_load) +  "s")

#Get file names
start_vec = time()
for file in os.listdir(path):
  f = open(path + '/' + file)
  # Pre-process document.
  html = f.read() 
  soup = BeautifulSoup(html, "html.parser")
  text = soup.get_text()
  text = porter.stem(text)
  text= lemmatizer.lemmatize(text)
  text = preprocess(text)
  text_vector = nlp(text)
  # Add to corpus for training Word2Vec.
  corpus[file] = text_vector
  f.close()
print("Done Vectorization: " + str(time() - start_vec) + "s")

G = nx.Graph()
pairings = get_doc_pairings(corpus)
pool = ThreadPool(30)
edge = pool.imap(get_edge, pairings)
G.add_edges_from(edge)
graph_json = json_graph.node_link_data(G)
json.dump(graph_json, open('clueweb09_multi.json', 'w'), indent = 2)
print ("Graph took %.2f seconds to run." %(time() - start_project))

