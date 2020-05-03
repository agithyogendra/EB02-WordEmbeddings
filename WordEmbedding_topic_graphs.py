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
import spacy
from spacy.lang.en import English
from multiprocessing.dummy import Pool as ThreadPool

start_project = time()
path = 'cw09_pool/clueweb09PoolFiles' 
sdm_results = 'result_TF_IDF_full_corpus.txt'
output = []
nltk.download('wordnet')
nltk.download("punkt")   # Download data for tokenizer.
stop_words = stopwords.words('english')

def preprocess(doc):
  doc = doc.lower()  # Lower the text.
  doc = word_tokenize(doc)  # Split into words.
  doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
  doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation
  return ' '.join(doc)

def vectorize(doc_names):
  #Get file names
  graph_docs = {}
  start_vec = time()
  for doc_name in doc_names:
    f = open(path + '/' + doc_name[1])
    # Pre-process document.
    html = f.read() 
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    text = preprocess(text)
    text_vector = nlp(text)
    # Add to corpus for training Word2Vec.
    graph_docs[doc_name[1]] = text_vector
    f.close()
  print("Done Vectorization of graph: " + str(doc_names[0][0])+ ". Took " + str(time() - start_vec) + " s")
  return graph_docs

def get_doc_pairings(corpus):
  parings = []
  for doc_i in corpus:
    for doc_j in corpus:
      if doc_i == doc_j :
        continue
      parings.append(((doc_i, corpus[doc_i]), (doc_j, corpus[doc_j])))
  return parings

def get_edge(curr_pairing):
  process_start = time()
  doc1_name = curr_pairing[0][0]
  doc2_name = curr_pairing[1][0]
  print("Process Start: " + doc1_name + " , " + doc2_name)
  doc1_vector = curr_pairing[0][1]
  doc2_vector = curr_pairing[1][1]
  sim = doc1_vector.similarity(doc2_vector)
  edge = (doc1_name, doc2_name, {'weight' : sim})
  print("Process End: " + doc1_name + " , " + doc2_name + ". Took " + str(time() - process_start) + " s")
  return edge

# train model
start_load = time()
nlp = spacy.load('en_core_web_md')
print("Model Done loading: " + str(time() - start_load) +  " s")

# Get results
with open(sdm_results) as f:
    rankings = [line.rstrip('\n') for line in f]

sdm_docs = {}
# Map topic number to list of docs
for rank in rankings:
  split = rank.split()
  topic_number = split[0]
  doc_name = split[2]
  if topic_number not in sdm_docs:
    sdm_docs[topic_number] = []
  sdm_docs[topic_number].append((topic_number,doc_name))

pathSub = 'subgraphs/'
pool = ThreadPool(30)
for topic_number in range(1, 201):
  if str(topic_number) not in sdm_docs:
    continue
  subgraph = nx.Graph()
  subgraph_docs = vectorize(sdm_docs[str(topic_number)])
  pairings = get_doc_pairings(subgraph_docs)
  edge = pool.imap(get_edge, pairings)
  subgraph.add_edges_from(edge)
  graph_json = json_graph.node_link_data(subgraph)
  json.dump(graph_json, open(pathSub + 'SubGraph' + str(topic_number) + '.json', 'w'), indent = 2)
print ("Results took %.2f seconds to run." %(time() - start_project))
