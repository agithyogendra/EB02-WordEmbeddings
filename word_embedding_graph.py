#from gensim.models import Word2Vec
import gensim.models.keyedvectors as Word2Vec
from gensim.similarities import WmdSimilarity
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

nltk.download("punkt")   # Download data for tokenizer.
stop_words = stopwords.words('english')

G = nx.Graph()
doc_corpus = []
w2v_corpus = []
path = 'clueweb09PoolFilesTest'  

def preprocess(doc):
    doc = doc.lower()  # Lower the text.
    doc = word_tokenize(doc)  # Split into words.
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    return doc
i = 0
#Get file names
for file in os.listdir(path):
    if i == 50:
        break
    f = open(path + '/' + file)
    doc_corpus.append(file)
    # Pre-process document.
    tmp = f.read() 
    text = preprocess(tmp)
    # Add to corpus for training Word2Vec.
    w2v_corpus.append(text)
    f.close()
    i = i + 1

if not os.path.exists('GoogleNews-vectors-negative300.bin'):
    raise ValueError("SKIP: You need to download the google news model")
start = time()
model = Word2Vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)    
model.init_sims(replace=True)
model.save('word_embeddings')

for i in range(len(doc_corpus)):
    for j in range(len(doc_corpus)):
        if i == j:
            continue
        G.add_edge(doc_corpus[i], doc_corpus[j], weight = Word2Vec.KeyedVectors.load('word_embeddings', mmap = 'r').wmdistance(w2v_corpus[i], w2v_corpus[j]))

graph_json = json_graph.node_link_data(G)
json.dump(graph_json, open('graphGood.json', 'w'), indent = 2)
print ('Cell took %.2f seconds to run.' %(time() - start))

