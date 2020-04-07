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

start = time()

nltk.download('wordnet')
nltk.download("punkt")   # Download data for tokenizer.
stop_words = stopwords.words('english')

porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

G = nx.Graph()
doc_corpus = []
w2v_corpus = []
path = 'clueweb09_50'  

def preprocess(doc):
    doc = doc.lower()  # Lower the text.
    doc = word_tokenize(doc)  # Split into words.
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation
    return ' '.join(doc)
    
#Get file names
for file in os.listdir(path):
    f = open(path + '/' + file)
    doc_corpus.append(file)
    # Pre-process document.
    html = f.read() 
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    text = porter.stem(text)
    text= lemmatizer.lemmatize(text)
    text = preprocess(text)
    # Add to corpus for training Word2Vec.
    w2v_corpus.append(text)
    f.close()

# train model
nlp = spacy.load('en_core_web_md')

for i in range(len(doc_corpus)):
    for j in range(len(doc_corpus)):
        if i == j or G.has_edge(doc_corpus[i], doc_corpus[j]):
            continue
        # Vectorize docs
        doc1 = nlp(w2v_corpus[i])
        doc2 = nlp(w2v_corpus[j])
        sim = doc1.similarity(doc2)
        G.add_edge(doc_corpus[i], doc_corpus[j], weight = sim)
graph_json = json_graph.node_link_data(G)
json.dump(graph_json, open('clueweb09_50.json', 'w'), indent = 2)
print ('Cell took %.2f seconds to run.' %(time() - start))

