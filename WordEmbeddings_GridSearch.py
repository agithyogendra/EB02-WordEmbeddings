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
from multiprocessing import Pool
import pytrec_eval

start_project = time()

path_subgraph = 'subgraphs'
sdm_results = 'result_TF_IDF.txt'
output = []
scores = {}
nltk.download('wordnet')
nltk.download("punkt")   # Download data for tokenizer.
stop_words = stopwords.words('english')

# Get graph from json file 
def getGraph(filename):
  with open(filename) as f:
    js_graph = json.load(f)
  return json_graph.node_link_graph(js_graph)

def normalize(score, scores):
  normal = (score - min(scores))/(max(scores) - min(scores))
  return normal

# Get Threshold
def get_threshold(G):
  weight_sum = 0
  count = 0
  for u,v,weight in G.edges.data('weight'):
    if weight > 0:
      count += 1
      weight_sum += weight
  return weight_sum/count

def get_scores(doc_contents):
  sdm_scores = {}
  for element in doc_contents:
    sdm_scores[element[1]] = element[2]
  return sdm_scores

# Return top 20 ranked documents from subgraph
def subgraphResults(topic_number):
  process_start = time()
  try:  
    subgraph = getGraph(path_subgraph + '/' + 'SubGraph' + topic_number + '.json')
    threshold = get_threshold(subgraph)
  except:
    print("Error: " + topic_number)
    return
  print("Process Start Topic: " + topic_number)
  top_results = {}
  G = nx.Graph()
  for u,v,weight in subgraph.edges.data('weight'):
    if weight > 0 and weight > threshold:
      G.add_edge(u, v, weight=weight)
  betweenness_centrality = nx.betweenness_centrality(G, normalized=False)
  doc_count = 20
  for doc, score in sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse = True):
    if doc_count == 0:
      break
    scores[topic_number][doc] = score
    doc_count = doc_count - 1
  print("Process End Topic: " + topic_number + ". Took " + str(time() - process_start) + " s")

# Get results
with open(sdm_results) as f:
    rankings = [line.rstrip('\n') for line in f]

sdm_docs = {}
start_result_retrieval = time()
# Map topic number to list of docs
for rank in rankings:
  split = rank.split()
  topic_number = split[0]
  doc_name = split[2]
  sdm_value = split[4]
  if topic_number not in sdm_docs:
    sdm_docs[topic_number] = []
  sdm_docs[topic_number].append((topic_number,doc_name, sdm_value))
print("Result Retrieval Completed " + str(time() - start_result_retrieval) +  " s")

start_load = time()
for topic_number in sdm_docs:
  if topic_number not in scores:
    scores[topic_number] = {}
  subgraphResults(topic_number)
print("Subgraphs completed: " + str(time() - start_load) +  "s")
#pool.join()
best_score = -1
best_results = None
for lambda1 in [x/10 for x in range(1, 11)]:
  start_test = time()
  lambda2 = 1 - lambda1
  lambda2 = round(lambda2, 1)
  temp_results = []
  for topic_number in range(1, 201):
    if str(topic_number) not in scores:
      continue
    sdm_doc_score_map = get_scores(sdm_docs[str(topic_number)])
    cur_topic_subgraph = scores[str(topic_number)]
    doc_count = 1
    for doc, score in sorted(cur_topic_subgraph.items(), key=lambda item: item[1], reverse = True):
      sdm_score = float(sdm_doc_score_map[doc])
      try:
        centrality_score = normalize(score, cur_topic_subgraph.values())
      except:
        centrality_score = 0
      combined_score = lambda1*sdm_score + lambda2*centrality_score
      temp_results.append(str(topic_number) + " Q0 " + doc + " " + str(doc_count) + " " + str(combined_score)+ " STANDARD")
      doc_count+=1
  with open("temp_file.test", "w") as outfile:
    outfile.write("\n".join(temp_results))
  run = pytrec_eval.TrecRun('temp_file.test')
  qrels = pytrec_eval.QRels('qrels_file.test')
  curr_result = pytrec_eval.evaluate(run, qrels, [pytrec_eval.ndcg])[0]
  if curr_result > best_score:
    if best_results is not None:
      best_results.clear()
    best_score = curr_result
    best_results = list(temp_results)
  print("Run completed with lambda1=" + str(lambda1) + ", lambda2=" + str(lambda2) + " and NDCG=" + str(curr_result) + ". Took: " + str(time() - start_load) +  " s")

for result in best_results:
  output.append(result)
with open("results_file.test", "w") as outfile:
    outfile.write("\n".join(output))
print ("Results took %.2f seconds to run." %(time() - start_project))

