import networkx as nx
from networkx.readwrite import json_graph
import json
from multiprocessing.dummy import Pool as ThreadPool
from time import time

sdm_results = 'results.txt'
start_project = time()
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

graph = getGraph('clueweb09PoolFilesTest.json')
threshold = get_threshold(graph)
scores = {}
# Return top 20 ranked documents from subgraph
def subgraphResults(sdm_result):
  process_start = time()
  topic_number = sdm_result[0]
  doc_name = sdm_result[1]
  print("Process Start: " + doc_name)
  G = nx.Graph()
  top_results = {}
  for edge in list(nx.bfs_edges(graph, source=doc_name)):
    weight = graph[edge[0]][edge[1]]['weight']
    if weight > threshold and weight > 0:
      G.add_edge(edge[0], edge[1], weight=weight)
  betweenness_centrality = nx.betweenness_centrality(G, normalized=False)
  doc_count = 20
  for doc, score in sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse = True):
    if doc_count == 0:
      break
    top_results[doc] = score
    if doc in scores[topic_number] and score < scores[topic_number][doc]:
      continue
    scores[topic_number][doc] = score
    doc_count = doc_count - 1
  print("Process End: " + doc_name + ". Took " + str(time() - process_start) + " s")

# Get results
with open(sdm_results) as f:
    rankings = [line.rstrip('\n') for line in f]

sdm_docs = {}
output = []

sdm_count = 1
# Map topic number to list of docs
for rank in rankings:
  split = rank.split()
  topic_number = split[0]
  doc_name = split[2]
  if topic_number not in sdm_docs:
    sdm_docs[topic_number] = []
  sdm_docs[topic_number].append((topic_number,doc_name))

start_load = time()
pool = ThreadPool(30)
for topic_number in sdm_docs:
  if topic_number not in scores:
    scores[topic_number] = {}
  subgraph = pool.map(subgraphResults, sdm_docs[topic_number])
print("Subgraphs completed: " + str(time() - start_load) +  "s")

for topic_number in scores:
  cur_topic_subgraphs = scores[topic_number]
  doc_count = 1
  for doc, score in sorted(cur_topic_subgraphs.items(), key=lambda item: item[1], reverse = True):
    if doc_count > 20:
      break
    output.append(topic_number + " Q0 " + doc + " " + str(doc_count) + " " + str(normalize(score, cur_topic_subgraphs.values()))+ " STANDARD")
    doc_count += 1

with open("results_file.test", "w") as outfile:
    outfile.write("\n".join(output))
print ("Results took %.2f seconds to run." %(time() - start_project))
