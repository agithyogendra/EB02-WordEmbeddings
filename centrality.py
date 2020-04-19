import networkx as nx
from networkx.readwrite import json_graph
import json
from decimal import Decimal

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

# Return top 20 ranked documents from subgraph
def subgraphResults(doc_name, graph, threshold):
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
    doc_count = doc_count - 1
  return top_results

sdm_results = 'EB02-WordEmbeddings\TF_Results.txt'
G = getGraph('EB02-WordEmbeddings\clueweb09PoolFilesTest.json')
threshold = get_threshold(G)

# Get results
with open(sdm_results) as f:
    rankings = [line.rstrip('\n') for line in f]
output = []
scores = {}
# Get query i sdm results
prev_topic_number = 0
sdm_count = 1
for rank in rankings:
  split = rank.split()
  topic_number = split[0]
  if prev_topic_number == 0:
    prev_topic_number = topic_number
  doc_name = split[2]
  if prev_topic_number != topic_number or sdm_count == len(rankings):
    doc_count = 1
    for doc, score in sorted(scores.items(), key=lambda item: item[1], reverse = True):
      if doc_count > 20:
        break
      output.append(prev_topic_number + " Q0 " + doc + " " + str(doc_count) + " " + str(normalize(score, scores.values()))+ " STANDARD")
      doc_count += 1
    scores = {}
    prev_topic_number = topic_number
  print("Current Doc: ", doc_name)
  try:
    subgraph = subgraphResults(doc_name, G, threshold)
  except:
    print("Document does not exist!")
    continue
  for doc, score in subgraph.items():
    scores[doc] = score
    print(score)
  sdm_count += 1
  print(sdm_count)

print(len(rankings))
with open("results_file.test", "w") as outfile:
    outfile.write("\n".join(output))