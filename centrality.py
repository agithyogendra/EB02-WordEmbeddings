import networkx as nx
from networkx.readwrite import json_graph
import json

def getGraph(filename):
  with open(filename) as f:
    js_graph = json.load(f)
  return json_graph.node_link_graph(js_graph)

G1 = getGraph('G1.json')

G2 = nx.Graph()
G3 = nx.Graph()

sum = 0
count = 1
for u,v,weight in G1.edges.data('weight'):
  weight = abs(1 - weight)
  G2.add_edge(u,v,weight = weight)
  sum = sum + weight
  count = count + 1

graph_json = json_graph.node_link_data(G2)
json.dump(graph_json, open('G2.json', 'w'), indent = 2)

threshold = sum/count
print('Threshold', threshold)
for u,v,weight in G2.edges.data('weight'):
  if weight < threshold:
    G3.add_edge(u,v,weight = 1/weight)

betweenness_centrality = nx.betweenness_centrality(G3,k = None, normalized=True, seed=None)
doc_count = 20
for key, value in sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse = True):
  if doc_count == 0:
    break
  print("%s: %s" % (key,value))
  doc_count = doc_count - 1
