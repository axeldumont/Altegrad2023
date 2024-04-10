"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

G = nx.read_edgelist("./datasets/CA-HepTh.txt", delimiter='\t')
print("Nodes in the graph: ", G.number_of_nodes())
print("Edges in the graph: ", G.number_of_edges())
print("\n")

############## Task 2
cc = nx.connected_components(G)
length = [len(c) for c in sorted(cc, key=len, reverse=True)]
largest_cc = max(nx.connected_components(G), key=len)
S = G.subgraph(largest_cc)
print('Number of nodes in the largest connected component: ', S.number_of_nodes())
print("Fraction of nodes: ", S.number_of_nodes()/G.number_of_nodes())
print('Number of edges in the largest connected component: ', S.number_of_edges())
print("Fraction of edges: ", S.number_of_edges()/G.number_of_edges())
print("\n")


############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

min_deg = np.min(degree_sequence)
max_deg = np.max(degree_sequence)
mean_deg = np.mean(degree_sequence)
med_deg = np.median(degree_sequence)

print("Degree minimum: ", min_deg)
print("Degree maximum: ", max_deg)
print("Degree mean: ", mean_deg)
print("Degree median: ", med_deg)
print("\n")


############## Task 4

hist = nx.degree_histogram(G)

# Regular histogram
plt.subplot(1, 2, 1) 
plt.bar(range(len(hist)), hist, width=1)
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution')

# Log-log plot
x = np.log(range(1, len(hist)+1))
y = np.log(hist)
plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.xlabel('log(Degree)')
plt.ylabel('log(Frequency)')
plt.title('Log-Log Plot')

plt.show()




############## Task 5

g_cluster_coef = nx.transitivity(G)

print("\nGlobal clustering coefficient: ", g_cluster_coef)
