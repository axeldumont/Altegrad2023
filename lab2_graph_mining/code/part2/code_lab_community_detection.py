"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):

    # Get adjacency matrix
    A = nx.to_numpy_array(G)

    # Compute normalized Laplacian
    n = A.shape[0]
    D = np.diag(np.sum(A, axis=1))
    D = np.array(D)
    D_inv = np.linalg.inv(D)
    D_inv = np.array(D_inv)
    L = np.identity(n) - (D_inv @ A)
    # Get eigenvectors
    _, eigenvectors = eigs(L, k=k, which='SR')
    eigenvectors = eigenvectors.real
    

    # Cluster rows of U with k-means 
    km = KMeans(n_clusters=k)
    km.fit(eigenvectors)
    
    clustering = dict()
    for i,node in enumerate(G.nodes()):
        clustering[node] = km.labels_[i]
    
    return clustering





############## Task 7

G = nx.read_edgelist("./datasets/CA-HepTh.txt", delimiter='\t')
CC = nx.connected_components(G) 
gcc_nodes = max(CC, key=len)
gcc = G.subgraph(gcc_nodes)
map = spectral_clustering(G, 50)




############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    modularity = 0
    clusters = set(clustering.values())
    m = G.number_of_edges()
    
    for cluster in clusters:
        nodes_in_clusters = [node for node in G.nodes() if clustering[node]==cluster]
        
        subG = G.subgraph(nodes_in_clusters)
        l_c = subG.number_of_edges()
        
        d_c = 0
        for node in nodes_in_clusters:
            d_c +=G.degree(node)
        
        modularity += (l_c/m) - (d_c/(2*m))**2
    ##################
    
    return modularity



############## Task 9

print("Modularity spectral clustering:", modularity(gcc, map))

random_clustering = dict()
for node in G.nodes():
    random_clustering[node] = randint(0,49)
    
print("Modularity random clustering:", modularity(gcc, random_clustering))







