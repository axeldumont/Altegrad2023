"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor


# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
hidden_dim = 32
output_dim = 4
dropout = 0.0
neighbor_aggr = 'mean'
readout = 'mean'


############## Task 4
        
##################
# your code here #
##################

Gs = list()
for i in range(10):
    n = 10+i
    G = nx.cycle_graph(n)
    Gs.append(G)

############## Task 5
        
##################
# your code here #
##################

# Create block diagonal adjacency matrix
adj_mat = sp.block_diag([nx.adjacency_matrix(G) for G in Gs])

features = np.ones((adj_mat.shape[0], 1))

idx_batch = np.concatenate([np.full(G.number_of_nodes(), i) for i,G in enumerate(Gs)])

adj_batch = sparse_mx_to_torch_sparse_tensor(adj_mat).to(device) 
features_batch = torch.from_numpy(features).to(device)
idx_batch = torch.from_numpy(idx_batch).to(device)

adj_batch = adj_batch.float()
features_batch = features_batch.float()
idx_batch = idx_batch.long()

############## Task 8
        
##################
# your code here #
##################
# Mean aggregation, mean readout 
model = GNN(1, hidden_dim, output_dim, 'mean', 'mean', dropout).to(device)

out = model(features_batch, adj_batch, idx_batch)
print(out)

# Sum aggregation, sum readout
model = GNN(1, hidden_dim, output_dim, 'sum', 'sum', dropout).to(device) 

out = model(features_batch, adj_batch, idx_batch)
print(out)

# Sum aggregation, mean readout
model = GNN(1, hidden_dim, output_dim, 'sum', 'mean', dropout).to(device)

out = model(features_batch, adj_batch, idx_batch) 
print(out)

# Mean aggregation, sum readout  
model = GNN(1, hidden_dim, output_dim, 'mean', 'sum', dropout).to(device)

out = model(features_batch, adj_batch, idx_batch)
print(out)



############## Task 9
        
##################
# your code here #
##################
G1 = nx.Graph()

"""
# Add nodes to the graph
G1.add_nodes_from(range(6))  # 6 total nodes

# Add edges to form the two cycle graphs
G1.add_edges_from([(0, 1), (1, 2), (2, 0)])  # First cycle graph
G1.add_edges_from([(3, 4), (4, 5), (5, 3)])  # Second cycle graph
"""

G1 = nx.cycle_graph(4)
G1 = nx.disjoint_union(G1,nx.cycle_graph(4))

G2 = nx.cycle_graph(8)

############## Task 10
        
##################
# your code here #
##################

# Create block diagonal adjacency matrix
adj_mat = sp.block_diag([nx.adjacency_matrix(G1), nx.adjacency_matrix(G2)])

features = np.ones((adj_mat.shape[0], 1))  

idx_batch = np.concatenate([np.zeros(8), np.ones(8)])

adj_batch = sparse_mx_to_torch_sparse_tensor(adj_mat).to(device)
features_batch = torch.from_numpy(features).to(device)
idx_batch = torch.from_numpy(idx_batch).to(device)

adj_batch = adj_batch.float()
features_batch = features_batch.float()
idx_batch = idx_batch.long()


############## Task 11
        
##################
# your code here #
##################

model = GNN(1, hidden_dim, output_dim, 'sum', 'sum', dropout).to(device)
out = model(features_batch, adj_batch, idx_batch)
print('\n')
print('\n')
print(out)