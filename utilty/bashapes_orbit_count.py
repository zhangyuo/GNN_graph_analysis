import io
import math
import os
import pickle
import sys

import networkx as nx
import pandas as pd
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from utilty.orbit_table_generator import OrbitTableGenerator, generate_orbit_tables_from_count
import utilty.orca as orca
from torch_geometric.utils import to_networkx

# from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from torch_geometric.data import DataLoader

res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(os.path.dirname(res))
sys.path.insert(0, base_path)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


dataset_path = base_path + "/dataset"
# Create PyG Data object
with open(dataset_path + "/TreeCycle.pickle", "rb") as f:
    pyg_data = CPU_Unpickler(f).load()
edge_index = pyg_data.edge_index

G = nx.Graph()

# Add edges to the graph
for i in range(edge_index.size(1)):
    src = edge_index[0, i].item()
    dst = edge_index[1, i].item()
    G.add_edge(src, dst)

print("Generating graph")
G = to_networkx(pyg_data, to_undirected=True)
print("Done generating graph")
print(len(list(G.nodes)))

orbit_counts = orca.orbit_counts("node", 5, G)
print("Done counting orbit")
orbit_df = generate_orbit_tables_from_count(orbit_counts, sorted(list(G.nodes)))
orbit_df.to_csv(dataset_path + "/orbit/TreeCycle_orbit_df.csv")
