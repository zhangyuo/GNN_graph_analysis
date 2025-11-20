import math
import os
import sys

res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(os.path.dirname(res))
sys.path.insert(0, base_path)
import networkx as nx
import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset
from orbit_table_generator import generate_orbit_tables_from_count
import orca
from torch_geometric.utils import to_networkx

# from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from torch_geometric.data import DataLoader

if __name__ == "__main__":
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=base_path + '/dataset/')
    dataset.get_idx_split()
    edge_index = dataset[0].edge_index

    G = nx.Graph()

    # Add edges to the graph
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        G.add_edge(src, dst)

    print("Generating graph")
    G = to_networkx(dataset[0], to_undirected=True)
    print("Done generating graph")
    print(len(list(G.nodes)))

    orbit_counts = orca.orbit_counts("node", 5, G)
    print("Done counting orbit")
    orbit_df = generate_orbit_tables_from_count(orbit_counts, sorted(list(G.nodes)))
    orbit_df.to_csv(base_path + "/dataset/orbit/arxiv_orbit_df.csv")
