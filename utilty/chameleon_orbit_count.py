import math
import os
import sys

res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(os.path.dirname(res))
sys.path.insert(0, base_path)
import networkx as nx
import pandas as pd
from torch_geometric.datasets import WikipediaNetwork
from orbit_table_generator import generate_orbit_tables_from_count
import orca
from torch_geometric.utils import to_networkx, to_undirected, remove_self_loops

if __name__ == "__main__":
    dataset = WikipediaNetwork(root=base_path + '/dataset', name='chameleon')
    data = dataset[0]
    edge_index = data.edge_index

    edge_set = set((u.item(), v.item()) for u, v in edge_index.t())
    is_symmetric = all((v, u) in edge_set for (u, v) in edge_set)
    print(f"Edge index is symmetric: {is_symmetric}")
    if not is_symmetric:
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index)

    # Add edges to the graph
    edge_list = edge_index.t().cpu().numpy().tolist()
    unique_edges = set()
    for src, dst in edge_list:
        if src != dst:
            unique_edges.add(tuple(sorted((src, dst))))

    G = nx.Graph()
    print("Generating graph")
    G.add_edges_from(unique_edges)
    # G = to_networkx(dataset[0], to_undirected=True)
    print("Done generating graph")
    print(G.number_of_nodes())
    print(G.number_of_edges())

    orbit_counts = orca.orbit_counts("node", 5, G)
    print("Done counting orbit")
    orbit_df = generate_orbit_tables_from_count(orbit_counts, sorted(list(G.nodes)))
    orbit_df.to_csv(base_path + "/dataset/orbit/chameleon_orbit_df.csv")
