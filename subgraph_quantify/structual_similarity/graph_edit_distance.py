#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/23 14:18
# @Author   : **
# @Email    : **@**
# @File     : graph_edit_distance.py
# @Software : PyCharm
# @Desc     :
"""
import time

import networkx as nx


def node_match(n1, n2):
    """
    define node attribute match function
    :param n1:
    :param n2:
    :return:
    """
    return n1.get('id') == n2.get('id')


def compute_graph_edit_distance(G1, G2):
    start_time = time.time()

    # add node id into subgraph
    for id in G1.nodes:
        G1.nodes[id]['id'] = id
    for id in G2.nodes:
        G2.nodes[id]['id'] = id

    ged_gen = nx.optimize_graph_edit_distance(
        G1,
        G2,
        node_match=node_match,
        edge_match=None
    )

    # get first approximate ged
    approx_ged = next(ged_gen)
    print(f"approximate GED: {approx_ged}")

    # normalize the ged
    # longest_graph_size = max(len(G1.nodes) + len(G1.edges), len(G2.nodes) + len(G2.edges))
    longest_graph_size = len(G1.nodes) + len(G1.edges) + len(G2.nodes) + len(G2.edges)
    ged = round(approx_ged / longest_graph_size, 2)

    elapsed = time.time() - start_time
    print(f"graph edit distance computed in {elapsed:.4f}s!")

    return ged


if __name__ == '__main__':
    # 1. create two graph and ensure same node number or use different node number
    G1 = nx.cycle_graph(10)  # 4-nodes cycle graph
    G2 = nx.path_graph(10)  # 4-nodes path graph

    # compute ged
    ged = nx.graph_edit_distance(G1, G2)
    print(f"ged = {ged}")

    # 2. compute ged with node attribute
    G1 = nx.cycle_graph(4)  # 4-nodes cycle graph
    G2 = nx.path_graph(4)  # 4-nodes path graph
    G1.add_node(0, label='A')
    G1.add_node(1, label='C')
    G2.add_node(0, label='B')
    G2.add_node(1, label='C')
    ged_custom = nx.graph_edit_distance(G1, G2, node_match=node_match)
    print(f"ged_custom={ged_custom}")

    # 3. compute ged with edge attribute
    G1 = nx.cycle_graph(4)  # 4-nodes cycle graph
    G2 = nx.path_graph(4)  # 4-nodes path graph
    G1.add_edge(0, 1, weight=1)
    G2.add_edge(0, 1, weight=2)
    ged_custom = nx.graph_edit_distance(G1, G2, edge_match=lambda e1, e2: e1.get('weight') == e2.get('weight'))
    print(f"ged_custom={ged_custom}")

    # 4. compute ged with node and edge attribute
    G1 = nx.cycle_graph(4)  # 4-nodes cycle graph
    G2 = nx.path_graph(4)  # 4-nodes path graph
    G1.add_node(2, label='B')
    G2.add_node(2, label='A')
    G1.add_edge(0, 1, weight=1)
    G2.add_edge(0, 1, weight=2)
    ged_custom = nx.graph_edit_distance(G1, G2, node_match=lambda n1, n2: n1.get('label') == n2.get('label'),
                                        edge_match=lambda e1, e2: e1.get('weight') == e2.get('weight'))
    print(f"ged_custom={ged_custom}")
