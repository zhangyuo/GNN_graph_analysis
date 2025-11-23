#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/23 14:19
# @Author   : **
# @Email    : **@**
# @File     : maximum_common_subgraph.py
# @Software : PyCharm
# @Desc     : Computes and visualizes Maximum Common Subgraph (MCS) with enhanced labeling
"""
import networkx as nx
import time
import itertools


def is_isomorphic_with_labels(subgraph1, subgraph2):
    """Check if two subgraphs are isomorphic considering node labels"""
    return nx.is_isomorphic(
        subgraph1,
        subgraph2,
        node_match=lambda n1, n2: n1.get('label') == n2.get('label')
    )


def backtrack_mcs(G1, G2, current_mapping, best_mapping, best_size):
    """Backtracking search for Maximum Common Subgraph"""
    if len(current_mapping) > best_size[0]:
        best_size[0] = len(current_mapping)
        best_mapping.clear()
        best_mapping.update(current_mapping)

    unmapped_G1 = [n for n in G1.nodes() if n not in current_mapping]
    unmapped_G2 = [n for n in G2.nodes() if n not in current_mapping.values()]

    for u in unmapped_G1:
        for v in unmapped_G2:
            if G1.nodes[u].get('label') != G2.nodes[v].get('label'):
                continue

            valid = True
            for mapped_u, mapped_v in current_mapping.items():
                if (G1.has_edge(u, mapped_u) != G2.has_edge(v, mapped_v)) or \
                        (G1.has_edge(mapped_u, u) != G2.has_edge(mapped_v, v)):
                    valid = False
                    break

            if not valid:
                continue

            current_mapping[u] = v
            backtrack_mcs(G1, G2, current_mapping, best_mapping, best_size)
            del current_mapping[u]


def maximum_common_subgraph_match_node_id(G1, G2):
    """Computes Maximum Common Subgraph (MCS) between two graphs"""
    best_mapping = {}
    best_size = [0]
    start_time = time.time()
    backtrack_mcs(G1, G2, {}, best_mapping, best_size)
    elapsed = time.time() - start_time

    mcs_nodes = list(best_mapping.keys())
    print(f"[MCS found in {elapsed:.4f}s] Size: {len(mcs_nodes)} nodes")
    return G1.subgraph(mcs_nodes), best_mapping


def maximum_common_subgraph(G1, G2, target_node=None):
    """计算最大公共子图（MCS），要求节点ID和标签属性必须匹配"""
    start_time = time.time()
    # Step 1: Screen valid candidate nodes (must meet both ID and label matching)
    valid_nodes = []
    for node in G1.nodes():
        if node in G2.nodes():  # The node ID must exist in both graphs
            # if G1.nodes[node].get('label') == G2.nodes[node].get('label'):
            #     valid_nodes.append(node)
            valid_nodes.append(node)  # do not care node attribution (label)

    # Step 2: Build consistency graph (H)
    H = nx.Graph()
    H.add_nodes_from(valid_nodes)

    # Add edges (need to satisfy edge consistency in both directions)
    for u, v in itertools.combinations(valid_nodes, 2):
        # # Check the consistency of bidirectional edges
        # edge_forward = G1.has_edge(u, v) == G2.has_edge(u, v)
        # edge_backward = G1.has_edge(v, u) == G2.has_edge(v, u)
        # if edge_forward and edge_backward:
        #     H.add_edge(u, v)
        # undirected graph
        if G1.has_edge(u, v) and G2.has_edge(u, v):
            H.add_edge(u, v)

    # step 3: compute size of mcs
    mcs_size = len(H.nodes) + len(H.edges)
    longest_graph_size = max(len(G1.nodes) + len(G1.edges), len(G2.nodes) + len(G2.edges))
    mcs = round(mcs_size / longest_graph_size, 2)

    # Step 4: Find the largest common subgraph, determine connectivity, and find the largest connected subgraph
    if H.number_of_edges() == 0:  # Handling the case of boundless graphs
        if valid_nodes:
            if target_node in valid_nodes:
                best_component = {target_node}  # Include the target node first
            else:
                best_component = {valid_nodes[0]}  # Second choice: the first node of valid_nodes
        else:
            best_component = set()  # Returns an empty set when there are no valid nodes.
    else:
        # Get all connected components and choose the largest
        components = list(nx.connected_components(H))
        best_component = max(components, key=len) if components else set()

    best_connected_common_subgraph = H.subgraph(best_component)
    mapping = {node: node for node in best_component}  # Node ID maps to itself

    # Step 4: Output results
    elapsed = time.time() - start_time
    print(f"[MCS found in {elapsed:.4f}s] Size: {len(best_component)} nodes")

    return best_connected_common_subgraph, mapping, mcs


if __name__ == "__main__":
    # # Create sample graphs with labels and edge attributes
    # G1 = nx.Graph()
    # G1.add_nodes_from([
    #     (1, {"label": "A"}),
    #     (2, {"label": "B"}),
    #     (3, {"label": "C"})
    # ])
    # G1.add_edges_from([
    #     (1, 2, {"weight": 5, "type": "strong"}),
    #     (2, 3, {"weight": 3, "type": "weak"})
    # ])
    #
    # G2 = nx.Graph()
    # G2.add_nodes_from([
    #     (4, {"label": "A"}),
    #     (5, {"label": "B"}),
    #     (6, {"label": "D"})
    # ])
    # G2.add_edges_from([
    #     (4, 5, {"weight": 4, "type": "medium"}),
    #     (5, 6, {"weight": 2, "type": "weak"})
    # ])
    #
    # # Compute MCS
    # mcs, mapping = maximum_common_subgraph(G1, G2)
    # print("MCS Nodes:", mcs.nodes)
    # print("Node Mapping:", mapping)
    #
    # print("Execution completed")

    # Create sample graphs with labels and edge attributes
    G1 = nx.Graph()
    G1.add_nodes_from([
        (1, {"label": "A"}),
        (2, {"label": "B"}),
        (3, {"label": "C"})
    ])
    G1.add_edges_from([
        (1, 2, {"weight": 5, "type": "strong"}),
        (2, 3, {"weight": 3, "type": "weak"})
    ])

    G2 = nx.Graph()
    G2.add_nodes_from([
        (1, {"label": "A"}),
        (2, {"label": "C"}),
        (4, {"label": "D"})
    ])
    G2.add_edges_from([
        (1, 2, {"weight": 4, "type": "medium"}),
        (2, 4, {"weight": 2, "type": "weak"})
    ])

    # Compute MCS
    mcs, mapping = maximum_common_subgraph(G1, G2)
    print("MCS Nodes:", mcs.nodes)
    print("Node Mapping:", mapping)

    print("Execution completed")
