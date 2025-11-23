#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/19 17:53
# @Author   : **
# @Email    : **@**
# @File     : attack_visualization.py
# @Software : PyCharm
# @Desc     :
"""
import heapq
import os
import sys
import time
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse, csr_matrix
from deeprobust.graph.utils import *
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def adj_to_nx(adj, features, labels=None):
    """
    transfer adjacency matrix to networkx graph with node labels in the term of scipy.sparse (such as on deeprobust lib)
    :param adj: adjacency matrix
    :param labels: node labels
    :return: NetworkX graph
    """
    # Convert adjacency matrix to NetworkX graph
    if issparse(adj):
        adj = adj.tocoo()
    else:
        adj = csr_matrix(adj)
        adj = adj.tocoo()
        print("Warning: adjacency matrix is not sparse")
    G = nx.Graph()

    # add nodes' label
    # coo_fea = sp.coo_matrix(features)
    for i in range(adj.shape[0]):
        # G.add_node(i, label=int(labels[i]) if labels is not None else -1, feature=coo_fea.A[i])
        G.add_node(i, label=int(labels[i]) if labels is not None else -1)

    # add edges
    for i, j in zip(adj.row, adj.col):
        if i < j:  # avoid repeating edges
            G.add_edge(i, j)
    return G


def generate_timestamp_key():
    """
    Generates 19-digit unique key:
    [16-digit microsecond timestamp] + [3-digit random number]
    """
    # Get microsecond timestamp (16-digit integer)
    timestamp = int(time.time() * 1e0)

    # Generate 3-digit random number (000-999)
    random_suffix = random.randint(0, 999)

    # Combine components with zero-padding
    return f"{timestamp}{random_suffix:03d}"


def visualize_restricted_attack_subgraph(
        perturbed_adj,
        original_adj,
        labels,
        features,
        changed_label,
        target_node,
        attack_state,
        k_hop=2,
        max_nodes=20,
        title="Visualization for Adversarial Attack Subgraph",
        pic_path=None
):
    """
    Visualization for adversarial attack subgraph
    :param perturbed_adj: modified adjacency matrix
    :param original_adj: original adjacency matrix
    :param labels: node labels
    :param changed_label: changed label
    :param target_node: target node
    :param k_hop: k-hop nodes
    :param max_nodes: restricted nodes
    :param title: visualization title
    :param pic_path: save path of pictures
    :return:
    """
    start_time = time.time()
    # ===== 1. Build graphs and compute edge differences =====
    G_pert = adj_to_nx(perturbed_adj, features, labels)
    G_orig = adj_to_nx(original_adj, features, labels)

    # Compute added and removed edges
    added_edges = set(G_pert.edges()) - set(G_orig.edges())
    removed_edges = set(G_orig.edges()) - set(G_pert.edges())
    # removed_edges = set()
    all_mod_edges = added_edges | removed_edges

    # ===== 2. Hierarchical node sampling：Priority expansion with real-time connectivity guarantee =====
    # Critical nodes: target + all endpoints of modified edges
    critical_nodes = {target_node} | {n for edge in all_mod_edges for n in edge}

    # Base nodes: k-hop neighbors of target
    base_nodes = set(nx.single_source_shortest_path_length(G_pert, target_node, k_hop).keys())

    # Merge nodes and control total count
    candidate_nodes = {target_node} | {n for edge in added_edges for n in edge}  # The start must contain the target node and the key perturbation node (but the case of deleting the edge is excluded, because deleting the end node of the edge makes the subgraph no longer connected)
    visited = {target_node}  # Record visited nodes

    # dist_map = nx.single_source_shortest_path_length(G_pert, target_node)
    # candidate_nodes = sorted(candidate_nodes, key=lambda n: dist_map.get(n, float('inf')))[:max_nodes]
    # centrality = nx.degree_centrality(G_pert)
    # degree_dict = dict(G_pert.degree())
    centrality_dict = nx.degree_centrality(G_pert)

    # Max heap storage (-centrality, node) implements high node priority
    heap = []
    # Initialization: Add neighbors of the current candidate node
    for node in candidate_nodes:
        for neighbor in G_pert.neighbors(node):
            if neighbor not in visited and neighbor in base_nodes:
                heapq.heappush(heap, (-centrality_dict[neighbor], neighbor))  # Use centrality as weight

    while heap and len(candidate_nodes) < max_nodes:
        neg_centrality, current = heapq.heappop(heap)

        if current in visited:  # Avoid duplication of processing
            continue

        # Ensure connectivity: check if any neighbor is already in the candidate set
        if any(neighbor in candidate_nodes for neighbor in G_pert.neighbors(current)):
            candidate_nodes.add(current)
            visited.add(current)
        else:
            continue

        # Add unvisited neighbors of new node
        for neighbor in G_pert.neighbors(current):
            if neighbor not in visited and neighbor in base_nodes:
                heapq.heappush(heap, (-centrality_dict[neighbor], neighbor))  # Continue to use centrality weights

    # ===== 3. Build final subgraph =====
    true_subgraph = G_pert.subgraph(candidate_nodes).copy()

    elapsed = time.time() - start_time
    start_time = time.time()
    print(f"attack subgraph generated in {elapsed:.4f}s!")

    # Add deleted edges and end nodes for visualization
    unique_nodes = candidate_nodes | {n for edge in removed_edges for n in edge}
    subgraph = G_pert.subgraph(unique_nodes).copy()
    subgraph.add_edges_from(removed_edges)

    # Extract modified edges within subgraph

    sub_added = [e for e in added_edges if e[0] in critical_nodes and e[1] in critical_nodes]
    sub_removed = [e for e in removed_edges if e[0] in critical_nodes and e[1] in critical_nodes]

    # ===== 4. Visual configuration =====
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(subgraph, seed=42, k=0.7 / max_nodes ** 0.5)

    # --- Node coloring by class ---
    # Get unique classes and create color map
    unique_classes = np.unique(labels)
    class_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_color_map = {cls: class_colors[i] for i, cls in enumerate(unique_classes)}

    # Create node color list
    node_colors = [class_color_map[labels[n]] for n in subgraph.nodes()]

    # Draw nodes with class-based colors
    nx.draw_networkx_nodes(
        subgraph, pos,
        node_size=100,
        node_color=node_colors,
        edgecolors='grey',
        linewidths=0.8,
        alpha=0.9
    )

    # Highlight target node
    nx.draw_networkx_nodes(
        subgraph, pos,
        nodelist=[target_node],
        node_size=300,
        node_color=class_color_map[labels[target_node]],
        edgecolors='gold',
        linewidths=2.5,
        alpha=0.9
    )

    # Highlight critical nodes (endpoints of modified edges)
    critical_nodes_no_target = [n for n in critical_nodes if n != target_node]
    nx.draw_networkx_nodes(
        subgraph, pos,
        nodelist=critical_nodes_no_target,
        node_size=150,
        node_color=[class_color_map[labels[n]] for n in critical_nodes_no_target],
        edgecolors='red',
        linewidths=1.5,
        alpha=0.9
    )

    # --- Edge drawing ---
    # 1. Unmodified edges (gray)
    unmod_edges = set(subgraph.edges()) - set(sub_added)
    nx.draw_networkx_edges(
        subgraph, pos,
        edgelist=unmod_edges,
        edge_color='grey',
        width=0.8,
        alpha=0.3
    )

    # 2. Added edges (red solid)
    nx.draw_networkx_edges(
        subgraph, pos,
        edgelist=sub_added,
        edge_color='red',
        width=1.8,
        alpha=0.9,
        style='solid'
    )

    # 3. Removed edges (red dashed)
    nx.draw_networkx_edges(
        subgraph, pos,
        edgelist=sub_removed,
        edge_color='red',
        width=1.8,
        alpha=0.7,
        style='dashed'
    )

    # --- Label system ---
    # Node labels: ID:Class

    label_dict = {n: f"{n}:{labels[n]}" for n in unique_nodes if n in subgraph and n != target_node}
    label_dict[target_node] = f"{target_node}:{labels[target_node]}-->{changed_label}"

    # Create offset positions for labels (move labels upward)
    # label_pos = {node: (x, y + 0.05) for node, (x, y) in pos.items()}  # Increase the Y-axis offset
    label_pos = pos

    nx.draw_networkx_labels(
        subgraph,
        label_pos,  # Use the offset position
        labels=label_dict,
        font_size=7,
        font_weight='normal',
        verticalalignment='bottom',  # Set vertical alignment to the bottom
        horizontalalignment='center',  # Center horizontally
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec="gray",
            alpha=0.1
        )
    )

    # ===== 5. Legend system =====
    # Create legend elements
    legend_elements = []

    # Class color legend
    for cls in unique_classes:
        legend_elements.append(
            Patch(facecolor=class_color_map[cls], label=f'Class {cls}')
        )

    # Special elements legend
    legend_elements.extend([
        Patch(facecolor=class_color_map[labels[target_node]], edgecolor='gold',
              linewidth=2.5, label='Target Node'),
        Patch(facecolor='white', edgecolor='red', linewidth=1.5,
              label='Critical Node (Modified Edge Endpoint)'),
        Patch(edgecolor='red', linestyle='-', linewidth=2,
              label='Added Edge', facecolor='none'),
        Patch(edgecolor='red', linestyle='--', linewidth=2,
              label='Removed Edge', facecolor='none')
    ])

    # Position legend outside the plot
    plt.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        framealpha=0.9,
        title="Visual Legend"
    )

    # Title with statistics
    plt.title(
        f"{title}\n"
        f"Total Nodes: {len(subgraph.nodes())}/{max_nodes} | "
        f"Added Edges: {len(sub_added)} | "
        f"Removed Edges: {len(sub_removed)} | "
        f"Attack_State: {attack_state}",
        fontsize=14
    )
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make space for legend
    plt.savefig(pic_path + f"/{attack_state}_attack_{target_node}.png")
    # plt.show()

    elapsed = time.time() - start_time
    print(f"attack subgraph visualized in {elapsed:.4f}s!")

    return subgraph, true_subgraph

def visualize_attack_subgraph(
        perturbed_adj,
        original_adj,
        labels,
        features,
        changed_label,
        target_node,
        attack_state,
        title="Visualization for Adversarial Attack Subgraph",
        pic_path=None
):
    """
    Visualization for adversarial attack subgraph
    :param perturbed_adj: modified adjacency matrix
    :param original_adj: original adjacency matrix
    :param labels: node labels
    :param changed_label: changed label
    :param target_node: target node
    :param k_hop: k-hop nodes
    :param max_nodes: restricted nodes
    :param title: visualization title
    :param pic_path: save path of pictures
    :return:
    """
    start_time = time.time()
    # ===== 1. Build graphs and compute edge differences =====
    G_pert = adj_to_nx(perturbed_adj, features, labels)
    G_orig = adj_to_nx(original_adj, features, labels)

    # Compute added and removed edges
    added_edges = set(G_pert.edges()) - set(G_orig.edges())
    removed_edges = set(G_orig.edges()) - set(G_pert.edges())

    # save E+ or E-
    E_type = None
    if len(added_edges) > len(removed_edges):
        E_type = "E+"
    else:
        E_type = "E-"
    # removed_edges = set()
    all_mod_edges = added_edges | removed_edges

    # ===== 2. Hierarchical node sampling：Priority expansion with real-time connectivity guarantee =====
    # Critical nodes: target + all endpoints of modified edges
    critical_nodes = {target_node} | {n for edge in all_mod_edges for n in edge}
    candidate_nodes = {target_node} | {n for edge in added_edges for n in edge}

    # ===== 3. Build final subgraph =====
    true_subgraph = G_pert.subgraph(candidate_nodes).copy()

    elapsed = time.time() - start_time
    start_time = time.time()
    print(f"attack subgraph generated in {elapsed:.4f}s!")

    # Add attacked edges and end nodes for visualization
    subgraph = nx.Graph()  # Use undirected graph
    # add nodes
    for node in critical_nodes:
        node_label = int(labels[node]) if labels is not None else -1
        subgraph.add_node(node, label=node_label)
    # Add edges (store importance values)
    subgraph.add_edges_from(all_mod_edges)

    # Extract modified edges within subgraph
    sub_added = [e for e in added_edges if e[0] in critical_nodes and e[1] in critical_nodes]
    sub_removed = [e for e in removed_edges if e[0] in critical_nodes and e[1] in critical_nodes]

    # ===== 4. Visual configuration =====
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(subgraph, seed=42, k=0.7 / len(critical_nodes) ** 0.5)

    # --- Node coloring by class ---
    # Get unique classes and create color map
    unique_classes = np.unique(labels)
    class_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_color_map = {cls: class_colors[i] for i, cls in enumerate(unique_classes)}

    # Create node color list
    node_colors = [class_color_map[labels[n]] for n in subgraph.nodes()]

    # Draw nodes with class-based colors
    nx.draw_networkx_nodes(
        subgraph, pos,
        node_size=100,
        node_color=node_colors,
        edgecolors='grey',
        linewidths=0.8,
        alpha=0.9
    )

    # Highlight target node
    nx.draw_networkx_nodes(
        subgraph, pos,
        nodelist=[target_node],
        node_size=300,
        node_color=class_color_map[labels[target_node]],
        edgecolors='gold',
        linewidths=2.5,
        alpha=0.9
    )

    # Highlight critical nodes (endpoints of modified edges)
    critical_nodes_no_target = [n for n in critical_nodes if n != target_node]
    nx.draw_networkx_nodes(
        subgraph, pos,
        nodelist=critical_nodes_no_target,
        node_size=150,
        node_color=[class_color_map[labels[n]] for n in critical_nodes_no_target],
        edgecolors='red',
        linewidths=1.5,
        alpha=0.9
    )

    # --- Edge drawing ---
    # 1. Unmodified edges (gray)
    unmod_edges = set(subgraph.edges()) - set(sub_added)
    nx.draw_networkx_edges(
        subgraph, pos,
        edgelist=unmod_edges,
        edge_color='grey',
        width=0.8,
        alpha=0.3
    )

    # 2. Added edges (red solid)
    nx.draw_networkx_edges(
        subgraph, pos,
        edgelist=sub_added,
        edge_color='red',
        width=1.8,
        alpha=0.9,
        style='solid'
    )

    # 3. Removed edges (red dashed)
    nx.draw_networkx_edges(
        subgraph, pos,
        edgelist=sub_removed,
        edge_color='red',
        width=1.8,
        alpha=0.7,
        style='dashed'
    )

    # --- Label system ---
    # Node labels: ID:Class

    label_dict = {n: f"{n}:{labels[n]}" for n in critical_nodes if n in subgraph and n != target_node}
    label_dict[target_node] = f"{target_node}:{labels[target_node]}-->{changed_label}"

    # Create offset positions for labels (move labels upward)
    # label_pos = {node: (x, y + 0.05) for node, (x, y) in pos.items()}  # Increase the Y-axis offset
    label_pos = pos

    nx.draw_networkx_labels(
        subgraph,
        label_pos,  # Use the offset position
        labels=label_dict,
        font_size=12,
        font_weight='normal',
        verticalalignment='bottom',  # Set vertical alignment to the bottom
        horizontalalignment='center',  # Center horizontally
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec="gray",
            alpha=0.1
        )
    )

    # ===== 5. Legend system =====
    # Create legend elements
    legend_elements = []

    # Class color legend
    for cls in unique_classes:
        legend_elements.append(
            Patch(facecolor=class_color_map[cls], label=f'Class {cls}')
        )

    # Special elements legend
    legend_elements.extend([
        Patch(facecolor=class_color_map[labels[target_node]], edgecolor='gold',
              linewidth=2.5, label='Target Node'),
        Patch(facecolor='white', edgecolor='red', linewidth=1.5,
              label='Critical Node (Modified Edge Endpoint)'),
        Patch(edgecolor='red', linestyle='-', linewidth=2,
              label='Added Edge', facecolor='none'),
        Patch(edgecolor='red', linestyle='--', linewidth=2,
              label='Removed Edge', facecolor='none')
    ])

    # Position legend outside the plot
    plt.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        framealpha=0.9,
        title="Visual Legend"
    )

    # Title with statistics
    plt.title(
        f"{title}\n"
        f"Total Nodes: {len(subgraph.nodes())} | Total Edges: {len(subgraph.edges)} | "
        f"Added Edges: {len(sub_added)} | "
        f"Removed Edges: {len(sub_removed)} | "
        f"Attack_State: {attack_state}",
        fontsize=14
    )
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make space for legend
    plt.savefig(pic_path + f"/{attack_state}_attack_{target_node}_{E_type}.png")
    # plt.show()

    elapsed = time.time() - start_time
    print(f"attack subgraph visualized in {elapsed:.4f}s!")

    return subgraph, true_subgraph, E_type

if __name__ == '__main__':
    from deeprobust.graph.data import Dataset

    res = os.path.abspath(__file__)  # acquire absolute path of current file
    base_path = os.path.dirname(os.path.dirname(res))  # acquire the parent path of current file's parent path
    sys.path.insert(0, base_path)

    # test case
    data = Dataset(root=base_path + '/dataset', name='cora')
    original_adj = data.adj
    original_features = data.features
    labels = data.labels
    idx_train = data.idx_train
    idx_val = data.idx_val

    # PGD attack example using API
    from deeprobust.graph.global_attack import PGDAttack
    from deeprobust.graph.defense import GCN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_gcn = GCN(nfeat=original_features.shape[1],
                     nhid=16,
                     nclass=labels.max().item() + 1,
                     dropout=0.5, device=device)
    target_gcn = target_gcn.to(device)
    target_gcn.fit(original_features, original_adj, labels, idx_train, idx_val, patience=30)
    target_gcn.eval()
    output = target_gcn.predict()

    attack = PGDAttack(model=target_gcn, nnodes=original_adj.shape[0])
    attack.attack(original_features, original_adj, labels, idx_train, n_perturbations=50)  # perturbation budgets 50
    perturbed_adj = attack.modified_adj

    # visualization
    visualize_attack_graph(
        perturbed_adj,
        original_adj,
        labels,
        original_features,
        title="Visualization of PGDAttack evasion attack on the Cora dataset (the red dotted line represents the modified edges"
    )

    print("ok")

    # #1 Dynamic interactive visualization (PyVis)
    # from pyvis.network import Network
    #
    # # Generate interactive HTML
    # net = Network(notebook=True, cdn_resources='remote', height="800px")
    # net.from_nx(G_pert)
    # net.show("attacked_graph.html")

    # #2 Visualization of feature dimensionality reduction
    # from sklearn.manifold import TSNE
    #
    # # Reduce dimensionality features to 2D
    # tsne = TSNE(n_components=2, random_state=42)
    # feat_2d = tsne.fit_transform(perturbed_features.toarray())
    #
    # # Draw feature space distribution
    # plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=labels, cmap=plt.cm.tab10, s=20)
    # plt.colorbar(label="Node Class")
    # plt.title("Node feature distribution of adversarial samples (t-SNE dimensionality reduction)")

    # #3 Quantification of attack intensity
    # # Calculate the disturbance ratio
    # n_edges_orig = original_adj.sum() // 2
    # n_edges_pert = perturbed_adj.sum() // 2
    # perturb_ratio = abs(n_edges_pert - n_edges_orig) / n_edges_orig
    #
    # # Show in title
    # plt.title(f"Attack modification ratio: {perturb_ratio:.2%}", fontsize=14)

    # #4 Label overlap processing
    # # Adjust the label density through the font_size and alpha parameters of nx.draw_networkx_labels, or only display node labels with high degree centrality
    # degrees = dict(G_pert.degree())
    # high_degree_nodes = [n for n in G_pert.nodes() if degrees[n] > 5]
    # labels_subset = {n: label_dict[n] for n in high_degree_nodes}
