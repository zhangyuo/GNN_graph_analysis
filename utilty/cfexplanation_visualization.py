#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/19 17:53
# @Author   : **
# @Email    : **@**
# @File     : cfexplanation_visualization.py
# @Software : PyCharm
# @Desc     :
"""
import time
import networkx as nx
import matplotlib.pyplot as plt
from deeprobust.graph.utils import *
from matplotlib.patches import Patch

from utilty.attack_visualization import adj_to_nx


def visualize_cfexp_subgraph(
        perturbed_adj,
        original_adj,
        labels,
        sub_labels,
        features,
        changed_label,
        target_node,
        cfexp_name='CFExplanation',
        k_hop=2,
        max_nodes=20,
        title="Visualization for Adversarial Attack Subgraph",
        pic_path=None,
        full_mapping=None
):
    """
    Visualization for adversarial attack subgraph
    :param perturbed_adj: modified adjacency matrix
    :param original_adj: original adjacency matrix
    :param sub labels: node sub labels
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
    G_pert = adj_to_nx(perturbed_adj, features, sub_labels)
    G_orig = adj_to_nx(original_adj, features, sub_labels)

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

    # 增加被攻击的边和末端节点，用于可视化
    reversed_mapping = {idx: orig_id for orig_id, idx in full_mapping.items()}
    subgraph = nx.Graph()  # Use undirected graph
    # add nodes
    for node in critical_nodes:
        node_label = int(sub_labels[node]) if sub_labels is not None else -1
        subgraph.add_node(reversed_mapping[node], label=node_label)
    # Add edges (store importance values)
    # subgraph.add_edges_from(all_mod_edges)
    for mod_edges_data in all_mod_edges.copy():
        u, v = mod_edges_data
        # importance = pruned_edge_importance[i]
        u = reversed_mapping[u]
        v = reversed_mapping[v]
        subgraph.add_edge(u, v)

    # Extract modified edges within subgraph
    sub_added = [(reversed_mapping[i], reversed_mapping[j]) for i, j in added_edges if i in critical_nodes and j in critical_nodes]
    sub_removed = [(reversed_mapping[i], reversed_mapping[j]) for i, j in removed_edges if i in critical_nodes and j in critical_nodes]

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
        nodelist=[reversed_mapping[target_node]],
        node_size=300,
        node_color=class_color_map[sub_labels[target_node]],
        edgecolors='gold',
        linewidths=2.5,
        alpha=0.9
    )

    # Highlight critical nodes (endpoints of modified edges)
    critical_nodes_no_target = [reversed_mapping[n] for n in critical_nodes if n != target_node]
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
    critical_nodes = subgraph.nodes()
    target_node = reversed_mapping[target_node]
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
        f"CounterFactual Explanation Name: {cfexp_name}",
        fontsize=14
    )
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make space for legend
    plt.savefig(pic_path + f"/{cfexp_name}_attack_{target_node}_{E_type}.png")
    # plt.show()

    elapsed = time.time() - start_time
    print(f"attack subgraph visualized in {elapsed:.4f}s!")

    return subgraph, true_subgraph, E_type
