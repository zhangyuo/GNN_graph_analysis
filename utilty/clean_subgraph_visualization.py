#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/25 14:24
# @Author   : **
# @Email    : **@**
# @File     : clean_subgraph_visualization.py
# @Software : PyCharm
# @Desc     :
"""
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from utilty.attack_visualization import adj_to_nx
from matplotlib.patches import Patch


def visualize_restricted_clean_subgraph(
        original_adj,
        labels,
        features,
        target_node,
        k_hop=2,
        max_nodes=50,
        title="Visualization for Clean Subgraph",
        pic_path=None
):
    """
    Visualization for Clean Subgraph
    :param original_adj: original adjacency matrix
    :param labels: node labels
    :param target_node: target node
    :param k_hop: k-hop nodes
    :param max_nodes: restricted nodes
    :param title: visualization title
    :param pic_path: save path of pictures
    :return:
    """
    # ===== 1. Build graphs =====
    G_orig = adj_to_nx(original_adj, features, labels)

    # ===== 2. Hierarchical node sampling =====
    # Critical nodes: target + all endpoints of modified edges
    critical_nodes = {target_node}

    # Base nodes: k-hop neighbors of target
    base_nodes = set(nx.single_source_shortest_path_length(G_orig, target_node, k_hop).keys())

    # Merge nodes and control total count
    candidate_nodes = critical_nodes | base_nodes
    # if len(candidate_nodes) > max_nodes:
    #     dist_map = nx.single_source_shortest_path_length(G_pert, target_node)
    #     candidate_nodes = sorted(candidate_nodes, key=lambda n: dist_map.get(n, float('inf')))[:max_nodes]
    if len(candidate_nodes) > max_nodes:
        # Hierarchical sampling: First, prioritize the retention of key nodes, and then use centrality to make up for the rest.
        remaining = max_nodes - len(critical_nodes)
        if remaining > 0:
            centrality = nx.degree_centrality(G_orig)
            high_centrality = sorted(
                base_nodes - critical_nodes,
                key=lambda n: centrality[n],
                reverse=True
            )[:remaining]
            candidate_nodes = critical_nodes | set(high_centrality)
        else:
            candidate_nodes = critical_nodes  # The critical nodes have exceeded the limit.

    # exclude isolated nodes
    subgraph = G_orig.subgraph(candidate_nodes).copy()
    # 检查连通性，同时强制保留关键节点
    if not nx.is_connected(subgraph):
        # 获取所有连通分量，按大小排序
        connected_components = sorted(nx.connected_components(subgraph), key=len, reverse=True)

        # 步骤1：优先保留含关键节点的连通分量
        critical_components = set()
        for comp in connected_components:
            if any(node in comp for node in critical_nodes):
                critical_components |= comp  # 合并含关键节点的分量

        # 步骤2：若关键节点分散，合并其所在分量（确保连通性）
        if critical_components:
            candidate_nodes = critical_components
        else:
            # 若无关键节点，默认取最大分量（安全回退）
            candidate_nodes = connected_components[0]

    # ===== 3. Build final subgraph =====
    subgraph = G_orig.subgraph(candidate_nodes).copy()

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
    unmod_edges = set(subgraph.edges())
    nx.draw_networkx_edges(
        subgraph, pos,
        edgelist=unmod_edges,
        edge_color='grey',
        width=0.8,
        alpha=0.3
    )

    # --- Label system ---
    # Node labels: ID:Class

    label_dict = {n: f"{n}:{labels[n]}" for n in candidate_nodes if n in subgraph and n != target_node}
    label_dict[target_node] = f"{target_node}:{labels[target_node]}"

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
              linewidth=2.5, label='Target Node')
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
        f"Total Nodes: {len(subgraph.nodes())}/{max_nodes}",
        fontsize=14
    )
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make space for legend
    plt.savefig(pic_path + f"/clean_{target_node}.png")
    # plt.show()
    return subgraph
