#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/22 14:37
# @Author   : **
# @Email    : **@**
# @File     : explanation_visualization.py
# @Software : PyCharm
# @Desc     : GNN Explanation Subgraph Visualization Tool - Enhanced Version
             This module provides subgraph visualization based on node and edge importance,
             supporting node category distinction, display of node ID and category labels,
             display of edge importance values, and golden border highlighting for target node.
"""
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import heapq
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import scipy.sparse as sp
from collections import defaultdict
from utilty.attack_visualization import generate_timestamp_key


def explanation_resricted_subgraph_visualization(
        explanation,
        target_node,
        edge_mask,
        labels,
        features,
        node_mask,
        threshold=0.8,
        max_nodes=20,
        title="Visualization for Explanation Subgraph",
        min_edge_value=0.8,
        ex_type='clean',
        pic_path=None,
        full_mapping=None
):
    """
    Enhanced Explanation Subgraph Visualization - Supports node importance, category distinction, label display, and target node highlighting
    :param explanation: PyG Data object - Contains graph structure information
    :param target_node: int - Index of the target node to be explained
    :param edge_mask: Tensor - Importance mask for all edges in the original graph
    :param labels: List/Tensor - List of node labels/categories
    :param node_mask: Tensor(optional) - Importance mask for all nodes in the original graph
    :param threshold: threshold for important edges
    :param max_nodes: int - Maximum number of nodes to visualize
    :param title: str - Chart title
    :param min_edge_value: float - Minimum threshold for displaying edge importance (values below this will not be displayed)
    :return: nx.Graph: Visualized NetworkX graph object
    """
    start_time = time.time()
    true_target_node = target_node
    if ex_type == 'clean':
        title = f"Visualization for Clean Explanation Subgraph"
    elif ex_type == 'attack':
        title = f"Visualization for Attack Explanation Subgraph"

    # ====================== 1. Edge Importance Processing ======================
    edge_importance = edge_mask[edge_mask > min_edge_value].detach().cpu().numpy()
    edge_index = explanation.edge_index[:, edge_mask > min_edge_value].cpu().numpy()

    # ====================== 2. Extract Key Nodes Based on Edge Importance ======================
    # 构建邻接表（高效存储节点连接关系）
    adj = defaultdict(list)
    for i in range(edge_index.shape[1]):
        u, v = map(int, edge_index[:, i])
        imp = edge_importance[i]
        adj[u].append((v, imp, i))
        adj[v].append((u, imp, i))

    # 初始化数据结构
    if full_mapping:
        target_node = full_mapping[target_node]
    connected_nodes = {target_node}  # 必须包含目标节点
    selected_edges = set()
    heap = []  # 最大堆存储 (-imp, u, v, edge_idx)

    # 初始化：将目标节点的邻边加入优先队列[5,9](@ref)
    for neighbor, imp, idx in adj[target_node]:
        heapq.heappush(heap, (-imp, target_node, neighbor, idx))

    # 优先队列扩展连通子图
    while heap and len(connected_nodes) < max_nodes:
        neg_imp, u, v, idx = heapq.heappop(heap)
        imp = -neg_imp

        # 跳过已处理的边
        if idx in selected_edges:
            continue

        # 记录新节点（如果未超限）
        new_node = None
        if v not in connected_nodes:
            if len(connected_nodes) >= max_nodes:
                continue
            connected_nodes.add(v)
            new_node = v

        # 添加当前边（无论是否添加新节点）
        selected_edges.add(idx)

        # 将新节点的邻边加入优先队列
        if new_node is not None:
            for neighbor, new_imp, new_idx in adj[new_node]:
                # 避免重复添加已处理边
                if new_idx not in selected_edges:
                    heapq.heappush(heap, (-new_imp, new_node, neighbor, new_idx))

    # ====================== 3. Prune Edges ======================
    # 提取最终节点和边
    unique_nodes = np.array(list(connected_nodes))
    selected_edges = list(selected_edges)

    pruned_edge_index = edge_index[:, selected_edges]
    pruned_edge_importance = edge_importance[selected_edges]

    # ====================== 4. Build Undirected NetworkX Graph ======================
    subgraph = nx.Graph()  # Use undirected graph
    if full_mapping:
        reversed_mapping = {idx: orig_id for orig_id, idx in full_mapping.items()}
        # Add nodes (with label and category info)
        # coo_fea = sp.coo_matrix(features)
        for node in unique_nodes:
            node = reversed_mapping[node]
            # node_label = f"{node}:{labels[node]}"
            node_label = int(labels[node]) if labels is not None else -1
            # subgraph.add_node(node, label=node_label, feature=coo_fea.A[node])
            subgraph.add_node(node, label=node_label)

        # Add edges (store importance values)
        for i in range(pruned_edge_index.shape[1]):
            u, v = pruned_edge_index[:, i]
            importance = pruned_edge_importance[i]
            u = reversed_mapping[u]
            v = reversed_mapping[v]
            subgraph.add_edge(u, v, weight=importance, label=f"{importance:.2f}")
    else:
        # Add nodes (with label and category info)
        # coo_fea = sp.coo_matrix(features)
        for node in unique_nodes:
            # node_label = f"{node}:{labels[node]}"
            node_label = int(labels[node]) if labels is not None else -1
            # subgraph.add_node(node, label=node_label, feature=coo_fea.A[node])
            subgraph.add_node(node, label=node_label)

        # Add edges (store importance values)
        for i in range(pruned_edge_index.shape[1]):
            u, v = pruned_edge_index[:, i]
            importance = pruned_edge_importance[i]
            subgraph.add_edge(u, v, weight=importance, label=f"{importance:.2f}")

    # 连通性检查（强制包含目标节点）
    if not nx.is_connected(subgraph):
        largest_cc = max(nx.connected_components(subgraph), key=len)
        if target_node not in largest_cc:
            largest_cc.add(target_node)
        subgraph = subgraph.subgraph(largest_cc)

    G = subgraph
    unique_nodes_new = np.array(list(G.nodes))
    target_node = true_target_node
    n = len(unique_nodes_new)
    # Empty graph check
    if n == 0:
        print("Warning: Subgraph is empty, skipping visualization")
    elapsed = time.time() - start_time
    print(f"explanation generated in {elapsed:.4f}s!")
    start_time = time.time()

    # ====================== 5. Node Importance Processing ======================
    # Create full node importance array (non-important nodes=0)
    full_importance = None
    important_nodes = node_mask > threshold
    if node_mask is not None and important_nodes is not None:
        full_importance = torch.zeros_like(node_mask, dtype=torch.float)
        # Ensure important_nodes is boolean tensor and length matches
        if len(important_nodes) == len(node_mask):
            full_importance[important_nodes] = node_mask[important_nodes].float()
        else:
            print("Warning: Length of important_nodes does not match node_mask, ignoring node importance")
            full_importance = None

    # Crop node importance to current subgraph nodes
    if full_mapping:
        unique_nodes = np.array([full_mapping[node] for node in unique_nodes_new])
    else:
        unique_nodes = unique_nodes_new
    node_importance_subgraph = None
    if full_importance is not None and len(unique_nodes_new) > 0:
        try:
            # Precisely index node importance for current subgraph
            node_importance_subgraph = full_importance[unique_nodes].detach().cpu().numpy()
        except IndexError:
            print("Warning: Node index out of bounds, ignoring node importance")
            node_importance_subgraph = None

    # ====================== 6. Node Categories and Color Mapping ======================
    # Get unique categories and create color mapping
    unique_classes = np.unique(labels)
    class_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_color_map = {cls: class_colors[i] for i, cls in enumerate(unique_classes)}

    # Assign colors to each node in subgraph
    node_colors = [class_color_map[labels[node]] for node in unique_nodes_new]

    # ====================== 7. Target Node Border Settings ======================
    # Create border color list (gold for target node, gray for others)
    edge_colors = ['grey'] * n
    # Create border width list (thicker for target node)
    line_widths = [0.5] * n

    # Find position of target node in unique_nodes_new
    target_idx = np.where(unique_nodes_new == target_node)[0]
    if len(target_idx) > 0:
        target_idx = target_idx[0]
        edge_colors[target_idx] = 'gold'  # Set gold border
        line_widths[target_idx] = 2.5  # Set thicker border
    else:
        print(f"Warning: Target node {target_node} not in subgraph, forcing to add")
        # If target node not in subgraph, force add
        unique_nodes_new = np.append(unique_nodes_new, target_node)
        node_colors.append(class_color_map[labels[target_node]])
        edge_colors.append('gold')  # Gold border
        line_widths.append(2.5)  # Thicker border
        n = len(unique_nodes_new)  # Update node count

    # ====================== 8. Visualization Parameter Settings ======================
    plt.figure(figsize=(14, 10))

    # Smart layout selection - ensure target node at center
    # if n <= 50:
    #     # Use spring_layout with target node fixed at center
    #     pos = nx.spring_layout(G, seed=42, iterations=100, k=1.5,
    #                            fixed=[target_node], pos={target_node: (0, 0)})
    # elif n <= 200:
    #     pos = nx.kamada_kawai_layout(G)
    # else:
    #     pos = nx.random_layout(G, seed=42)
    pos = nx.spring_layout(G, seed=42, k=0.7 / max_nodes ** 0.5)

    # ====================== 9. Node Visualization ======================
    # Map node size to importance (100-500 pixel range)
    base_size = min(300, 3000 // max(1, n))

    if node_importance_subgraph is not None and len(node_importance_subgraph) == n:
        node_importance_subgraph = np.abs(node_importance_subgraph)
        node_importance_subgraph = np.sum(node_importance_subgraph, axis=1)
        # Normalize node importance (0-1 range)
        if node_importance_subgraph.max() > 0:
            node_imp_normalized = node_importance_subgraph / node_importance_subgraph.max()
        else:
            node_imp_normalized = np.zeros_like(node_importance_subgraph)

        node_size = np.clip(node_imp_normalized * 400 + 100, 100, 500)
    else:
        node_size = [base_size] * n  # Ensure node_size is list of length n

    # Draw nodes (using category colors and custom borders)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_colors,
        alpha=0.9,
        edgecolors=edge_colors,  # Custom border colors
        linewidths=line_widths  # Custom border widths
    )

    # Create offset positions for labels (move labels upward)
    # label_pos = {node: (x, y + 0.1) for node, (x, y) in pos.items()}  # Increase Y-axis offset
    label_pos = pos

    # Add node labels (display ID and category)
    label_dict = {n: f"{n}:{labels[n]}" for n in unique_nodes_new if n in G and n != target_node}
    label_dict[target_node] = f"{target_node}:{labels[target_node]}"
    # node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(
        G, label_pos,
        labels=label_dict,
        font_size=7,
        font_weight='normal',
        font_color='black',
        font_family='sans-serif',
        verticalalignment='bottom',
        horizontalalignment='center',
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec="gray",
            alpha=0.1
        )
    )

    # ====================== 10. Edge Visualization (Undirected Edges) ======================
    # Map edge width to importance (1-8 pixel range)
    edge_widths = np.clip(pruned_edge_importance * 2 + 0.5, 0.5, 4)

    # Draw undirected edges (remove arrows)
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color='gray',  # Fixed color
        alpha=0.8,
        arrows=False  # Key modification: remove arrows
    )

    # ====================== 11. Add Edge Importance Labels ======================
    # Prepare edge labels (only show values above threshold)
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        if data['weight'] >= min_edge_value:
            edge_labels[(u, v)] = data['label']

    # Draw edge labels
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=7,
        font_color='red',
        font_weight='normal',
        label_pos=0.5,  # Label position at edge center
        bbox=dict(alpha=0.5, facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
    )

    # ====================== 12. Legend and Decoration ======================
    plt.title(f"{title}\nTarget Node: {target_node} | Nodes: {n} | Edges: {len(pruned_edge_index[0])}",
              fontsize=16, pad=20)

    # Create custom legend (show node category colors)
    legend_elements = []
    for cls, color in class_color_map.items():
        legend_elements.append(Patch(facecolor=color, label=f'Class {cls}'))

    # Add target node legend item
    legend_elements.append(Patch(facecolor='white', edgecolor='gold', linewidth=2.5, label='Target Node'))

    plt.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        framealpha=0.9,
        title="Visual Legend"
    )

    # Add importance description
    plt.text(
        0.05, 0.05,
        "Node Size = Importance | Edge Width = Importance | Edge Label = Importance Value",
        transform=plt.gcf().transFigure,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(pic_path + f"/{ex_type}_explanation_{target_node}.png")
    # plt.show()

    elapsed = time.time() - start_time
    print(f"explanation subgraph visualized in {elapsed:.4f}s!")

    print(f"{ex_type} subgraph visualization completed for node {target_node}")

    return G


def explanation_subgraph_visualization(
        explanation,
        target_node,
        edge_mask,
        labels,
        features,
        node_mask,
        attack_subgraph_edge_num,
        title="Visualization for Explanation Subgraph",
        ex_type='clean',
        pic_path=None,
        full_mapping=None
):
    """
    Enhanced Explanation Subgraph Visualization - Supports node importance, category distinction, label display, and target node highlighting
    :param explanation: PyG Data object - Contains graph structure information
    :param target_node: int - Index of the target node to be explained
    :param edge_mask: Tensor - Importance mask for all edges in the original graph
    :param labels: List/Tensor - List of node labels/categories
    :param node_mask: Tensor(optional) - Importance mask for all nodes in the original graph
    :param attack_subgraph_edge_num: attack subgraph number
    :param title: str - Chart title
    :param min_edge_value: float - Minimum threshold for displaying edge importance (values below this will not be displayed)
    :return: nx.Graph: Visualized NetworkX graph object
    """
    start_time = time.time()
    true_target_node = target_node
    if ex_type == 'clean':
        title = f"Visualization for Clean Explanation Subgraph"
    elif ex_type == 'attack':
        title = f"Visualization for Attack Explanation Subgraph"

    # ====================== 1. Edge Importance Processing ======================
    threshold = 0
    edge_importance = edge_mask[edge_mask > threshold].detach().cpu().numpy()
    edge_index = explanation.edge_index[:, edge_mask > threshold].cpu().numpy()

    # ====================== 2. Extract Key Nodes Based on Edge Importance ======================
    # 构建邻接表（高效存储节点连接关系）
    adj = defaultdict(list)
    for i in range(edge_index.shape[1]):
        u, v = map(int, edge_index[:, i])
        imp = edge_importance[i]
        adj[u].append((v, imp, i))
        adj[v].append((u, imp, i))

    # 初始化数据结构
    if full_mapping:
        target_node = full_mapping[target_node]
    connected_nodes = {target_node}  # 必须包含目标节点
    selected_edges = set()
    heap = []  # 最大堆存储 (-imp, u, v, edge_idx)

    # 初始化：将目标节点的邻边加入优先队列
    for neighbor, imp, idx in adj[target_node]:
        heapq.heappush(heap, (-imp, target_node, neighbor, idx))

    # 优先队列扩展连通子图
    while heap and len(selected_edges) < attack_subgraph_edge_num:
        neg_imp, u, v, idx = heapq.heappop(heap)
        imp = -neg_imp

        # 跳过已处理的边
        if idx in selected_edges:
            continue

        # 记录新节点（如果未超限）
        new_node = None
        if v not in connected_nodes:
            if len(selected_edges) >= attack_subgraph_edge_num:
                continue
            connected_nodes.add(v)
            new_node = v

        # 添加当前边（无论是否添加新节点）
        selected_edges.add(idx)

        # 将新节点的邻边加入优先队列
        if new_node is not None:
            for neighbor, new_imp, new_idx in adj[new_node]:
                # 避免重复添加已处理边
                if new_idx not in selected_edges:
                    heapq.heappush(heap, (-new_imp, new_node, neighbor, new_idx))

    # ====================== 3. Prune Edges ======================
    # 提取最终节点和边
    selected_edges = list(selected_edges)
    pruned_edge_index = edge_index[:, selected_edges]
    pruned_edge_importance = edge_importance[selected_edges]

    # ====================== 4. Build Undirected NetworkX Graph ======================
    subgraph = nx.Graph()  # Use undirected graph
    if full_mapping:
        reversed_mapping = {idx: orig_id for orig_id, idx in full_mapping.items()}
        # add nodes
        for node in connected_nodes:
            node = reversed_mapping[node]
            node_label = int(labels[node]) if labels is not None else -1
            subgraph.add_node(node, label=node_label)

        # Add edges (store importance values)
        for i in range(pruned_edge_index.shape[1]):
            u, v = pruned_edge_index[:, i]
            importance = pruned_edge_importance[i]
            u = reversed_mapping[u]
            v = reversed_mapping[v]
            subgraph.add_edge(u, v, weight=importance, label=f"{importance:.2f}")
    else:
        # add nodes
        for node in connected_nodes:
            node_label = int(labels[node]) if labels is not None else -1
            subgraph.add_node(node, label=node_label)

        # Add edges (store importance values)
        for i in range(pruned_edge_index.shape[1]):
            u, v = pruned_edge_index[:, i]
            importance = pruned_edge_importance[i]
            subgraph.add_edge(u, v, weight=importance, label=f"{importance:.2f}")

    G = subgraph
    unique_nodes_new = np.array(list(G.nodes))
    target_node = true_target_node
    n = len(unique_nodes_new)
    # Empty graph check
    if n == 0:
        print("Warning: Subgraph is empty, skipping visualization")
    elapsed = time.time() - start_time
    print(f"explanation generated in {elapsed:.4f}s!")
    start_time = time.time()

    # ====================== 5. Node Importance Processing ======================
    # Create full node importance array (non-important nodes=0)
    full_importance = None
    important_nodes = node_mask > threshold
    if node_mask is not None and important_nodes is not None:
        full_importance = torch.zeros_like(node_mask, dtype=torch.float)
        # Ensure important_nodes is boolean tensor and length matches
        if len(important_nodes) == len(node_mask):
            full_importance[important_nodes] = node_mask[important_nodes].float()
        else:
            print("Warning: Length of important_nodes does not match node_mask, ignoring node importance")
            full_importance = None

    # Crop node importance to current subgraph nodes
    if full_mapping:
        unique_nodes = np.array([full_mapping[node] for node in unique_nodes_new])
    else:
        unique_nodes = unique_nodes_new
    node_importance_subgraph = None
    if full_importance is not None and len(unique_nodes_new) > 0:
        try:
            # Precisely index node importance for current subgraph
            node_importance_subgraph = full_importance[unique_nodes].detach().cpu().numpy()
        except IndexError:
            print("Warning: Node index out of bounds, ignoring node importance")
            node_importance_subgraph = None

    # ====================== 6. Node Categories and Color Mapping ======================
    # Get unique categories and create color mapping
    unique_classes = np.unique(labels)
    class_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_color_map = {cls: class_colors[i] for i, cls in enumerate(unique_classes)}

    # Assign colors to each node in subgraph
    node_colors = [class_color_map[labels[node]] for node in unique_nodes_new]

    # ====================== 7. Target Node Border Settings ======================
    # Create border color list (gold for target node, gray for others)
    edge_colors = ['grey'] * n
    # Create border width list (thicker for target node)
    line_widths = [0.5] * n

    # Find position of target node in unique_nodes_new
    target_idx = np.where(unique_nodes_new == target_node)[0]
    if len(target_idx) > 0:
        target_idx = target_idx[0]
        edge_colors[target_idx] = 'gold'  # Set gold border
        line_widths[target_idx] = 2.5  # Set thicker border
    else:
        print(f"Warning: Target node {target_node} not in subgraph, forcing to add")
        # If target node not in subgraph, force add
        unique_nodes_new = np.append(unique_nodes_new, target_node)
        node_colors.append(class_color_map[labels[target_node]])
        edge_colors.append('gold')  # Gold border
        line_widths.append(2.5)  # Thicker border
        n = len(unique_nodes_new)  # Update node count

    # ====================== 8. Visualization Parameter Settings ======================
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=0.7 / len(unique_nodes_new) ** 0.5)

    # ====================== 9. Node Visualization ======================
    # Map node size to importance (100-500 pixel range)
    base_size = min(300, 3000 // max(1, n))

    if node_importance_subgraph is not None and len(node_importance_subgraph) == n:
        node_importance_subgraph = np.abs(node_importance_subgraph)
        node_importance_subgraph = np.sum(node_importance_subgraph, axis=1)
        # Normalize node importance (0-1 range)
        if node_importance_subgraph.max() > 0:
            node_imp_normalized = node_importance_subgraph / node_importance_subgraph.max()
        else:
            node_imp_normalized = np.zeros_like(node_importance_subgraph)

        node_size = np.clip(node_imp_normalized * 400 + 100, 100, 500)
    else:
        node_size = [base_size] * n  # Ensure node_size is list of length n

    # Draw nodes (using category colors and custom borders)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_colors,
        alpha=0.9,
        edgecolors=edge_colors,  # Custom border colors
        linewidths=line_widths  # Custom border widths
    )

    # Create offset positions for labels (move labels upward)
    # label_pos = {node: (x, y + 0.1) for node, (x, y) in pos.items()}  # Increase Y-axis offset
    label_pos = pos

    # Add node labels (display ID and category)
    label_dict = {n: f"{n}:{labels[n]}" for n in unique_nodes_new if n in G and n != target_node}
    label_dict[target_node] = f"{target_node}:{labels[target_node]}"
    nx.draw_networkx_labels(
        G, label_pos,
        labels=label_dict,
        font_size=12,
        font_weight='normal',
        font_color='black',
        font_family='sans-serif',
        verticalalignment='bottom',
        horizontalalignment='center',
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec="gray",
            alpha=0.1
        )
    )

    # ====================== 10. Edge Visualization (Undirected Edges) ======================
    # Map edge width to importance (1-8 pixel range)
    edge_widths = np.clip(pruned_edge_importance * 2 + 0.5, 0.5, 4)

    # Draw undirected edges (remove arrows)
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color='gray',  # Fixed color
        alpha=0.8,
        arrows=False  # Key modification: remove arrows
    )

    # ====================== 11. Add Edge Importance Labels ======================
    # Prepare edge labels (only show values above threshold)
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        if data['weight'] >= threshold:
            edge_labels[(u, v)] = data['label']

    # Draw edge labels
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=7,
        font_color='red',
        font_weight='normal',
        label_pos=0.5,  # Label position at edge center
        bbox=dict(alpha=0.5, facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
    )

    # ====================== 12. Legend and Decoration ======================
    plt.title(f"{title}\nTarget Node: {target_node} | Control Edges: {attack_subgraph_edge_num} | Total Nodes: {n} | Total Edges: {len(G.edges)}",
              fontsize=16, pad=20)

    # Create custom legend (show node category colors)
    legend_elements = []
    for cls, color in class_color_map.items():
        legend_elements.append(Patch(facecolor=color, label=f'Class {cls}'))

    # Add target node legend item
    legend_elements.append(Patch(facecolor='white', edgecolor='gold', linewidth=2.5, label='Target Node'))

    plt.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        framealpha=0.9,
        title="Visual Legend"
    )

    # Add importance description
    plt.text(
        0.05, 0.05,
        "Node Size = Importance | Edge Width = Importance | Edge Label = Importance Value",
        transform=plt.gcf().transFigure,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(pic_path + f"/{ex_type}_explanation_{target_node}.png")
    # plt.show()

    elapsed = time.time() - start_time
    print(f"explanation subgraph visualized in {elapsed:.4f}s!")

    print(f"{ex_type} subgraph visualization completed for node {target_node}")

    return G