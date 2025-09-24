#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/24 16:39
# @Author   : **
# @Email    : **@**
# @File     : maximum_common_graph_visualization.py
# @Software : PyCharm
# @Desc     :
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def mx_com_graph_view(target_node, G1, G2, mcs, mapping, path, graph_one_name="Graph G1", graph_two_name="Graph G2",
                      pic_name='at_vs_ex'):
    # Create figure with enhanced labeling
    fig = plt.figure(figsize=(30, 10), dpi=100)
    plt.suptitle(f"Maximum Common Subgraph Visualization\nTarget Node: {target_node}", fontsize=16, fontweight='bold')

    # ========== Graph G1 Visualization ==========
    ax1 = plt.subplot(131)
    pos1 = nx.spring_layout(G1, seed=42)
    nx.draw_networkx_nodes(G1, pos1, node_size=500,
                           node_color="skyblue", edgecolors='black', linewidths=2)
    nx.draw_networkx_edges(G1, pos1, width=2, edge_color='gray')
    nx.draw_networkx_labels(G1, pos1, font_size=14,
                            font_weight='normal', font_color='black')

    # Add edge labels with weights and types
    edge_labels1 = {(u, v): f"{d.get('weight', '')}{d.get('type', '')}"
                    for u, v, d in G1.edges(data=True)}
    nx.draw_networkx_edge_labels(G1, pos1, edge_labels=edge_labels1,
                                 font_size=8, bbox=dict(alpha=0.1))

    ax1.set_title(graph_one_name, fontsize=14, pad=20)
    # ax1.text(0.5, -0.1, f"Nodes: {G1.nodes()}\nEdges: {G1.edges(data=True)}",
    #          transform=ax1.transAxes, ha='center', fontsize=9)

    # ========== Graph G2 Visualization ==========
    ax2 = plt.subplot(132)
    pos2 = nx.spring_layout(G2, seed=42)
    nx.draw_networkx_nodes(G2, pos2, node_size=500,
                           node_color="lightcoral", edgecolors='black', linewidths=2)
    nx.draw_networkx_edges(G2, pos2, width=2, edge_color='gray')
    nx.draw_networkx_labels(G2, pos2, font_size=14,
                            font_weight='normal', font_color='black')

    # Add edge labels with weights and types
    edge_labels2 = {(u, v): f"{round(float(d.get('weight', '-0')), 2)}{d.get('type', '')}"
                    for u, v, d in G2.edges(data=True)}
    nx.draw_networkx_edge_labels(G2, pos2, edge_labels=edge_labels2,
                                 font_size=8, bbox=dict(alpha=0.1))

    ax2.set_title(graph_two_name, fontsize=14, pad=20)
    # ax2.text(0.5, -0.1, f"Nodes: {G2.nodes()}\nEdges: {G2.edges(data=True)}",
    #          transform=ax2.transAxes, ha='center', fontsize=9)

    # ========== MCS Visualization ==========
    ax3 = plt.subplot(133)
    pos_mcs = {node: pos1[node] for node in mcs.nodes()}  # Maintain G1's layout for consistency

    # Highlight MCS nodes in green
    nx.draw_networkx_nodes(mcs, pos_mcs, node_size=500,
                           node_color="lightgreen", edgecolors='darkgreen', linewidths=3)
    nx.draw_networkx_edges(mcs, pos_mcs, width=3, edge_color='green')
    nx.draw_networkx_labels(mcs, pos_mcs, font_size=14,
                            font_weight='normal', font_color='darkgreen')

    # Add mapping annotations
    # for node in mcs.nodes():
    #     plt.annotate(f"→ {mapping[node]}",
    #                  xy=pos_mcs[node],
    #                  xytext=(pos_mcs[node][0], pos_mcs[node][1] + 0.15),
    #                  ha='center', fontsize=10, color='purple',
    #                  arrowprops=dict(arrowstyle="->", color="purple", alpha=0.7))

    # Add edge labels for MCS
    edge_labels_mcs = {(u, v): f"{d.get('weight', '')}{d.get('type', '')}"
                       for u, v, d in mcs.edges(data=True)}
    nx.draw_networkx_edge_labels(mcs, pos_mcs, edge_labels=edge_labels_mcs,
                                 font_size=8, bbox=dict(alpha=0.1))

    ax3.set_title("Maximum Common Subgraph", fontsize=14, pad=20)
    # ax3.text(0.5, -0.1, f"MCS Size: {len(mcs.nodes)} nodes\nNode Mapping: {mapping}",
    #          transform=ax3.transAxes, ha='center', fontsize=9)

    # Final formatting
    for ax in [ax1, ax2, ax3]:
        ax.set_axis_off()
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(path + f"/mcs_{target_node}_{pic_name}.png",
                bbox_inches='tight')
    print(f"Visualization saved as mcs_{target_node}.png")
    # plt.show()
