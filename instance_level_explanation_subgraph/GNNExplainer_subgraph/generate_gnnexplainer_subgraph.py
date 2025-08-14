#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/24 11:47
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : generate_gnnexplainer_subgraph.py
# @Software : PyCharm
# @Desc     :
"""
import os
import time

import numpy as np
import torch
from torch_geometric.utils import add_self_loops, k_hop_subgraph
from torch_sparse import SparseTensor
from utilty.explanation_visualization import explanation_subgraph_visualization


def gnnexplainer_subgraph(explainer, pyg_data, target_node, labels, features,
                          instance_level_explanation_subgraph_path, attack_subgraph_edge_num,
                          ex_type='clean'):
    start_time = time.time()
    # generate explanation for target node from specified explainer
    subset, edge_index_sub, mapping, _ = k_hop_subgraph(
        node_idx=target_node,
        num_hops=3,
        edge_index=pyg_data.edge_index,
        relabel_nodes=True,
        num_nodes=pyg_data.num_nodes
    )

    # 创建子图特征
    x_sub = pyg_data.x[subset]

    # 全节点映射字典
    full_mapping = {int(orig_id): idx for idx, orig_id in enumerate(subset.tolist())}

    # 目标节点的新ID
    target_new_id = full_mapping[target_node]  # 若 node_idx 是单个节点

    # 执行解释
    explanation = explainer(
        x=x_sub,
        edge_index=edge_index_sub,
        index=target_new_id
    )

    # explanation = explainer(
    #     x=pyg_data.x,
    #     edge_index=pyg_data.edge_index,
    #     index=target_node
    # )

    # get mask of edges an nodes
    edge_mask = explanation.edge_mask
    node_mask = explanation.node_mask

    elapsed = time.time() - start_time
    print(f"explainer generates results in {elapsed:.4f}s!")

    # choose top important edges or nodes by thresholds
    # Dynamic calculation of threshold (selecting the top 20% important edges)
    # threshold = torch.quantile(explanation.edge_mask, 0.9)
    # important_edges = edge_mask > threshold
    # important_nodes = node_mask > threshold
    # print(f"Edge mask shape: {explanation.edge_mask.shape}")
    # print(f"Node mask shape: {explanation.node_mask.shape}")

    # Visualize subgraph
    # res = os.path.abspath(__file__)
    # base_path = os.path.dirname(os.path.dirname(os.path.dirname(res)))
    explanation_subgraph = explanation_subgraph_visualization(explanation, target_node, edge_mask,
                                                              labels, features, node_mask, attack_subgraph_edge_num,
                                                              ex_type=ex_type,
                                                              pic_path=instance_level_explanation_subgraph_path,
                                                              full_mapping=full_mapping)

    return explanation_subgraph
