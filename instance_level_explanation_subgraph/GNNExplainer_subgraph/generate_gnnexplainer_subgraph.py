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
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.utils import add_self_loops, k_hop_subgraph
from torch_sparse import SparseTensor
from utilty.explanation_visualization import explanation_subgraph_visualization
from utilty.utils import normalize_adj


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


def generate_gnnexplainer_cf_subgraph(target_node, gcn_layer, pyg_data, explainer, gnn_model, pre_output, dataset_name):
    start_time = time.time()
    # generate explanation for target node from specified explainer
    subset, edge_index_sub, mapping, _ = k_hop_subgraph(
        node_idx=target_node,
        num_hops=gcn_layer + 1,
        edge_index=pyg_data.edge_index,
        relabel_nodes=True,
        num_nodes=pyg_data.num_nodes
    )

    sub_labels = pyg_data.y[subset]

    # 创建空邻接矩阵
    num_nodes = edge_index_sub.max() + 1
    sub_adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # 填充邻接矩阵
    for i in range(edge_index_sub.shape[1]):
        src, dst = edge_index_sub[:, i]
        # 无向图设置双向连接
        sub_adj[src, dst] = 1
        sub_adj[dst, src] = 1
    sub_adj = torch.tensor(sub_adj)

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

    # get mask of edges an nodes
    edge_mask = explanation.edge_mask
    node_mask = explanation.node_mask

    # time_cost = time.time() - start_time
    # print(f"factual explainer generates results in {time_cost:.4f}s!")

    cf_example = None
    threshold = 0
    edge_importance = edge_mask[edge_mask > threshold].detach().cpu().numpy()
    edge_index = explanation.edge_index[:, edge_mask > threshold].cpu().numpy()

    # 1. 将边索引和重要性值组合成元组列表
    edge_data = [(edge_index[:, i], edge_importance[i]) for i in range(edge_importance.shape[0])]

    # 2. 按重要性值降序排序
    sorted_edge_data = sorted(edge_data, key=lambda x: x[1], reverse=True)

    # 3. 分离出排序后的边索引和重要性值
    sorted_edge_index = np.array([data[0] for data in sorted_edge_data]).T  # 转置回(2, N)格式
    # sorted_edge_importance = np.array([data[1] for data in sorted_edge_data])

    cf_adj = sub_adj.clone()
    explanation_size = 0
    removed_edges = []
    target_node_label = pre_output[target_node].argmax().item()
    # norm_adj = normalize_adj(sub_adj)
    # target_node_label_1 = gnn_model.forward(x_sub, norm_adj)[target_new_id].argmax().item()

    max_edits = 5
    if dataset_name in ["BA-SHAPES", "TREE-CYCLES"]:
        max_edits = 6
    # 4. 遍历边索引，根据重要性降序分别删除对应，判断预测是否翻转
    for index, edge in enumerate(sorted_edge_index.T):  # 转置后每行代表一条边
        u, v = edge
        cf_adj[u][v] = 0
        cf_adj[v][u] = 0
        explanation_size += 1
        removed_edges.append((u, v))
        norm_adj = normalize_adj(cf_adj)
        y_new_output = gnn_model.forward(x_sub, norm_adj)
        new_idx_label = y_new_output[target_new_id].argmax().item()
        if new_idx_label != target_node_label:
            print("find counterfactual explanation")
            cf_example = {
                "success": True,
                "target_node": target_node,
                "new_idx": target_new_id,
                "added_edges": [],
                "removed_edges": removed_edges,
                "explanation_size": explanation_size,
                "original_pred": target_node_label,
                "new_pred": new_idx_label,
                "extended_adj": sub_adj,
                "cf_adj": cf_adj,
                "extended_feat": x_sub,
                "sub_labels": sub_labels
            }
            break
        if index + 1 == max_edits or index + 1 == len(sorted_edge_index.T):
            print("Don't find counterfactual explanation")
            cf_example = {
                "success": False,
                "target_node": target_node,
                "new_idx": target_new_id,
                "added_edges": [],
                "removed_edges": removed_edges,
                "explanation_size": explanation_size,
                "original_pred": target_node_label,
                "new_pred": new_idx_label,
                "extended_adj": sub_adj,
                "cf_adj": cf_adj,
                "extended_feat": x_sub,
                "sub_labels": sub_labels
            }
            break
    time_cost = time.time() - start_time

    return cf_example, time_cost
