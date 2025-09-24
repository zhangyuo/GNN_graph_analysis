#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/20 11:31
# @Author   : **
# @Email    : **@**
# @File     : GNNExplainer.py
# @Software : PyCharm
# @Desc     :
"""
import os
import sys
import time

import torch
import warnings
from deeprobust.graph.data import Dataset
from scipy.sparse import issparse
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

from config.config import DROPOUT, DEVICE
from model.GCN import GCN_model, PyGCompatibleGCN, transfer_weights, adj_to_edge_index, dr_data_to_pyg_data, GCNtoPYG

from utilty.explanation_visualization import explanation_subgraph_visualization

# Ignore warnings
warnings.filterwarnings("ignore")

# Set up paths
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

# Load deeprobust data
device = "cpu"
dataset_name = 'cora'
data_robust = Dataset(root=base_path + '/dataset', name=dataset_name)
adj, features, labels = data_robust.adj, data_robust.features, data_robust.labels
idx_train, idx_val, idx_test = data_robust.idx_train, data_robust.idx_val, data_robust.idx_test

# Initialize model (using deeprobust adjacency matrix)
gnn_model, output = GCN_model(adj, features, labels, device, idx_train, idx_val)

# Create PyG Data object
pyg_data = dr_data_to_pyg_data(adj, features, labels)

# initialize pyg model
pyg_gcn = GCNtoPYG(gnn_model, device, features, labels)

# Create explainer (using PyG-formatted data)
explainer = Explainer(
    model=pyg_gcn,
    algorithm=GNNExplainer(
        epochs=200,  # 减少训练轮次
        # lr=0.1,  # 提高学习率
        log=False,  # 禁用日志
        # coeffs={'edge_size': 0.005, 'node_feat_size': 0.1}  # 添加正则化防止梯度爆炸
    ),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs'
    )
)

# generate explanation for target node from specified explainer
target_node = 2182
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
threshold = 0.9  # given a threshold
# Dynamic calculation of threshold (selecting the top 20% important edges)
# threshold = torch.quantile(explanation.edge_mask, 0.9)
# important_edges = edge_mask > threshold
# important_nodes = node_mask > threshold
# print(f"Edge mask shape: {explanation.edge_mask.shape}")
# print(f"Node mask shape: {explanation.node_mask.shape}")

# w = edge_mask[important_edges]
# print(w.numpy())

# plt.hist(explanation.edge_mask.detach().numpy(), bins=20)
# plt.title("Edge Mask Value Distribution")
# plt.show()

# Create subgraph object by important nodes
# subgraph = Data(
#     x=pyg_data.x[important_nodes],
#     edge_index=pyg_data.edge_index[:, important_edges],
#     edge_attr=pyg_data.edge_attr[important_edges] if pyg_data.edge_attr is not None else None
# )

# Visualize subgraph
attack_subgraph_edge_num = 5
explanation_subgraph_visualization(explanation, target_node, edge_mask, labels, features,
                                   node_mask, attack_subgraph_edge_num, ex_type='clean',
                                   pic_path=base_path + '/',
                                   full_mapping=full_mapping)

print("ok")
