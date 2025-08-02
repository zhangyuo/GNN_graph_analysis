#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/20 11:31
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
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

from model.GCN import GCN_model, PyGCompatibleGCN, transfer_weights, adj_to_edge_index

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

# # Convert adjacency matrix to PyG's edge_index [Critical step]
# coo_adj = sp.coo_matrix(adj)  # Convert to COO format
# coo_fea = sp.coo_matrix(features)
# fea_indices = torch.tensor([coo_fea.row, coo_fea.col], dtype=torch.long)
# fea_values = torch.tensor(coo_fea.data, dtype=torch.float)
# fea_shape = coo_fea.shape
#
# edge_index = torch.tensor([coo_adj.row, coo_adj.col], dtype=torch.long)  # Extract row and column indices
# values = torch.tensor(coo_fea.data, dtype=torch.float)

# Create PyG Data object
edge_index = adj_to_edge_index(adj)
features_dense = features.toarray() if issparse(features) else features
pyg_data = Data(
    x=torch.tensor(features_dense, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(labels)
)

# initialize pyg model
pyg_gcn = PyGCompatibleGCN(
    in_channels=features.shape[1],
    hidden_channels=16,
    out_channels=labels.max().item() + 1
)
pyg_gcn = pyg_gcn.to(device)

# Initialize model (using deeprobust adjacency matrix)
dr_trained_model, output = GCN_model(adj, features, labels, device, idx_train, idx_val)
pyg_gcn = transfer_weights(dr_trained_model, pyg_gcn)

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
