#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/20 15:22
# @Author   : **
# @Email    : **@**
# @File     : test_model.py
# @Software : PyCharm
# @Desc     :
"""
import os
import sys
import time

import torch.nn as nn
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from scipy.sparse import issparse
import scipy.sparse as sp
from deeprobust.graph.data import Dataset
import numpy as np
from torch_geometric.utils import k_hop_subgraph, add_self_loops

from model.GCN import GCN_model


class PyGCompatibleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, bias=True)  # 强制启用偏置
        self.conv2 = GCNConv(hidden_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)


def transfer_weights(dr_model, pyg_model):
    print("PyG模型结构验证:")
    print(f"conv1.lin存在: {hasattr(pyg_model.conv1, 'lin')}")  # 应为True
    print(f"conv1.lin.weight形状: {pyg_model.conv1.lin.weight.shape}")
    print(f"DeepRobust gc1.weight形状: {dr_model.gc1.weight.shape}")
    """ 处理权重矩阵转置 + 偏置安全迁移 """
    # 第一层权重转置 (1433,16) -> (16,1433)
    pyg_model.conv1.lin.weight.data = dr_model.gc1.weight.data.t().clone()

    # 安全处理偏置
    if hasattr(pyg_model.conv1.lin, 'bias') and pyg_model.conv1.lin.bias is not None:
        pyg_model.conv1.lin.bias.data.zero_()
        if hasattr(dr_model.gc1, 'bias') and dr_model.gc1.bias is not None:
            pyg_model.conv1.lin.bias.data.copy_(dr_model.gc1.bias.data)

    # 第二层权重转置 (16,7) -> (7,16)
    pyg_model.conv2.lin.weight.data = dr_model.gc2.weight.data.t().clone()

    if hasattr(pyg_model.conv2.lin, 'bias') and pyg_model.conv2.lin.bias is not None:
        pyg_model.conv2.lin.bias.data.zero_()
        if hasattr(dr_model.gc2, 'bias') and dr_model.gc2.bias is not None:
            pyg_model.conv2.lin.bias.data.copy_(dr_model.gc2.bias.data)
    return pyg_model


def adj_to_edge_index(adj):
    """
    transfer adjacency matrix in deeprobust data to edge_index in pyg data
    :param adj:
    :return:
    """
    coo_adj = sp.coo_matrix(adj)
    # 使用np.vstack提高效率
    edge_array = np.vstack([coo_adj.row, coo_adj.col])
    return torch.tensor(edge_array, dtype=torch.long)


def dr_data_to_pyg_data(adj, features, labels):
    """
    transfer deeprobust data to pyg data
    :return:
    """
    edge_index = adj_to_edge_index(adj)
    features_dense = features.toarray() if issparse(features) else features
    pyg_data = Data(
        x=torch.tensor(features_dense, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels)
    )
    return pyg_data


if __name__ == '__main__':
    # Set up paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, base_path)

    # Load deeprobust data
    device = "cpu"
    dataset_name = 'cora'
    data_robust = Dataset(root=base_path + '/dataset', name=dataset_name)
    adj, features, labels = data_robust.adj, data_robust.features, data_robust.labels
    idx_train, idx_val, idx_test = data_robust.idx_train, data_robust.idx_val, data_robust.idx_test

    # 初始化PyG模型
    print(f"特征矩阵类型: {type(features)}")  # 应为scipy.sparse.csr.csr_matrix
    print(f"邻接矩阵类型: {type(adj)}")  # 应为scipy.sparse.csr.csr_matrix
    pyg_gcn = PyGCompatibleGCN(
        in_channels=features.shape[1],
        hidden_channels=16,
        out_channels=labels.max().item() + 1
    )
    pyg_gcn = pyg_gcn.to(device)

    # 迁移参数
    dr_trained_model, output = GCN_model(adj, features, labels, device, idx_train, idx_val)
    pyg_model = transfer_weights(dr_trained_model, pyg_gcn)

    # 创建PyG Data对象
    edge_index = adj_to_edge_index(adj)
    features_dense = features.toarray() if issparse(features) else features
    pyg_data = Data(
        x=torch.tensor(features_dense, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels)
    )

    # 预测一致性检查
    # dr_pred = dr_trained_model.predict(features, adj)
    dr_logits = torch.tensor(output, device=device)  # 确保同设备
    dr_pred = dr_logits.argmax(dim=1)

    pyg_logits = pyg_gcn(pyg_data.x, pyg_data.edge_index)
    pyg_pred = pyg_logits.argmax(dim=1)

    accuracy = (dr_pred == pyg_pred).float().mean()
    print(f"验证集预测一致性: {accuracy.item() * 100:.2f}%")

    # Create explainer (using PyG-formatted data)
    explainer = Explainer(
        model=pyg_gcn,
        algorithm=GNNExplainer(
            epochs=100,  # 减少训练轮次
            lr=0.1,  # 提高学习率
            log=False,  # 禁用日志
            coeffs={'edge_size': 0.005, 'node_feat_size': 0.1}  # 添加正则化防止梯度爆炸
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

    start_time = time.time()
    target_node = 1544
    # generate explanation for target node from specified explainer
    subset, edge_index_sub, mapping, _ = k_hop_subgraph(
        node_idx=target_node,
        num_hops=3,
        edge_index=pyg_data.edge_index,
        relabel_nodes=False,  # 关键：禁用节点重编号
        num_nodes=pyg_data.num_nodes
    )

    # 创建映射字典（原始节点ID → 子图索引）
    id_mapping = {idx: int(orig_id) for idx, orig_id in enumerate(subset.tolist())}
    reverse_mapping = {int(node_id): idx for idx, node_id in enumerate(subset.tolist())}

    # 创建子图特征
    x_sub = pyg_data.x[subset]

    # 目标节点在子图中的位置
    target_idx_in_subgraph = reverse_mapping[target_node]

    # 创建映射后的边索引（原始ID → 子图索引）
    edge_index_sub_mapped = torch.zeros_like(edge_index_sub)
    for i in range(edge_index_sub.size(1)):
        edge_index_sub_mapped[0, i] = reverse_mapping[edge_index_sub[0, i].item()]
        edge_index_sub_mapped[1, i] = reverse_mapping[edge_index_sub[1, i].item()]

    # 手动添加自环（解决GCNConv归一化问题）
    edge_index_sub_mapped, _ = add_self_loops(edge_index_sub_mapped, num_nodes=len(subset))

    # 执行解释
    explanation = explainer(
        x=x_sub,
        edge_index=edge_index_sub_mapped,
        index=target_idx_in_subgraph
    )

    # explanation = explainer(
    #     x=pyg_data.x,
    #     edge_index=pyg_data.edge_index,
    #     index=1544
    # )

    # get mask of edges an nodes
    edge_mask = explanation.edge_mask
    node_mask = explanation.node_mask

    # 打印原始节点ID
    print("子图节点ID与原始ID映射关系:")
    for sub_id, orig_id in id_mapping.items():
        print(f"子图ID {sub_id} -> 原始ID {orig_id}")

    # 验证目标节点ID保持不变
    print(f"目标节点在子图中的ID: {target_idx_in_subgraph} (对应原始ID {target_node})")

    elapsed = time.time() - start_time
    print(f"explainer generates results in {elapsed:.4f}s!")
