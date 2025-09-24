#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/9/24 13:08
# @Author   : **
# @Email    : **@**
# @File     : homo_comp.py
# @Software : PyCharm
# @Desc     :
"""
import torch
import numpy as np
from ogb.nodeproppred import NodePropPredDataset
import os
import pickle
import sys

from model.GCN import dr_data_to_pyg_data
from deeprobust.graph.data import Dataset

from utilty.utils import CPU_Unpickler

res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(os.path.dirname(res))  # acquire the parent path of current file's parent path
sys.path.insert(0, base_path)


def edge_homophily(edge_index, labels):
    src, dst = edge_index
    same_label = (labels[src] == labels[dst]).sum()
    return (same_label / len(src)).item()


dataset_path = base_path + '/dataset'


# data = Dataset(root=dataset_path, name='cora')
# adj, features, labels = data.adj, data.features, data.labels
# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
# # Create PyG Data object
# pyg_data = dr_data_to_pyg_data(adj, features, labels)
# homophily_ratio = edge_homophily(pyg_data.edge_index, pyg_data.y)


# # 加载ogbn-arxiv数据集
# dataset = NodePropPredDataset(name='ogbn-arxiv', root=dataset_path)
# graph, labels = dataset[0]
# # 计算边同质性
# homophily_ratio = edge_homophily(graph['edge_index'], labels)


# # Create PyG Data object
# with open(dataset_path + "/BAShapes.pickle", "rb") as f:
#     pyg_data = CPU_Unpickler(f).load()
#     homophily_ratio = edge_homophily(pyg_data.edge_index, pyg_data.y)

# # Create PyG Data object
# with open(dataset_path + "/TreeCycle.pickle", "rb") as f:
#     pyg_data = CPU_Unpickler(f).load()
#     homophily_ratio = edge_homophily(pyg_data.edge_index, pyg_data.y)

# Create PyG Data object
with open(dataset_path + "/LoanDecision.pickle", "rb") as f:
    pyg_data = CPU_Unpickler(f).load()
    homophily_ratio = edge_homophily(pyg_data.edge_index, pyg_data.y)

print(f"ogbn-arxiv Homophily Ratio: {homophily_ratio:.3f}")
