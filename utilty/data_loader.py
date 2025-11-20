#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/8/29 12:15
# @Author   : **
# @Email    : **@**
# @File     : data_loader.py
# @Software : PyCharm
# @Desc     :
"""
import os
import sys

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import WikipediaNetwork

if __name__ == '__main__':
    res = os.path.abspath(__file__)  # acquire absolute path of current file
    base_path = os.path.dirname(os.path.dirname(res))
    sys.path.insert(0, base_path)

    # dataset = PygNodePropPredDataset(name="ogbn-arxiv")
    # graph = dataset[0]
    # print(graph)
    # print(graph.x.shape)
    # print(graph.node_year.shape)
    #
    # node_index = 100
    # node_feat_i = graph.x[node_index]
    # print(f"Node {node_index} feature shape: {node_feat_i.shape}")
    # print(f"Node {node_index} feature: {node_feat_i}")
    #
    # node_year_i = graph.node_year[node_index]
    # print(f"Node {node_index} year: {node_year_i.item()}")
    #
    # split_idx = dataset.get_idx_split()
    # # Dictionary containing train/valid/test indices.
    # train_idx = split_idx["train"]
    # # torch.tensor storing a list of training indices.
    # pass

    dataset = WikipediaNetwork(root=base_path + "/dataset", name='chameleon')
    data = dataset[0]  # only one graph in dataset
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Graph is undirected: {data.is_undirected()}")
    print(data.x)
    print(data.edge_index)
    print(data.edge_attr)
    print(data.y)
