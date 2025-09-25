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

from ogb.nodeproppred import PygNodePropPredDataset

if __name__ == '__main__':
    dataset = PygNodePropPredDataset(name="ogbn-arxiv")
    graph = dataset[0]
    print(graph)
    print(graph.x.shape)
    print(graph.node_year.shape)

    node_index = 100
    node_feat_i = graph.x[node_index]
    print(f"Node {node_index} feature shape: {node_feat_i.shape}")
    print(f"Node {node_index} feature: {node_feat_i}")

    node_year_i = graph.node_year[node_index]
    print(f"Node {node_index} year: {node_year_i.item()}")

    split_idx = dataset.get_idx_split()
    # Dictionary containing train/valid/test indices.
    train_idx = split_idx["train"]
    # torch.tensor storing a list of training indices.
    pass