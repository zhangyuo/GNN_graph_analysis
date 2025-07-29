#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/7/17 12:43
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : cfexplainer_subgraph.py
# @Software : PyCharm
# @Desc     :
"""
import os
import sys
import time
import torch
import numpy as np
from deeprobust.graph.data import Dataset
from torch_geometric.utils import k_hop_subgraph

from model.gcn_model import GCN_model
from model.model_transfer import dr_data_to_pyg_data
from subgraph_quantify.graph_analysis import select_test_nodes, GCNtoPYG


def get_neighbourhood(param, edge_index, param1, features, labels):
    # generate explanation for target node from specified explainer
    subset, edge_index_sub, mapping, _ = k_hop_subgraph(
        node_idx=target_node,
        num_hops=3,
        edge_index=pyg_data.edge_index,
        relabel_nodes=True,
        num_nodes=pyg_data.num_nodes
    )

    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index[0])  # Get all nodes involved
    edge_subset_relabel = subgraph(edge_subset[0], edge_index[0], relabel_nodes=True)  # Get relabelled subset of edges
    sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
    sub_feat = features[edge_subset[0], :]
    sub_labels = labels[edge_subset[0]]
    new_index = np.array([i for i in range(len(edge_subset[0]))])
    node_dict = dict(zip(edge_subset[0].numpy(), new_index))  # Maps orig labels to new
    # print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
    return sub_adj, sub_feat, sub_labels, node_dict


def generate_cfexplainer_subgraph(target_node, edge_index, features, labels):
    start = time.time()
    sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(target_node, edge_index,features, labels)
    new_idx = node_dict[target_node]


if __name__ == '__main__':
    res = os.path.abspath(__file__)  # acquire absolute path of current file
    base_path = os.path.dirname(
        os.path.dirname(os.path.dirname(res)))  # acquire the parent path of current file's parent path
    sys.path.insert(0, base_path)

    ######################### initialize random state  #########################
    np.random.seed(102)

    ######################### initialize data experiment  #########################
    # general parameters
    dataset_name = 'cora'
    test_model = 'GCN'  # GSAGE, GCN, GIN
    device = "cpu"

    # explainer parameters
    explanation_type = 'counterfactual'  # ['instance-level', 'class-level', 'counterfactual']
    explainer_method = 'CFExplainer'  # ['GNNExplainer', 'CFExplainer']

    ######################### Loading dataset  #########################
    data = Dataset(root=base_path + '/dataset', name=dataset_name)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    # Create PyG Data object
    pyg_data = dr_data_to_pyg_data(adj, features, labels)

    gnn_model = None

    ######################### GNN model generate  #########################
    ori_output = None
    if test_model == 'GCN':
        gnn_model, ori_output = GCN_model(adj, features, labels, device, idx_train, idx_val)
    file_path = os.path.join('.', 'gcn_model.pth')
    torch.save(gnn_model.state_dict(), file_path)

    # transfer deeprobust GCN model into torch_geometric GCN model, 2-layer GCN model is used to train cf-explainer
    pyg_gcn = GCNtoPYG(gnn_model, device, features, labels)
    pyg_gcn.eval()

    pyg_logits = pyg_gcn.forward(pyg_data.x, pyg_data.edge_index)
    y_pred_orig = pyg_logits.argmax(dim=1)
    print("y_true counts: {}".format(np.unique(labels, return_counts=True)))
    print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.numpy(), return_counts=True)))

    ######################### select test nodes  #########################
    attack_type = '1'
    target_node_list = select_test_nodes(attack_type, explanation_type, idx_test, ori_output, labels)
    target_node_list = target_node_list[0:20]

    ######################### GNN explainer generate  #########################
    # Get CF examples in test set
    test_cf_examples = []
    for target_node in target_node_list:
        edge_index =pyg_data.edge_index
        generate_cfexplainer_subgraph(target_node, edge_index, features, labels)