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
from torch_geometric.utils import k_hop_subgraph, subgraph, to_dense_adj

from model.GCN import GCN_model, dr_data_to_pyg_data
from subgraph_quantify.graph_analysis import select_test_nodes, GCNtoPYG
from config.config import HIDDEN_CHANNELS, DROP_OUT, BETA, OPTIMIZER, LEARNING_RATE, N_Momentum, NUM_EPOCHS
from explainer.cf_explanation.cf_explainer import CFExplainer


def get_neighbourhood(target_node, edge_index, features, labels):
    # generate explanation for target node from specified explainer
    subset, edge_index_sub, mapping, _ = k_hop_subgraph(
        node_idx=target_node,
        num_hops=3,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=pyg_data.num_nodes
    )

    sub_adj = to_dense_adj(edge_index_sub).squeeze()
    sub_feat = features[subset, :]
    sub_labels = labels[subset]
    node_dict = {int(orig_id): idx for idx, orig_id in enumerate(subset.tolist())}
    # print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
    return sub_adj, sub_feat, sub_labels, node_dict


def generate_cfexplainer_subgraph(target_node, pyg_data, adj, features, labels, output, model, device, idx_test):

    sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(target_node, pyg_data.edge_index, features, labels)
    new_idx = node_dict[target_node]
    sub_pyg_data = dr_data_to_pyg_data(sub_adj, sub_feat, sub_labels)
    print("Output original model, full adj: {}".format(output[target_node]))
    print("Output original model, sub adj: {}".format(model(sub_pyg_data.x, sub_pyg_data.adj)[new_idx]))
    # Need to instantitate new cf model every time because size of P changes based on size of sub_adj
    explainer = CFExplainer(model=model,
                            sub_adj=sub_adj,
                            sub_feat=sub_feat,
                            n_hid=HIDDEN_CHANNELS,
                            dropout=DROP_OUT,
                            sub_labels=sub_labels,
                            y_pred_orig=y_pred_orig[target_node],
                            num_classes=labels.max().item() + 1,
                            beta=BETA,
                            device=device)
    if device == 'cuda':
        model.cuda()
        explainer.cf_model.cuda()
        adj = adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_test = idx_test.cuda()

    cf_example = explainer.explain(node_idx=target_node, cf_optimizer=OPTIMIZER, new_idx=new_idx, lr=LEARNING_RATE,
                                   n_momentum=N_Momentum, num_epochs=NUM_EPOCHS)
    test_cf_examples.append(cf_example)
    print("Time for {} epochs of one example: {:.4f}min".format(NUM_EPOCHS, (time.time() - start) / 60))

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

    pyg_logits = pyg_gcn.forward(pyg_data.x, pyg_data.adj)
    y_pred_orig = pyg_logits.argmax(dim=1)
    print("y_true counts: {}".format(np.unique(labels, return_counts=True)))
    print("y_pred_orig counts: {}".format(np.unique(y_pred_orig.numpy(), return_counts=True)))

    ######################### select test nodes  #########################
    attack_type = '1'
    target_node_list = select_test_nodes(attack_type, explanation_type, idx_test, ori_output, labels)
    target_node_list = target_node_list[0:20]

    ######################### GNN explainer generate  #########################
    # Get CF examples in test set
    start = time.time()
    test_cf_examples = []
    for target_node in target_node_list:
        generate_cfexplainer_subgraph(target_node, pyg_data, adj, features, labels, pyg_logits, pyg_gcn, device, idx_test)
    print("Total time elapsed: {:.4f}s".format((time.time() - start) / 60))
    print("Number of CF examples found: {}/{}".format(len(test_cf_examples), len(idx_test)))