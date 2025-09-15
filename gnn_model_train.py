#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/8/6 13:16
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : gnn_model_train.py
# @Software : PyCharm
# @Desc     : target node for node classification on GNN model
"""
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")
res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(res)
sys.path.insert(0, base_path)

import numpy as np
import torch
from deeprobust.graph.data import Dataset
from config.config import *
from model.GCN import GCN_model, load_GCN_model
from utilty.utils import normalize_adj, accuracy, CPU_Unpickler, BAShapesDataset, TreeCyclesDataset, \
    LoanDecisionDataset, OGBNArxivDataset
from ogb.nodeproppred import PygNodePropPredDataset

if __name__ == "__main__":
    dataset_name = DATA_NAME
    test_model = TEST_MODEL
    device = DEVICE
    nhid = HIDDEN_CHANNELS
    dropout = DROPOUT
    lr = LEARNING_RATE
    weight_decay = WEIGHT_DECAY
    with_bias = WITH_BIAS
    gcn_layer = GCN_LAYER

    np.random.seed(SEED_NUM)
    torch.manual_seed(SEED_NUM)

    ######################### loading deeprobust dataset  #########################
    data = None
    pyg_data = None
    # dataset path
    dataset_path = base_path + '/dataset'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    # adjacency matrix is a high compressed sparse row format
    if dataset_name == 'cora':
        data = Dataset(root=dataset_path, name=dataset_name)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    elif dataset_name == 'BA-SHAPES':
        # Create PyG Data object
        with open(dataset_path + "/BAShapes.pickle", "rb") as f:
            pyg_data = CPU_Unpickler(f).load()
        # Create deeprobust Data object
        data = BAShapesDataset(pyg_data)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    elif dataset_name == 'TREE-CYCLES':
        # Create PyG Data object
        with open(dataset_path + "/TreeCycle.pickle", "rb") as f:
            pyg_data = CPU_Unpickler(f).load()
        # Create deeprobust Data object
        data = TreeCyclesDataset(pyg_data)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    elif dataset_name == 'Loan-Decision':
        # Create PyG Data object
        with open(dataset_path + "/LoanDecision.pickle", "rb") as f:
            pyg_data = CPU_Unpickler(f).load()
        # Create deeprobust Data object
        data = LoanDecisionDataset(pyg_data)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    elif dataset_name == 'ogbn-arxiv':
        # Create PyG Data object
        ogbn_arxiv_data = PygNodePropPredDataset(name="ogbn-arxiv", root=dataset_path + '/loader/dataset/')
        pyg_data = ogbn_arxiv_data[0]
        # Create deeprobust Data object
        data = OGBNArxivDataset(ogbn_arxiv_data)
        # pyg_data = data.get_pyg_data()
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    else:
        adj, features, labels = None, None, None
        idx_train, idx_val, idx_test = None, None, None

    ######################### GNN model generate  #########################
    gnn_model = None
    pre_output = None
    # model save path
    model_save_path = f'{base_path}/model_save/{test_model}/{dataset_name}/{gcn_layer}-layer/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if not gnn_model and test_model == 'GCN':
        file_path = os.path.join(model_save_path, 'gcn_model.pth')
        if os.path.exists(file_path):
            gnn_model = load_GCN_model(file_path, features, labels, nhid, dropout, device, lr, weight_decay,
                                       with_bias, gcn_layer)
            dense_adj = torch.tensor(adj.toarray())
            norm_adj = normalize_adj(dense_adj)
            pre_output = gnn_model.forward(torch.tensor(features.toarray()), norm_adj)
        else:
            gnn_model, pre_output = GCN_model(adj, features, labels, device, idx_train, idx_val, nhid,
                                              dropout, lr, weight_decay, with_bias, gcn_layer)
            file_path = os.path.join(model_save_path, 'gcn_model.pth')
            torch.save(gnn_model.state_dict(), file_path)
            gnn_model.eval()
    elif not gnn_model and test_model == 'GraphSAGE':
        pass
    elif not gnn_model and test_model == 'GIN':
        pass
    else:
        pass

    y_pred_orig_gnn = pre_output.argmax(dim=1)
    print("y_true counts: {}".format(np.unique(labels[idx_test], return_counts=True)))
    print("y_pred_orig_gnn counts: {}".format(np.unique(y_pred_orig_gnn.numpy()[idx_test], return_counts=True)))
    acc_test = accuracy(y_pred_orig_gnn[idx_test], labels[idx_test])
    print("Test set results:", "accuracy = {:.4f}".format(acc_test))
