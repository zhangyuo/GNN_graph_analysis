#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/9/9 21:29
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : gnnexplainer_cf_subgraph.py
# @Software : PyCharm
# @Desc     :
"""
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
import time
from datetime import datetime

import torch
import numpy as np
from deeprobust.graph.data import Dataset
from tqdm import tqdm
from config.config import *
from model.GCN import GCN_model, dr_data_to_pyg_data, GCNtoPYG, load_GCN_model
from utilty.utils import normalize_adj, select_test_nodes
from instance_level_explanation_subgraph.GNNExplainer_subgraph.generate_gnnexplainer_subgraph import \
    generate_gnnexplainer_cf_subgraph
from subgraph_quantify.graph_analysis import gnn_explainer_generate

if __name__ == '__main__':
    res = os.path.abspath(__file__)  # acquire absolute path of current file
    base_path = os.path.dirname(
        os.path.dirname(os.path.dirname(res)))  # acquire the parent path of current file's parent path
    sys.path.insert(0, base_path)

    ######################### initialize random state  #########################
    dataset_name = DATA_NAME
    test_model = TEST_MODEL
    device = DEVICE
    nhid = HIDDEN_CHANNELS
    dropout = DROPOUT
    lr = LEARNING_RATE
    weight_decay = WEIGHT_DECAY
    with_bias = WITH_BIAS
    gcn_layer = GCN_LAYER
    attack_type = ATTACK_TYPE
    explanation_type = EXPLANATION_TYPE
    attack_method = ATTACK_METHOD
    attack_budget_list = ATTACK_BUDGET_LIST
    explainer_method = "GNNExplainer"

    np.random.seed(SEED_NUM)

    time_name = datetime.now().strftime("%Y-%m-%d")
    # counterfactual explanation subgraph path
    counterfactual_explanation_subgraph_path = base_path + f'/results/{time_name}/counterfactual_subgraph/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{attack_budget_list}'
    if not os.path.exists(counterfactual_explanation_subgraph_path):
        os.makedirs(counterfactual_explanation_subgraph_path)

    ######################### Loading dataset  #########################
    data = None
    # dataset path
    dataset_path = base_path + '/dataset'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    # adjacency matrix is a high compressed sparse row format
    if dataset_name == 'cora':
        data = Dataset(root=dataset_path, name=dataset_name)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    else:
        adj, features, labels = None, None, None
        idx_train, idx_val, idx_test = None, None, None

    ######################### Loading GCN model  #########################
    model_save_path = f'{base_path}/model_save/{test_model}/{dataset_name}/{gcn_layer}-layer/'
    file_path = os.path.join(model_save_path, 'gcn_model.pth')
    gnn_model = load_GCN_model(file_path, features, labels, nhid, dropout, device, lr, weight_decay,
                               with_bias, gcn_layer)
    dense_adj = torch.tensor(adj.toarray())
    norm_adj = normalize_adj(dense_adj)
    pre_output = gnn_model.forward(torch.tensor(features.toarray()), norm_adj)

    # Create PyG Data object
    pyg_data = dr_data_to_pyg_data(adj, features, labels)

    ######################### select test nodes  #########################
    target_node_list, target_node_list1 = select_test_nodes(attack_type, explanation_type, idx_test, pre_output, labels)
    target_node_list = target_node_list + target_node_list1
    # target_node_list = target_node_list[384:500]

    ######################### GNN explainer generate  #########################
    explainer = gnn_explainer_generate(gnn_model, device, features, labels, gcn_layer)

    ######################### GNN explainer generate  #########################
    # Get CF examples in test set
    start_0 = time.time()
    test_cf_examples = []
    # cfexp_subgraph = {}
    time_list = []
    for target_node in tqdm(target_node_list):
        cf_example, time_cost = generate_gnnexplainer_cf_subgraph(target_node, gcn_layer, pyg_data, explainer,
                                                                  gnn_model, pre_output)
        # print(cf_example)
        print("Time for {} epochs of one example: {:.4f}s".format(200, time_cost))
        time_list.append(time_cost)
        # cfexp_subgraph[target_node] = cf_example["subgraph"] if cf_example else None
        test_cf_examples.append({"data": cf_example, "time_cost": time_cost})
    print("Total time elapsed: {:.4f}min".format((time.time() - start_0) / 60))
    print("Number of CF examples found: {}/{}".format(len(test_cf_examples), len(target_node_list)))

    # with open(counterfactual_explanation_subgraph_path + "/cfexp_subgraph.pickle", "wb") as fw:
    #     pickle.dump(cfexp_subgraph, fw)

    # Save CF examples in test set
    with open(counterfactual_explanation_subgraph_path + "/{}_cf_examples_gcnlayer{}_lr{}_seed{}".format(
            DATA_NAME,
            GCN_LAYER,
            LEARNING_RATE,
            SEED_NUM), "wb") as f:
        pickle.dump(test_cf_examples, f)
