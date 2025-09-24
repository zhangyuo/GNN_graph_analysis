#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/24 09:27
# @Author   : **
# @Email    : **@**
# @File     : gnn_graph_generate.py
# @Software : PyCharm
# @Desc     :
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(res)
sys.path.insert(0, base_path)
from datetime import datetime
import numpy as np
import torch
import pickle
import pandas as pd
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from model.GCN import GCN_model, dr_data_to_pyg_data, load_GCN_model
from config.config import *
from subgraph_quantify.graph_analysis import gnn_explainer_generate, generate_subgraph
from utilty.attack_visualization import generate_timestamp_key
from utilty.utils import normalize_adj, select_test_nodes

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
    attack_type = ATTACK_TYPE
    attack_method = ATTACK_METHOD
    attack_budget_list = ATTACK_BUDGET_LIST
    explainer_method = EXPLAINER_METHOD
    explanation_type = EXPLANATION_TYPE

    np.random.seed(SEED_NUM)

    ######################### loading deeprobust dataset  #########################
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

        # Create PyG Data object
        pyg_data = dr_data_to_pyg_data(adj, features, labels)
    else:
        adj, features, labels = None, None, None
        idx_train, idx_val, idx_test = None, None, None

    # create data results path
    time_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # time_name = datetime.now().strftime("%Y-%m-%d") + "_E-_for_cfexp"

    # clean subgraph path
    clean_subgraph_path = base_path + f'/results/{time_name}/clean_subgraph/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{attack_budget_list}'
    if not os.path.exists(clean_subgraph_path):
        os.makedirs(clean_subgraph_path)

    # evasion attack subgraph path
    evasion_attack_subgraph_path = None
    if attack_type == 'Evasion':
        evasion_attack_subgraph_path = base_path + f'/results/{time_name}/evasion_attack_subgraph/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{attack_budget_list}'
        if not os.path.exists(evasion_attack_subgraph_path):
            os.makedirs(evasion_attack_subgraph_path)

    # instance level explanation subgraph path
    instance_level_explanation_subgraph_path = None
    if explanation_type == 'instance-level':
        instance_level_explanation_subgraph_path = base_path + f'/results/{time_name}/instance_level_subgraph/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{attack_budget_list}'
        if not os.path.exists(instance_level_explanation_subgraph_path):
            os.makedirs(instance_level_explanation_subgraph_path)

    # poison attack subgraph path
    poison_attack_subgraph_path = None
    if attack_type == 'Poison':
        poison_attack_subgraph_path = base_path + f'/results/{time_name}/poison_attack_subgraph/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{attack_budget_list}'
        if not os.path.exists(poison_attack_subgraph_path):
            os.makedirs(poison_attack_subgraph_path)

    # class level explanation subgraph path
    class_level_explanation_subgraph_path = None
    if explanation_type == 'class-level':
        class_level_explanation_subgraph_path = base_path + f'/results/{time_name}/class_level_subgraph/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{attack_budget_list}'
        if not os.path.exists(class_level_explanation_subgraph_path):
            os.makedirs(class_level_explanation_subgraph_path)

    # counterfactual explanation subgraph path
    counterfactual_explanation_subgraph_path = None
    if explanation_type == 'counterfactual':
        counterfactual_explanation_subgraph_path = base_path + f'/results/{time_name}/counterfactual_subgraph/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{attack_budget_list}'
        if not os.path.exists(counterfactual_explanation_subgraph_path):
            os.makedirs(counterfactual_explanation_subgraph_path)

    # graph analysis subgraph path
    graph_analysis_subgraph_path = base_path + f'/results/{time_name}/subgraph_quantify/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{attack_budget_list}'
    if not os.path.exists(graph_analysis_subgraph_path):
        os.makedirs(graph_analysis_subgraph_path)

    ######################### Loading GCN model  #########################
    model_save_path = f'{base_path}/model_save/{test_model}/{dataset_name}/{gcn_layer}-layer/'
    file_path = os.path.join(model_save_path, 'gcn_model.pth')
    gnn_model = load_GCN_model(file_path, features, labels, nhid, dropout, device, lr, weight_decay,
                               with_bias, gcn_layer)
    dense_adj = torch.tensor(adj.toarray())
    norm_adj = normalize_adj(dense_adj)
    pre_output = gnn_model.forward(torch.tensor(features.toarray()), norm_adj)

    ######################### select test nodes  #########################
    target_node_list, target_node_list1 = select_test_nodes(attack_type, explanation_type, idx_test, pre_output, labels)
    target_node_list += target_node_list1
    # target_node_list = target_node_list[100:105]

    ######################### GNN explainer generate  #########################
    explainer = gnn_explainer_generate(gnn_model, device, features, labels, gcn_layer)

    ######################### generate subgraph  #########################
    subgraph_data = generate_subgraph(attack_type, explanation_type, target_node_list, gnn_model, explainer, pyg_data,
                                      device, test_model, attack_method, attack_budget_list, explainer_method, data,
                                      features, adj, labels, idx_train, idx_val, gcn_layer, dataset_name,
                                      clean_subgraph_path,
                                      evasion_attack_subgraph_path, instance_level_explanation_subgraph_path,
                                      poison_attack_subgraph_path, pre_output, idx_test, with_bias, counterfactual_explanation_subgraph_path)
    with open(graph_analysis_subgraph_path + "/subgraph_data.pickle", "wb") as fw:
        pickle.dump(subgraph_data, fw)
