#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/24 09:27
# @Author   : **
# @Email    : **@**
# @File     : gnn_graph_analysis.py
# @Software : PyCharm
# @Desc     :
"""
import os
import sys
import warnings
from datetime import datetime

from utilty.utils import normalize_adj

warnings.filterwarnings("ignore")
res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(res)
sys.path.insert(0, base_path)
import numpy as np
import torch
import pickle
import pandas as pd
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from model.GCN import GCN_model, dr_data_to_pyg_data, load_GCN_model, GCNtoPYG
from config.config import *
from subgraph_quantify.graph_analysis import select_test_nodes, subgraph_quantify

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

    # create data results path
    time_name = datetime.now().strftime("%Y-%m-%d") + "_origin"

    # graph analysis subgraph path
    graph_analysis_subgraph_path = base_path + f'/results/{time_name}/subgraph_quantify/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{attack_budget_list}'
    if not os.path.exists(graph_analysis_subgraph_path):
        os.makedirs(graph_analysis_subgraph_path)

    # counterfactual explanation subgraph path
    counterfactual_explanation_subgraph_path = base_path + f'/results/{time_name}/counterfactual_subgraph/{attack_type}_{attack_method}_{"counterfactual"}_{"CFExplainer"}_{dataset_name}_budget{attack_budget_list}'
    if not os.path.exists(counterfactual_explanation_subgraph_path):
        os.makedirs(counterfactual_explanation_subgraph_path)

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
    pyg_gcn = GCNtoPYG(gnn_model, device, features, labels, gcn_layer)

    ######################### select test nodes  #########################
    target_node_list, target_node_list1 = select_test_nodes(attack_type, explanation_type, idx_test, pre_output, labels)

    ######################### subgraph quantify  #########################
    with open(graph_analysis_subgraph_path + "/subgraph_data.pickle", "rb") as fr:
        subgraph_data = pickle.load(fr)
    clean_subgraph = subgraph_data['clean_subgraph']
    attack_subgraph = subgraph_data['attack_subgraph']
    clean_explanation_subgraph = subgraph_data['clean_explanation_subgraph']
    attacked_explanation_subgraph = subgraph_data['attacked_explanation_subgraph']

    with open(counterfactual_explanation_subgraph_path + "/cfexp_subgraph.pickle", "rb") as fr:
        cfexp_subgraph = pickle.load(fr)

    # # attack subgraph vs. clean explanation subgraph
    similarity_dict_1 = subgraph_quantify(attack_subgraph, clean_explanation_subgraph, pyg_data, pyg_gcn,
                                          graph_analysis_subgraph_path, target_node_list, target_node_list1,
                                          pic_name='at_vs_cl-ex')

    # attack subgraph vs. attacked explanation subgraph
    similarity_dict_2 = subgraph_quantify(attack_subgraph, attacked_explanation_subgraph, pyg_data, pyg_gcn,
                                          graph_analysis_subgraph_path, target_node_list, target_node_list1,
                                          pic_name='at_vs_at-ex')

    # attack subgraph vs. counterfactual explanation subgraph
    similarity_dict_3 = subgraph_quantify(attack_subgraph, cfexp_subgraph, pyg_data, pyg_gcn,
                                          graph_analysis_subgraph_path, target_node_list, target_node_list1,
                                          pic_name='at_vs_cf-ex', is_cf_explainer=True)

    # Save all results
    result1 = pd.DataFrame(similarity_dict_1.values(),
                           columns=['trained_gcn_test_state', 'attack_state', 'edge_attack_type', 'ged', 'mcs',
                                    'gev_at_ex',
                                    'gev_at_mcs', 'gev_ex_mcs'],
                           index=similarity_dict_1.keys())
    result1.to_csv(graph_analysis_subgraph_path + '/result_{}_{}_test.csv'.format('all', 'at_vs_cl-ex'))

    result2 = pd.DataFrame(similarity_dict_2.values(),
                           columns=['trained_gcn_test_state', 'attack_state', 'edge_attack_type', 'ged', 'mcs',
                                    'gev_at_ex',
                                    'gev_at_mcs', 'gev_ex_mcs'],
                           index=similarity_dict_2.keys())
    result2.to_csv(graph_analysis_subgraph_path + '/result_{}_{}_test.csv'.format('all', 'at_vs_at-ex'))

    result3 = pd.DataFrame(similarity_dict_3.values(),
                           columns=['trained_gcn_test_state', 'attack_state', 'edge_attack_type', 'ged', 'mcs',
                                    'gev_at_ex',
                                    'gev_at_mcs', 'gev_ex_mcs'],
                           index=similarity_dict_3.keys())
    result3.to_csv(graph_analysis_subgraph_path + '/result_{}_{}_test.csv'.format('all', 'at_vs_cf-ex'))

    print("graph analysis done")
