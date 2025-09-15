#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/19 11:38
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : evasion_GOttack.py
# @Software : PyCharm
# @Desc     :
"""
import os
import pickle
import random
import sys
res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(
    os.path.dirname(os.path.dirname(res)))  # acquire the parent path of current file's parent path
sys.path.insert(0, base_path)
import time
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from attack.GOttack.OrbitAttack import OrbitAttack
from model.GCN import dr_data_to_pyg_data, load_GCN_model
from utilty.attack_visualization import visualize_restricted_attack_subgraph
from attack.GOttack.orbit_table_generator import OrbitTableGenerator
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import classification_margin
from config.config import *
from utilty.utils import CPU_Unpickler, BAShapesDataset, TreeCyclesDataset, LoanDecisionDataset, normalize_adj, \
    select_test_nodes

warnings.filterwarnings("ignore")

if __name__ == '__main__':


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
    attack_method = "GOttack"
    attack_budget_list = ATTACK_BUDGET_LIST
    top_t = ATTACK_BUDGET_LIST[0]

    np.random.seed(SEED_NUM)
    torch.manual_seed(SEED_NUM)

    time_name = datetime.now().strftime("%Y-%m-%d")
    # counterfactual explanation subgraph path
    counterfactual_explanation_subgraph_path = base_path + f'/results/{time_name}/attack_subgraph/{attack_type}_{attack_method}_{dataset_name}_budget{attack_budget_list}'
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
        # Create PyG Data object
        pyg_data = dr_data_to_pyg_data(adj, features, labels)
    elif dataset_name == 'BA-SHAPES':
        # Create PyG Data object
        with open(dataset_path + "/BAShapes.pickle", "rb") as f:
            pyg_data = CPU_Unpickler(f).load()
        data = BAShapesDataset(pyg_data)
        # Create deeprobust Data object
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

    if gcn_layer != 2:
        # surrogate = set_up_surrogate_model(features, adj, labels, idx_train, idx_val, device=device)  # 代理损失:gnn model
        surrogate = gnn_model
    else:
        surrogate = gnn_model

    ######################### select test nodes  #########################
    target_node_list, target_node_list1 = select_test_nodes(dataset_name, attack_type, idx_test, pre_output, labels)
    target_node_list = target_node_list + target_node_list1
    target_node_list.sort()
    print(f"Test nodes number: {len(target_node_list)}, incorrect: {len(target_node_list1)}")
    # target_node_list = target_node_list[0:10]

    ######################### attack subgraph generate  #########################
    start_0 = time.time()
    df_orbit = OrbitTableGenerator(dataset_name).generate_orbit_table()
    time_list = []
    test_cf_examples = []
    mis_cases = 0
    for target_node in tqdm(target_node_list):
        start_time = time.time()
        attack_model = OrbitAttack(surrogate, df_orbit, nnodes=data.adj.shape[0], device=device, top_t=top_t,
                                   gcn_layer=gcn_layer)  # initialize the attack model
        attack_model = attack_model.to(device)
        attack_model.attack_cf(data.features, data.adj, data.labels, target_node, top_t)

        edited_edges = attack_model.structure_perturbations
        cf_adj = attack_model.modified_adj
        cf_adj = torch.tensor(cf_adj.toarray())
        extended_adj = data.adj
        extended_adj = torch.tensor(extended_adj.toarray())
        added_edges = []
        removed_edges = []

        for index, (u, v) in enumerate(edited_edges):
            if extended_adj[u,v] == 0.0:
                added_edges.append(edited_edges[index])
            else:
                removed_edges.append(edited_edges[index])

        norm_adj = normalize_adj(cf_adj)
        y_new_output = gnn_model.forward(pyg_data.x, norm_adj)
        target_node_label = pre_output[target_node].argmax().item()
        new_idx_label = y_new_output[target_node].argmax().item()

        if new_idx_label != target_node_label:
            print("find counterfactual explanation")
            cf_example = {
                "success": True,
                "target_node": target_node,
                "new_idx": target_node,
                "added_edges": added_edges,
                "removed_edges": removed_edges,
                "explanation_size": len(edited_edges),
                "original_pred": target_node_label,
                "new_pred": new_idx_label,
                "extended_adj": extended_adj,
                "cf_adj": cf_adj,
                "extended_feat": pyg_data.x,
                "sub_labels": pyg_data.y
            }
        else:
            print("Don't find counterfactual explanation")
            cf_example = {
                "success": False,
                "target_node": target_node,
                "new_idx": target_node,
                "added_edges": added_edges,
                "removed_edges": removed_edges,
                "explanation_size": len(edited_edges),
                "original_pred": target_node_label,
                "new_pred": new_idx_label,
                "extended_adj": extended_adj,
                "cf_adj": cf_adj,
                "extended_feat": pyg_data.x,
                "sub_labels": pyg_data.y
            }
        time_cost = time.time() - start_time

        print("Time for one example: {:.4f}s".format(time_cost))
        time_list.append(time_cost)
        test_cf_examples.append({"data": cf_example, "time_cost": time_cost})
        if cf_example['success']:
            mis_cases += 1

    print("Total time elapsed: {:.4f}min".format((time.time() - start_0) / 60))
    print("Number of CF examples found: {}/{}".format(mis_cases, len(target_node_list)))

    # Save results
    with open(
            counterfactual_explanation_subgraph_path + f"/{DATA_NAME}_cf_examples_gcnlayer{GCN_LAYER}_lr{LEARNING_RATE}_seed{SEED_NUM}",
            "wb") as f:
        pickle.dump(test_cf_examples, f)
