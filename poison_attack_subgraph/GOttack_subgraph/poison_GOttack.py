#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/7/3 09:52
# @Author   : **
# @Email    : **@**
# @File     : poison_GOttack.py
# @Software : PyCharm
# @Desc     :
"""
import os
import random
import sys
import warnings
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from attack.GOttack.OrbitAttack import OrbitAttack
from evasion_attack_subgraph.GOttack_subgraph.evasion_GOttack import select_nodes, set_up_surrogate_model
from utilty.attack_visualization import visualize_restricted_attack_subgraph
from attack.GOttack.orbit_table_generator import OrbitTableGenerator
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import classification_margin

warnings.filterwarnings("ignore")


def poison_test_acc_GCN(adj, features, data, target_node):
    idx_train, idx_val = data.idx_train, data.idx_val
    labels = data.labels
    ''' test on GCN '''
    gcn = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    gcn.eval()
    output = gcn.predict()
    probs = torch.exp(output[[target_node]])[0]
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    changed_label = output.argmax(1)[target_node]
    modified_gcn_model = gcn
    return acc_test.item(), changed_label, modified_gcn_model


def poison_attack(data, attack_model, target_node_list, budget, features, adj, labels, test_model, verbose=False):
    miss = 0
    false_class_node = []
    for target_node in tqdm(target_node_list):
        attack_model.attack(features, adj, labels, target_node, budget, verbose=verbose)
        modified_adj = attack_model.modified_adj
        if test_model == 'GCN':
            acc, changed_label, modified_gcn_model = poison_test_acc_GCN(modified_adj, features, data, target_node)
        else:
            raise Exception("Test model is not supported")

        if target_node in false_class_node:
            miss += 1
        elif acc == 0:
            miss += 1
            false_class_node.append(target_node)
        visualize_restricted_attack_subgraph(
            modified_adj,
            adj,
            labels,
            features,
            changed_label,
            target_node,
            'success',
            k_hop=2,
            max_nodes=10,
            title="Visualization for poison attack subgraph",
            pic_path=base_path + '/poison_attack_subgraph/results_bak/'
        )
        print("generate evasion attack subgraph for node {}".format(target_node))
    return miss / len(target_node_list)


if __name__ == '__main__':
    res = os.path.abspath(__file__)  # acquire absolute path of current file
    base_path = os.path.dirname(
        os.path.dirname(os.path.dirname(res)))  # acquire the parent path of current file's parent path
    sys.path.insert(0, base_path)

    method = ['1518']
    # budget_list = [5, 4, 3, 2, 1]
    budget_list = [5]
    random.seed(102)
    dataset_name = 'cora'  # ? (1)where can the dataset be downloaded? (2)how does the data of dpr orbit out generate?
    # For the question (2): the data of dpr orbit out is graph Orbit vector, and every line has 73 vectors of each node.
    df_orbit = OrbitTableGenerator(dataset_name).generate_orbit_table()  # acquire the orbit type of each node
    device = "cpu"

    print("INFO: Applying adversarial techniques {} on {} dataset with perturbation budget {} ".format(method,
                                                                                                       dataset_name,
                                                                                                       budget_list))

    ######################### Loading dataset  #########################
    data = Dataset(root=base_path + '/dataset', name=dataset_name)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    rowlist = []
    test_model = 'GCN'  # GSAGE, GCN, GIN

    for budget in budget_list:  # out-layer given a fixed budget, it's an isolated test
        row = []
        target_node, target_gcn = select_nodes(features, adj, labels, idx_train, idx_val, idx_test, device)  # select target nodes according to candidate nodes from test data.

        # Orbit attack(1518)
        surrogate = set_up_surrogate_model(features, adj, labels, idx_train, idx_val, device=device)  # 代理损失:gnn model
        model = OrbitAttack(surrogate, df_orbit, nnodes=adj.shape[0], device=device)  # initialize the attack model
        model = model.to(device)
        miss_percentage = poison_attack(data, model, target_node, budget, features, adj, labels, test_model)
        row.append(miss_percentage)

        rowlist.append(row)

    # Save results
    result = pd.DataFrame(rowlist, columns=method, index=budget_list)
    result.to_csv('./results/result_{}_{}.csv'.format(test_model, dataset_name))
