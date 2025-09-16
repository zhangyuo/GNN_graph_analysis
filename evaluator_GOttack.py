#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/9/4 17:17
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : evaluator_gnnexplainer.py
# @Software : PyCharm
# @Desc     :
"""
from __future__ import division
from __future__ import print_function
import sys

from tqdm import tqdm

from config.config import ATTACK_TYPE, ATTACK_METHOD, EXPLAINER_METHOD, EXPLANATION_TYPE, DATA_NAME, ATTACK_BUDGET_LIST, \
    TEST_MODEL, GCN_LAYER, HIDDEN_CHANNELS, DROPOUT, LEARNING_RATE, WEIGHT_DECAY, WITH_BIAS, DEVICE, SEED_NUM, α2, α3, \
    TAU_C, LEARNING_RATE_AC, k
from model.GCN import load_GCN_model, dr_data_to_pyg_data
from utilty.utils import normalize_adj, select_test_nodes, compute_deg_diff, compute_motif_viol, CPU_Unpickler, \
    BAShapesDataset, TreeCyclesDataset, LoanDecisionDataset
import numpy as np
import os
import pandas as pd
import pickle
import torch
from deeprobust.graph.data import Dataset

res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(res)
sys.path.insert(0, base_path)

######################### evaluated parameters setting  #########################
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
tau_c = TAU_C

np.random.seed(SEED_NUM)
torch.manual_seed(SEED_NUM)

######################### loading deeprobust dataset  #########################
data = None
# dataset path
dataset_path = base_path + '/dataset'
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
gnn_model.load_state_dict(torch.load(file_path))
gnn_model.eval()

dense_adj = torch.tensor(adj.toarray())
norm_adj = normalize_adj(dense_adj)
y_pred_orig = gnn_model.forward(torch.tensor(features.toarray()), norm_adj)

######################### select test nodes  #########################
target_node_list, target_node_list1 = select_test_nodes(dataset_name, attack_type, idx_test, y_pred_orig, labels)
target_node_list += target_node_list1
target_node_list.sort()
print(f"Test nodes number: {len(target_node_list)}, incorrect: {len(target_node_list1)}")

######################### Load CF examples  #########################
header = ['success','target_node', 'new_idx', 'added_edges', 'removed_edges', 'explanation_size', 'original_pred',
          'new_pred', 'cf_adj']

# counterfactual explanation subgraph path
time_name = '2025-09-16'
counterfactual_explanation_subgraph_path = base_path + f'/results/{time_name}/attack_subgraph/{attack_type}_{attack_method}_{dataset_name}_budget{attack_budget_list}'

with open(
        counterfactual_explanation_subgraph_path + f"/{DATA_NAME}_cf_examples_gcnlayer{GCN_LAYER}_lr{LEARNING_RATE}_seed{SEED_NUM}",
        "rb") as f:
    cf_examples = pickle.load(f)
    df_prep = []
    time_list = []
    for example in cf_examples:
        time_list.append(example["time_cost"])
        if example["data"]:
            df_prep.append(example["data"])
    df = pd.DataFrame(df_prep, columns=df_prep[0].keys())

######################### Metrics Evaluation  #########################
misclas_num = 0
fidelity = 0.0
added_edges_num = 0.0
deleted_edges_num = 0.0
edited_num = 0.0
S_plau = 0.0
orig_sub_adj = torch.tensor(data.adj.toarray())
sub_feat = pyg_data.x
for i in tqdm(df.index):
    edited_sub_adj = torch.tensor(df["cf_adj"][i].toarray())
    edited_norm_adj = normalize_adj(edited_sub_adj)
    new_label = gnn_model.forward(sub_feat, edited_norm_adj)

    # misclassification
    # if df["success"][i]:
    #     misclas_num += 1
    a1 = y_pred_orig[df["target_node"][i]].argmax()
    a2 = new_label[df["new_idx"][i]].argmax()
    if a1.item() != a2.item():
        misclas_num += 1

    # fidelity
    prob_pred_orig = torch.exp(y_pred_orig[df["target_node"][i]])
    label_pred_orig = y_pred_orig[df["target_node"][i]].argmax().item()
    prob_new_actual = torch.exp(new_label[df["new_idx"][i]])
    fidelity += prob_pred_orig[label_pred_orig] - prob_new_actual[label_pred_orig]

    # explanation size
    if df["success"][i]:
        added_edges_num += len(df["added_edges"][i])
        deleted_edges_num += len(df["removed_edges"][i])
        edited_num += df["explanation_size"][i]

    # plausibility
    if df["success"][i]:
        L_plau = α2 * compute_deg_diff(orig_sub_adj,
                                       edited_sub_adj) + α3 * compute_motif_viol(orig_sub_adj,
                                                                                                   edited_sub_adj,
                                                                                                   tau_c)
        S_plau += 2 * (1 - 1 / (1 + torch.exp(-1 * k * L_plau)))
        # print(S_plau)

print("Num of target nodes: ", len(target_node_list))
print("Num of misclassification: ", misclas_num)
print("Num of cf examples found: {}/{}".format(misclas_num, len(df)))
print("Metric 1 - Misclassification Rate: {:.2f}".format(misclas_num / len(target_node_list)))
print("Metric 2 - Fidelity: {:.4f}".format(fidelity / len(target_node_list)))
print("Metric 3 - Average Explanation Size: {:.2f}, E+: {:.2f}, E-: {:.2f}".format(edited_num / misclas_num,
                                                                                   added_edges_num / misclas_num,
                                                                                   deleted_edges_num / misclas_num))
print("Metric 4 - Average Plausibility: {:.2f}".format(S_plau / misclas_num))
print("Metric 5 - Average Time Cost: {:.2f}s/per".format(np.mean(np.array(time_list))))
