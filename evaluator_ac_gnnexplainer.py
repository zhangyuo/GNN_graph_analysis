#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/9/4 17:17
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : evaluator_ac_gnnexplainer.py
# @Software : PyCharm
# @Desc     :
"""
from __future__ import division
from __future__ import print_function
import sys

from config.config import ATTACK_TYPE, ATTACK_METHOD, EXPLAINER_METHOD, EXPLANATION_TYPE, DATA_NAME, ATTACK_BUDGET_LIST, \
    TEST_MODEL, GCN_LAYER, HIDDEN_CHANNELS, DROPOUT, LEARNING_RATE, WEIGHT_DECAY, WITH_BIAS, DEVICE, SEED_NUM, α2, α3, \
    TAU_C, LEARNING_RATE_AC, NUM_EPOCHS, NUM_EPOCHS_AC
from model.GCN import load_GCN_model
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
attack_type = ATTACK_TYPE
attack_method = ATTACK_METHOD
explainer_method = "ACExplainer"
explanation_type = EXPLANATION_TYPE
dataset_name = DATA_NAME
attack_budget_list = ATTACK_BUDGET_LIST
test_model = TEST_MODEL
gcn_layer = GCN_LAYER
nhid = HIDDEN_CHANNELS
dropout = DROPOUT
lr = LEARNING_RATE
weight_decay = WEIGHT_DECAY
with_bias = WITH_BIAS
device = DEVICE
tau_c = TAU_C

np.random.seed(SEED_NUM)

######################### loading deeprobust dataset  #########################
data = None
# dataset path
dataset_path = base_path + '/dataset'
# adjacency matrix is a high compressed sparse row format
if dataset_name == 'cora':
    data = Dataset(root=dataset_path, name=dataset_name)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
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

######################### Load CF examples  #########################
header = ['target_node', 'new_idx', 'added_edges', 'removed_edges', 'explanation_size', 'plau_loss', 'original_pred',
          'new_pred', 'extended_nodes', 'extended_adj', 'cf_adj', 'extended_feat', 'subgraph', 'true_subgraph',
          'E_type', "sub_labels"]

# counterfactual explanation subgraph path
time_name = '2025-09-11'
counterfactual_explanation_subgraph_path = base_path + f'/results/{time_name}/counterfactual_subgraph/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{attack_budget_list}'

with open(
        counterfactual_explanation_subgraph_path + f"/{DATA_NAME}_cf_examples_gcnlayer{GCN_LAYER}_lr{LEARNING_RATE}_epochs{NUM_EPOCHS_AC}_seed{SEED_NUM}",
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
num_edges_adj = (sum(sum(dense_adj)) / 2).item()
L_plau = 0.0
ps_num = 0
motif_accuracy = 0.0
for i in df.index:
    # plausibility
    orig_sub_adj = torch.tensor(df["extended_adj"][i])
    edited_sub_adj = torch.tensor(df["cf_adj"][i])
    # print(df["plau_loss"][i])
    # print(L_plau)
    try:
        L_plau += df["plau_loss"][i]
    except:
        L_plau += df["plau_loss"][i].item()

    # accuracy using F_NS or motif
    edge_in_motif_num = 0
    if dataset_name in ["BA-SHAPES", "TREE-CYCLES"]:
        perturbed_edges = df["extended_adj"][i] - df["cf_adj"][i]
        nonzero_indices = np.nonzero(perturbed_edges)
        perturbed_edge_list = list(zip(nonzero_indices[0], nonzero_indices[1]))
        perturbed_edge_list = [(u, v) for u, v in perturbed_edge_list if u < v]
        for u, v in perturbed_edge_list:
            if df['sub_labels'][i][u] != 0 and df['sub_labels'][i][v] != 0:
                edge_in_motif_num += 1
        motif_accuracy += edge_in_motif_num / len(perturbed_edge_list)
    elif dataset_name == "Loan-Decision":
        perturbed_edges = df["extended_adj"][i] - df["cf_adj"][i]
        nonzero_indices = np.nonzero(perturbed_edges)
        perturbed_edge_list = list(zip(nonzero_indices[0], nonzero_indices[1]))
        perturbed_edge_list = [(u, v) for u, v in perturbed_edge_list if u < v]
        for u, v in perturbed_edge_list:
            if u.item() == df['new_idx'][i] or v.item() == df['new_idx'][i]:
                edge_in_motif_num += 1
        motif_accuracy += (edge_in_motif_num - len(df['added_edges'][i])) / len(perturbed_edge_list)
    else:
        edited_norm_adj = normalize_adj(edited_sub_adj)
        sub_feat = df["extended_feat"][i]
        ps_label = gnn_model.forward(sub_feat, edited_norm_adj)
        label_pred_orig = y_pred_orig[df["target_node"][i]].argmax()
        ps_label_pred_new_actual = ps_label[df["new_idx"][i]].argmax()
        if label_pred_orig == ps_label_pred_new_actual:
            ps_num += 1

# plausibility
L_plau = L_plau.item() / len(df)

# accuracy using F_NS or motif
if dataset_name in ["BA-SHAPES", "TREE-CYCLES", "Loan-Decision"]:
    F_NS = motif_accuracy / len(df)
else:
    ps = ps_num / len(target_node_list)
    pn = len(df) / len(target_node_list)
    F_NS = 2 * ps * pn / (ps + pn)

print("Num cf examples found: {}/{}".format(len(df), len(target_node_list)))
print("Metric 1 - Fidelity+: {}".format(1 - len(df) / len(target_node_list)))
print("Metric 2 - Average Explanation Size: {}, std: {}".format(np.mean(df["explanation_size"]),
                                                                np.std(df["explanation_size"])))
print("Metric 3 - Average Sparsity: {}, std: {}".format(np.mean(1 - df["explanation_size"] / num_edges_adj),
                                                        np.std(1 - df["explanation_size"] / num_edges_adj)))
print("Metric 4 - Average Plausibility: {}".format(2 - 2 / (1 + np.exp(-1 * 0.05 * L_plau))))
print("Metric 5 - Average Accuracy: {}".format(F_NS))
print("Metric 6 - Average Time Cost: {:.4f}s/per".format(np.mean(np.array(time_list))))

# # Add num edges
# num_edges = []
# for i in df.index:
#     num_edges.append(sum(sum(df["sub_adj"][i])) / 2)
# df["num_edges"] = num_edges
#
# # For accuracy, only look at motif nodes
# df_motif = df[df["y_pred_orig"] != 0].reset_index(drop=True)
# accuracy = []
# # Get original predictions
# dict_ypred_orig = dict(zip(sorted(np.concatenate((idx_train.numpy(), idx_test.numpy()))), y_pred_orig.numpy()))
# for i in range(len(df_motif)):
#     node_idx = df_motif["node_idx"][i]
#     new_idx = df_motif["new_idx"][i]
#     _, _, _, node_dict = get_neighbourhood(int(node_idx), edge_index, 4, features, labels)
#
#     # Confirm idx mapping is correct
#     if node_dict[node_idx] == df_motif["new_idx"][i]:
#
#         cf_adj = df_motif["cf_adj"][i]
#         sub_adj = df_motif["sub_adj"][i]
#         perturb = np.abs(cf_adj - sub_adj)
#         perturb_edges = np.nonzero(perturb)  # Edge indices
#
#         nodes_involved = np.unique(np.concatenate((perturb_edges[0], perturb_edges[1]), axis=0))
#         perturb_nodes = nodes_involved[nodes_involved != new_idx]  # Remove original node
#
#         # Retrieve original node idxs for original predictions
#         perturb_nodes_orig_idx = []
#         for j in perturb_nodes:
#             perturb_nodes_orig_idx.append([key for (key, value) in node_dict.items() if value == j])
#         perturb_nodes_orig_idx = np.array(perturb_nodes_orig_idx).flatten()
#
#         # Retrieve original predictions
#         perturb_nodes_orig_ypred = np.array([dict_ypred_orig[k] for k in perturb_nodes_orig_idx])
#         nodes_in_motif = perturb_nodes_orig_ypred[perturb_nodes_orig_ypred != 0]
#         prop_correct = len(nodes_in_motif) / len(perturb_nodes_orig_idx)
#
#         accuracy.append([node_idx, new_idx, perturb_nodes_orig_idx,
#                          perturb_nodes_orig_ypred, nodes_in_motif, prop_correct])
#
# df_accuracy = pd.DataFrame(accuracy, columns=["node_idx", "new_idx", "perturb_nodes_orig_idx",
#                                               "perturb_nodes_orig_ypred", "nodes_in_motif", "prop_correct"])
# print("Accuracy", np.mean(df_accuracy["prop_correct"]), np.std(df_accuracy["prop_correct"]))
