#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/9/4 17:17
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : cf_evaluator.py
# @Software : PyCharm
# @Desc     :
"""
from __future__ import division
from __future__ import print_function
import sys

from config.config import ATTACK_TYPE, ATTACK_METHOD, EXPLAINER_METHOD, EXPLANATION_TYPE, DATA_NAME, ATTACK_BUDGET_LIST, \
    TEST_MODEL, GCN_LAYER, HIDDEN_CHANNELS, DROPOUT, LEARNING_RATE, WEIGHT_DECAY, WITH_BIAS, DEVICE, SEED_NUM, α2, α3, \
    TAU_C
from model.GCN import load_GCN_model
from utilty.utils import normalize_adj, select_test_nodes, compute_deg_diff, compute_motif_viol
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
explainer_method = EXPLAINER_METHOD
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
target_node_list, target_node_list1 = select_test_nodes(attack_type, explanation_type, idx_test, y_pred_orig, labels)
target_node_list += target_node_list1

######################### Load CF examples  #########################
header = ["node_idx", "new_idx", "cf_adj", "sub_adj", "y_pred_orig", "y_pred_new", "y_pred_new_actual",
          "label", "num_nodes", "loss_total", "loss_pred", "loss_graph_dist", "sub_feat"]

# counterfactual explanation subgraph path
time_name = '2025-09-04-CFGNNExplainer_all_explanation'
counterfactual_explanation_subgraph_path = base_path + f'/results/{time_name}/counterfactual_subgraph/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{attack_budget_list}'

with open(
        counterfactual_explanation_subgraph_path + "/cora_cf_examples_gcnlayer2_lr0.01_beta0.5_mom0.9_epochs500_seed102",
        "rb") as f:
    cf_examples = pickle.load(f)
    df_prep = []
    time_list = []
    for example in cf_examples:
        time_list.append(example["time_cost"])
        if example["data"] != []:
            df_prep.append(example["data"][0])
    df = pd.DataFrame(df_prep, columns=header)

######################### Metrics Evaluation  #########################
num_edges_adj = (sum(sum(dense_adj)) / 2).item()
L_plau = 0.0
ps_num = 0
for i in df.index:
    # plausibility
    orig_sub_adj = torch.tensor(df["sub_adj"][i])
    edited_sub_adj = torch.tensor(df["cf_adj"][i])
    L_plau += α2 * compute_deg_diff(orig_sub_adj, edited_sub_adj) + α3 * compute_motif_viol(orig_sub_adj,
                                                                                            edited_sub_adj, tau_c)
    # accuracy using F_NS
    edited_norm_adj = normalize_adj(edited_sub_adj)
    sub_feat = df["sub_feat"][i]
    ps_label = gnn_model.forward(sub_feat, edited_norm_adj)
    label_pred_orig = y_pred_orig[df["node_idx"][i]].argmax()
    ps_label_pred_new_actual = ps_label[df["new_idx"][i]].argmax()
    if label_pred_orig == ps_label_pred_new_actual:
        ps_num += 1
L_plau = L_plau / len(df)
ps = ps_num / len(target_node_list)
pn = len(df) / len(target_node_list)
F_NS = 2 * ps * pn / (ps + pn)

print("Num cf examples found: {}/{}".format(len(df), len(target_node_list)))
print("Metric 1 - Fidelity+: {}".format(1 - len(df) / len(target_node_list)))
print("Metric 2 - Average Explanation Size: {}, std: {}".format(np.mean(df["loss_graph_dist"]),
                                                                np.std(df["loss_graph_dist"])))
print("Metric 3 - Average Sparsity: {}, std: {}".format(np.mean(1 - df["loss_graph_dist"] / num_edges_adj),
                                                        np.std(1 - df["loss_graph_dist"] / num_edges_adj)))
print("Metric 4 - Average Plausibility: {}".format(1-1 / (1 + np.exp(-1 * 0.05 * L_plau))))
print("Metric 5 - Average Accuracy: {}".format(F_NS))
print("Metric 6 - Average Time Cost: {:.4f}s".format(np.mean(np.array(time_list))))


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