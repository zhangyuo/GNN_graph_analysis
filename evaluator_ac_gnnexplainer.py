#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/9/4 17:17
# @Author   : **
# @Email    : **@**
# @File     : evaluator_ac_gnnexplainer.py
# @Software : PyCharm
# @Desc     :
"""
from __future__ import division
from __future__ import print_function
import sys
import os
import warnings

warnings.filterwarnings("ignore")
res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(res)
sys.path.insert(0, base_path)
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import dense_to_sparse, to_undirected
from config.config import *
from model.GAT import load_GATNet_model
from model.GCN import load_GCN_model, dr_data_to_pyg_data
from model.GraphConv import load_GraphConv_model
from model.GraphTransformer import load_GraphTransforer_model
from utilty.utils import normalize_adj, select_test_nodes, compute_deg_diff, compute_motif_viol, CPU_Unpickler, \
    BAShapesDataset, TreeCyclesDataset, LoanDecisionDataset, compute_feat_sim, OGBNArxivDataset
import numpy as np
import pandas as pd
import pickle
import torch
from counterfactual_explanation_subgraph.ACExplainer_subgraph.acexplainer_subgraph import evaluate_test_data
from deeprobust.graph.data import Dataset
import torch.nn.functional as F

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
heads_num = HEADS_NUM if TEST_MODEL in ["GraphTransformer", "GAT"] else None

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
        if test_model == "GAT":
            # because of no features of nodes
            pyg_data.x = F.one_hot(pyg_data.y).float()
    data = BAShapesDataset(pyg_data)
    # Create deeprobust Data object
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
elif dataset_name == 'TREE-CYCLES':
    # Create PyG Data object
    with open(dataset_path + "/TreeCycle.pickle", "rb") as f:
        pyg_data = CPU_Unpickler(f).load()
        if test_model == "GAT":
            # because of no features of nodes
            pyg_data.x = F.one_hot(pyg_data.y).float()
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
    ogbn_arxiv_data = PygNodePropPredDataset(name="ogbn-arxiv", root=dataset_path)
    pyg_data = ogbn_arxiv_data[0]
    pyg_data.edge_index = to_undirected(pyg_data.edge_index)
    pyg_data.y = pyg_data.y.view(-1).long()
    # Create deeprobust Data object
    data = OGBNArxivDataset(ogbn_arxiv_data)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
else:
    adj, features, labels = None, None, None
    idx_train, idx_val, idx_test = None, None, None

######################### Loading GCN model  #########################
model_save_path = f'{base_path}/model_save/{test_model}/{dataset_name}/{gcn_layer}-layer/'
if test_model == 'GCN':
    file_path = os.path.join(model_save_path, 'gcn_model.pth')
    gnn_model = load_GCN_model(file_path, features, labels, nhid, dropout, device, lr, weight_decay,
                               with_bias, gcn_layer)
    if dataset_name != "ogbn-arxiv":
        dense_adj = torch.tensor(adj.toarray())
        norm_adj = normalize_adj(dense_adj)
        y_pred_orig = gnn_model.forward(torch.tensor(features.toarray()), norm_adj)
    else:
        output_path = os.path.join(model_save_path, 'pre_output.pikle')
        if os.path.exists(output_path):
            with open(output_path, "rb") as fr:
                result = pickle.load(fr)
                y_pred_orig, target_node_id = result["pre_output"], result["target_node_id"]
        else:
            y_pred_orig, target_node_id = evaluate_test_data(gnn_model, data, pyg_data, gcn_layer)
            result = {"pre_output": y_pred_orig, "target_node_id": target_node_id}
            with open(output_path, "wb") as fw:
                pickle.dump(result, fw)
elif test_model == 'GraphTransformer':
    file_path = os.path.join(model_save_path, 'graphTransformer_model.pth')
    gnn_model = load_GraphTransforer_model(file_path, data, nhid, dropout, device, lr, weight_decay, gcn_layer,
                                           heads_num)
    dense_adj = torch.tensor(adj.toarray())
    norm_adj = normalize_adj(dense_adj)
    edge_index, edge_weight = dense_to_sparse(norm_adj)
    y_pred_orig = gnn_model.forward(torch.tensor(features.toarray()), edge_index, edge_weight=edge_weight)
elif test_model == 'GraphConv':
    file_path = os.path.join(model_save_path, 'graphConv_model.pth')
    gnn_model = load_GraphConv_model(file_path, data, nhid, dropout, device, lr, weight_decay, gcn_layer)
    dense_adj = torch.tensor(adj.toarray())
    norm_adj = normalize_adj(dense_adj)
    edge_index, edge_weight = dense_to_sparse(norm_adj)
    y_pred_orig = gnn_model.forward(torch.tensor(features.toarray()), edge_index, edge_weight=edge_weight)
elif test_model == 'GAT':
    file_path = os.path.join(model_save_path, 'gat_model.pth')
    gnn_model = load_GATNet_model(file_path, data, nhid, dropout, device, lr, weight_decay, gcn_layer, heads_num)
    dense_adj = torch.tensor(adj.toarray())
    norm_adj = normalize_adj(dense_adj)
    edge_index, edge_weight = dense_to_sparse(norm_adj)
    y_pred_orig = gnn_model.forward(torch.tensor(features.toarray()), edge_index, edge_weight=edge_weight)

######################### select test nodes  #########################
if dataset_name == "ogbn-arxiv":
    idx_test = target_node_id
target_node_list, target_node_list1 = select_test_nodes(dataset_name, attack_type, idx_test, y_pred_orig, labels)
target_node_list += target_node_list1
target_node_list.sort()
print(f"Test nodes number: {len(target_node_list)}, incorrect: {len(target_node_list1)}")
# target_node_list = target_node_list[101:110]

######################### Load CF examples  #########################
header = ['success', 'target_node', 'new_idx', 'added_edges', 'removed_edges', 'explanation_size', 'plau_loss',
          'original_pred', 'new_pred', 'extended_adj', 'cf_adj', 'extended_feat', 'sub_labels', 'new_idx_map_tgt_node']

# counterfactual explanation subgraph path
time_name = '2025-09-25'
counterfactual_explanation_subgraph_path = base_path + f'/results/{time_name}/counterfactual_subgraph_{test_model}/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{[MAX_EDITS]}-{SEED_NUM}'

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
misclas_num = 0
fidelity = 0.0
added_edges_num = 0.0
deleted_edges_num = 0.0
edited_num = 0.0
S_plau = 0.0
for i in df.index:
    orig_sub_adj = torch.tensor(df["extended_adj"][i])
    edited_sub_adj = torch.tensor(df["cf_adj"][i])
    sub_feat = df["extended_feat"][i]
    edited_norm_adj = normalize_adj(edited_sub_adj)
    if test_model == "GCN":
        new_label = gnn_model.forward(sub_feat, edited_norm_adj)
    else:
        edge_index, edge_weight = dense_to_sparse(edited_norm_adj)
        new_label = gnn_model.forward(sub_feat, edge_index, edge_weight=edge_weight)

    # misclassification

    #     if df["success"][i]:
    #         misclas_num += 1
    if dataset_name == "ogbn-arxiv":
        tgt_node_map_new_idx = df["target_node"][i]  # target node idx in new constructed subgraph
        target_node = df["new_idx_map_tgt_node"][i][tgt_node_map_new_idx]
        output_target_idx = idx_test.index(target_node)
    else:
        output_target_idx = df["target_node"][i]

    a1 = y_pred_orig[output_target_idx].argmax()
    a2 = new_label[df["new_idx"][i]].argmax()
    if a1.item() != a2.item():
        misclas_num += 1
        # print(df["target_node"][i], df["new_idx"][i])

    # fidelity
    prob_pred_orig = torch.exp(y_pred_orig[output_target_idx])
    label_pred_orig = y_pred_orig[output_target_idx].argmax().item()
    prob_new_actual = torch.exp(new_label[df["new_idx"][i]])
    fidelity += prob_pred_orig[label_pred_orig] - prob_new_actual[label_pred_orig]

    # explanation size
    if df["success"][i]:
        added_edges_num += len(df["added_edges"][i])
        deleted_edges_num += len(df["removed_edges"][i])
        edited_num += df["explanation_size"][i]

    # plausibility
    tt = 0.0
    if df["success"][i]:
        # features = pyg_data.x
        # for u,v in df["added_edges"][i]:
        #     tt += compute_feat_sim(features[u], features[v])
        # for u,v in df["removed_edges"][i]:
        #     tt += compute_feat_sim(features[u], features[v])
        # tt = tt / df["explanation_size"][i]
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
