#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/7/17 12:43
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : cfexplainer_subgraph.py
# @Software : PyCharm
# @Desc     :
"""
import os
import pickle
import sys
import time
from datetime import datetime

import torch
import numpy as np
from deeprobust.graph.data import Dataset
from torch_geometric.utils import k_hop_subgraph, subgraph, to_dense_adj, dense_to_sparse
from tqdm import tqdm

from model.GCN import GCN_model, dr_data_to_pyg_data, GCNtoPYG, load_GCN_model
from config.config import *
from explainer.cf_explanation.cf_explainer import CFExplainer
from model.GraphConv import load_GraphConv_model
from model.GraphTransformer import load_GraphTransforer_model
from utilty.cfexplanation_visualization import visualize_cfexp_subgraph
from utilty.utils import safe_open, get_neighbourhood, normalize_adj, select_test_nodes, CPU_Unpickler, BAShapesDataset, \
    TreeCyclesDataset, LoanDecisionDataset


def attack_cfexplanation_subgraph_generate(target_node_list, attack_subgraph, features, labels, gnn_model,
                                           device, idx_test, gcn_layer, with_bias,
                                           counterfactual_explanation_subgraph_path):
    cfexp_subgraph = {}
    for target_node in tqdm(target_node_list):
        at_sbg_dt = attack_subgraph[target_node]
        modified_labels = at_sbg_dt['modified_labels']
        modified_adj = at_sbg_dt['modified_adj']
        modified_features = at_sbg_dt['modified_features'] if at_sbg_dt['modified_features'] else features
        attacked_pyg_data = dr_data_to_pyg_data(modified_adj, modified_features, modified_labels)
        edge_index = attacked_pyg_data.edge_index
        norm_modified_adj = normalize_adj(modified_adj)
        pre_output = gnn_model.forward(modified_features, norm_modified_adj)[target_node]
        subgraph, _ = generate_cfexplainer_subgraph(target_node, edge_index, modified_adj, modified_features, labels,
                                                    pre_output, gnn_model,
                                                    device, idx_test, gcn_layer, with_bias,
                                                    counterfactual_explanation_subgraph_path)
        cfexp_subgraph[target_node] = subgraph

    return cfexp_subgraph


def generate_cfexplainer_subgraph(target_node, edge_index, adj, features, labels, output, model, device, idx_test,
                                  gcn_layer, with_bias, counterfactual_explanation_subgraph_path, test_model,
                                  heads_num):
    start = time.time()
    sub_adj, sub_edge_index, sub_feat, sub_labels, node_dict = get_neighbourhood(target_node, edge_index,
                                                                                 features, labels, gcn_layer)
    new_idx = node_dict[target_node]
    # sub_pyg_data = dr_data_to_pyg_data(sub_adj, sub_feat, sub_labels)
    print("Output original model, full adj: {}".format(output[target_node]))
    norm_sub_adj = normalize_adj(sub_adj)
    if test_model == "GCN":
        print("Output original model, sub adj: {}".format(model.forward(sub_feat, norm_sub_adj)[new_idx]))
    elif test_model in ["GraphTransformer", "GraphConv"]:
        edge_index, edge_weight = dense_to_sparse(norm_sub_adj)
        print("Output original model, sub adj: {}".format(model.forward(sub_feat, edge_index, edge_weight=edge_weight)[new_idx]))
    # output = gnn_model.predict(features=features, adj=modified_adj)
    # Need to instantitate new cf model every time because size of P changes based on size of sub_adj
    explainer = CFExplainer(model=model,
                            sub_adj=sub_adj,
                            sub_feat=sub_feat,
                            n_hid=HIDDEN_CHANNELS,
                            dropout=DROPOUT,
                            sub_labels=sub_labels,
                            y_pred_orig=output.argmax(dim=1)[target_node],
                            num_classes=labels.max().item() + 1,
                            beta=BETA,
                            device=device,
                            gcn_layer=gcn_layer,
                            with_bias=with_bias,
                            test_model=test_model,
                            heads=heads_num)
    if device == 'cuda':
        model.cuda()
        explainer.cf_model.cuda()
        adj = adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_test = idx_test.cuda()

    cf_example = explainer.explain(node_idx=target_node, cf_optimizer=OPTIMIZER, new_idx=new_idx, lr=LEARNING_RATE,
                                   n_momentum=N_Momentum, num_epochs=NUM_EPOCHS)
    time_cost = time.time() - start

    # graph visualization
    subgraph = {
        "subgraph": None,
        "true_subgraph": None,
        "E_type": None,
    }
    if cf_example[-1]:
        modified_sub_adj = cf_example[2]
        changed_label = cf_example[5]
        subgraph, true_subgraph, E_type = visualize_cfexp_subgraph(
            modified_sub_adj,
            sub_adj.detach().numpy(),
            labels,
            sub_labels,
            sub_feat.numpy(),
            changed_label,
            new_idx,
            cfexp_name='CFExplanation',
            title="Visualization for counterfactual explanation subgraph",
            pic_path=counterfactual_explanation_subgraph_path,
            full_mapping=node_dict
        )
        print("Visualize ok for counterfactual explanation subgraph")
        subgraph = {
            "subgraph": subgraph,
            "true_subgraph": true_subgraph,
            "E_type": E_type,
        }
    return subgraph, cf_example, time_cost


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
    explainer_method = "CFExplainer"
    heads_num = HEADS_NUM if TEST_MODEL in ["GraphTransformer"] else None

    np.random.seed(SEED_NUM)
    torch.manual_seed(SEED_NUM)

    time_name = datetime.now().strftime("%Y-%m-%d")
    # counterfactual explanation subgraph path
    counterfactual_explanation_subgraph_path = base_path + f'/results/{time_name}/counterfactual_subgraph_{test_model}/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{attack_budget_list}'
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

    if test_model == 'GCN':
        file_path = os.path.join(model_save_path, 'gcn_model.pth')
        gnn_model = load_GCN_model(file_path, features, labels, nhid, dropout, device, lr, weight_decay,
                                   with_bias, gcn_layer)
        dense_adj = torch.tensor(adj.toarray())
        norm_adj = normalize_adj(dense_adj)
        pre_output = gnn_model.forward(torch.tensor(features.toarray()), norm_adj)
    elif test_model == 'GraphTransformer':
        file_path = os.path.join(model_save_path, 'graphTransformer_model.pth')
        gnn_model = load_GraphTransforer_model(file_path, data, nhid, dropout, device, lr, weight_decay, gcn_layer,
                                               heads_num)
        dense_adj = torch.tensor(adj.toarray())
        norm_adj = normalize_adj(dense_adj)
        edge_index, edge_weight = dense_to_sparse(norm_adj)
        pre_output = gnn_model.forward(torch.tensor(features.toarray()), edge_index, edge_weight=edge_weight)
    elif test_model == 'GraphConv':
        file_path = os.path.join(model_save_path, 'graphConv_model.pth')
        gnn_model = load_GraphConv_model(file_path, data, nhid, dropout, device, lr, weight_decay, gcn_layer)
        dense_adj = torch.tensor(adj.toarray())
        norm_adj = normalize_adj(dense_adj)
        edge_index, edge_weight = dense_to_sparse(norm_adj)
        pre_output = gnn_model.forward(torch.tensor(features.toarray()), edge_index, edge_weight=edge_weight)

    ######################### select test nodes  #########################
    target_node_list, target_node_list1 = select_test_nodes(dataset_name, attack_type, idx_test, pre_output, labels)
    target_node_list = target_node_list + target_node_list1
    target_node_list.sort()
    print(f"Test nodes number: {len(target_node_list)}, incorrect: {len(target_node_list1)}")
    # target_node_list = target_node_list[101:110]

    ######################### GNN explainer generate  #########################
    # Get CF examples in test set
    start_0 = time.time()
    test_cf_examples = []
    cfexp_subgraph = {}
    time_list = []
    mis_cases = 0
    for target_node in tqdm(target_node_list):
        edge_index = pyg_data.edge_index
        subgraph, cf_example, time_cost = generate_cfexplainer_subgraph(target_node, edge_index, adj, features, labels,
                                                                        pre_output,
                                                                        gnn_model, device, idx_test, gcn_layer,
                                                                        with_bias,
                                                                        counterfactual_explanation_subgraph_path,
                                                                        test_model, heads_num)
        print("Time for {} epochs of one example: {:.4f}s".format(NUM_EPOCHS, time_cost))
        time_list.append(time_cost)
        cfexp_subgraph[target_node] = subgraph
        test_cf_examples.append({"data": cf_example, "time_cost": time_cost})
        if cf_example[-1]:
            mis_cases += 1
    print("Total time elapsed: {:.4f}min".format((time.time() - start_0) / 60))
    print("Number of CF examples found: {}/{}".format(mis_cases, len(target_node_list)))

    with open(counterfactual_explanation_subgraph_path + "/cfexp_subgraph.pickle", "wb") as fw:
        pickle.dump(cfexp_subgraph, fw)

    # Save CF examples in test set
    with open(
            counterfactual_explanation_subgraph_path + f"/{DATA_NAME}_cf_examples_gcnlayer{GCN_LAYER}_lr{LEARNING_RATE}_beta{BETA}_mom{N_Momentum}_epochs{NUM_EPOCHS}_seed{SEED_NUM}",
            "wb") as f:
        pickle.dump(test_cf_examples, f)
