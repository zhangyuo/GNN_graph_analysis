#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/9/9 21:29
# @Author   : **
# @Email    : **@**
# @File     : gnnexplainer_cf_subgraph.py
# @Software : PyCharm
# @Desc     :
"""
import os
import pickle
import sys
import warnings
res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(
    os.path.dirname(os.path.dirname(res)))  # acquire the parent path of current file's parent path
sys.path.insert(0, base_path)
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import dense_to_sparse, to_undirected

from model.GAT import load_GATNet_model
from model.GraphConv import load_GraphConv_model
from model.GraphTransformer import load_GraphTransforer_model

warnings.filterwarnings("ignore")
import time
from datetime import datetime

import torch
import numpy as np
from deeprobust.graph.data import Dataset
from tqdm import tqdm
from config.config import *
from model.GCN import GCN_model, dr_data_to_pyg_data, GCNtoPYG, load_GCN_model
from utilty.utils import normalize_adj, select_test_nodes, CPU_Unpickler, BAShapesDataset, TreeCyclesDataset, \
    LoanDecisionDataset, OGBNArxivDataset
from instance_level_explanation_subgraph.GNNExplainer_subgraph.generate_gnnexplainer_subgraph import \
    generate_gnnexplainer_cf_subgraph
from subgraph_quantify.graph_analysis import gnn_explainer_generate
import torch.nn.functional as F
from counterfactual_explanation_subgraph.ACExplainer_subgraph.acexplainer_subgraph import evaluate_test_data


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
    explanation_type = EXPLANATION_TYPE
    attack_method = ATTACK_METHOD
    attack_budget_list = ATTACK_BUDGET_LIST
    explainer_method = "GNNExplainer"
    heads_num = HEADS_NUM if TEST_MODEL in ["GraphTransformer", "GAT"] else None
    tau_c = TAU_C

    np.random.seed(SEED_NUM)
    torch.manual_seed(SEED_NUM)

    # attack_budget_list = [5]
    budget = attack_budget_list[0]
    time_name = datetime.now().strftime("%Y-%m-%d")
    # counterfactual explanation subgraph path
    counterfactual_explanation_subgraph_path = base_path + f'/results/{time_name}/counterfactual_subgraph_{test_model}/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{attack_budget_list}-{SEED_NUM}'
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
            pre_output = gnn_model.forward(torch.tensor(features.toarray()), norm_adj)
        else:
            output_path = os.path.join(model_save_path, 'pre_output.pikle')
            if os.path.exists(output_path):
                with open(output_path, "rb") as fr:
                    result = pickle.load(fr)
                    pre_output, target_node_id = result["pre_output"], result["target_node_id"]
            else:
                pre_output, target_node_id = evaluate_test_data(gnn_model, data, pyg_data, gcn_layer)
                result = {"pre_output": pre_output, "target_node_id": target_node_id}
                with open(output_path, "wb") as fw:
                    pickle.dump(result, fw)
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
    elif test_model == 'GAT':
        file_path = os.path.join(model_save_path, 'gat_model.pth')
        gnn_model = load_GATNet_model(file_path, data, nhid, dropout, device, lr, weight_decay, gcn_layer, heads_num)
        dense_adj = torch.tensor(adj.toarray())
        norm_adj = normalize_adj(dense_adj)
        edge_index, edge_weight = dense_to_sparse(norm_adj)
        pre_output = gnn_model.forward(torch.tensor(features.toarray()), edge_index, edge_weight=edge_weight)

    ######################### select test nodes  #########################
    if dataset_name == "ogbn-arxiv":
        idx_test = target_node_id
    target_node_list, target_node_list1 = select_test_nodes(dataset_name, attack_type, idx_test, pre_output, labels)
    target_node_list = target_node_list + target_node_list1
    target_node_list.sort()
    print(f"Test nodes number: {len(target_node_list)}, incorrect: {len(target_node_list1)}")
    # target_node_list = target_node_list[101:110]

    ######################### GNN explainer generate  #########################
    explainer = gnn_explainer_generate(test_model, gnn_model, device, features, labels, gcn_layer,epoch=10)

    ######################### GNN explainer generate  #########################
    # Get CF examples in test set
    start_0 = time.time()
    test_cf_examples = []
    # cfexp_subgraph = {}
    time_list = []
    mis_cases = 0
    for target_node in tqdm(target_node_list):
        cf_example, time_cost = generate_gnnexplainer_cf_subgraph(test_model, target_node, gcn_layer, pyg_data, explainer,
                                                                  gnn_model, pre_output, dataset_name, budget=budget, output_idx=idx_test)
        # print(cf_example)
        print("Time for one example: {:.4f}s".format(time_cost))
        time_list.append(time_cost)
        # cfexp_subgraph[target_node] = cf_example["subgraph"] if cf_example else None
        test_cf_examples.append({"data": cf_example, "time_cost": time_cost})
        if cf_example['success']:
            mis_cases += 1
    print("Total time elapsed: {:.4f}min".format((time.time() - start_0) / 60))
    print("Number of CF examples found: {}/{}".format(mis_cases, len(target_node_list)))

    # with open(counterfactual_explanation_subgraph_path + "/cfexp_subgraph.pickle", "wb") as fw:
    #     pickle.dump(cfexp_subgraph, fw)

    # Save CF examples in test set
    with open(
            counterfactual_explanation_subgraph_path + f"/{DATA_NAME}_cf_examples_gcnlayer{GCN_LAYER}_lr{LEARNING_RATE}_seed{SEED_NUM}",
            "wb") as f:
        pickle.dump(test_cf_examples, f)
