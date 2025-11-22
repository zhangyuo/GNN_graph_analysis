#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/7/17 12:43
# @Author   : **
# @Email    : **@**
# @File     : c2explainer_subgraph.py
# @Software : PyCharm
# @Desc     :
"""
import os
import pickle
import sys

res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(
    os.path.dirname(os.path.dirname(res)))  # acquire the parent path of current file's parent path
sys.path.insert(0, base_path)
from explainer.c2_explainer.c2_explainer import C2Explainer
from ogb.nodeproppred import PygNodePropPredDataset
import time
from datetime import datetime

import torch
import numpy as np
from deeprobust.graph.data import Dataset
from torch_geometric.utils import k_hop_subgraph, subgraph, to_dense_adj, dense_to_sparse, to_undirected
from tqdm import tqdm
from counterfactual_explanation_subgraph.ACExplainer_subgraph.acexplainer_subgraph import evaluate_test_data
from model.DenseGAT import load_DenseGATNet_model
from model.GATNoWeight import load_GATNet_model
from model.GCN import GCN_model, dr_data_to_pyg_data, GCNtoPYG, load_GCN_model
from config.config import *
from explainer.cf_explanation.cf_explainer import CFExplainer
from model.GraphConv import load_GraphConv_model
from model.GraphTransformerNoWeight import load_GraphTransforer_model
from utilty.cfexplanation_visualization import visualize_cfexp_subgraph
from utilty.utils import safe_open, get_neighbourhood, normalize_adj, select_test_nodes, CPU_Unpickler, BAShapesDataset, \
    TreeCyclesDataset, LoanDecisionDataset, OGBNArxivDataset, ChameleonDataset
import torch.nn.functional as F
from torch_geometric.explain import Explainer


def analyze_edge_changes(original_edges, cf_edges, target_node):
    """
    分析边的具体变化：添加了哪些边，删除了哪些边
    """
    # 将边转换为集合以便比较（考虑无向图）
    original_set = set()
    for i in range(original_edges.shape[1]):
        u, v = original_edges[0, i].item(), original_edges[1, i].item()
        original_set.add((min(u, v), max(u, v)))

    cf_set = set()
    for i in range(cf_edges.shape[1]):
        u, v = cf_edges[0, i].item(), cf_edges[1, i].item()
        cf_set.add((min(u, v), max(u, v)))

    # 找出添加和删除的边
    added_edges = cf_set - original_set
    removed_edges = original_set - cf_set

    # 筛选与目标节点相关的边变化
    target_added = [edge for edge in added_edges if target_node in edge]
    target_removed = [edge for edge in removed_edges if target_node in edge]

    return {
        'added_edges': list(added_edges),
        'removed_edges': list(removed_edges),
        'total_added': len(added_edges),
        'total_removed': len(removed_edges),
        'target_added_edges': target_added,
        'target_removed_edges': target_removed,
        'target_related_added': len(target_added),
        'target_related_removed': len(target_removed)
    }


def generate_c2explainer_subgraph(target_node, explainer, pyg_data, adj, features, labels, output, model, pyg_gcn, device,
                                  idx_test,
                                  gcn_layer, with_bias, counterfactual_explanation_subgraph_path, test_model,
                                  dataset_name, heads_num, output_idx=None):
    start = time.time()
    sub_adj, sub_edge_index, sub_feat, sub_labels, node_dict = get_neighbourhood(target_node, pyg_data.edge_index,
                                                                                 features, labels, gcn_layer)
    new_idx = node_dict[target_node]
    # sub_pyg_data = dr_data_to_pyg_data(sub_adj, sub_feat, sub_labels)
    output_prob = None
    if dataset_name == 'ogbn-arxiv':
        print("Output original model, full adj: {}".format(output[output_idx.index(target_node)]))
        output_prob = output[output_idx.index(target_node)]
    else:
        print("Output original model, full adj: {}".format(output[target_node]))
        print("Output original model, full adj: label={}".format(output[target_node].argmax()))
        output_prob = output[target_node]
    norm_sub_adj = normalize_adj(sub_adj)
    if test_model == "GCN":
        print("Output original model, sub adj: {}".format(model.forward(sub_feat, norm_sub_adj)[new_idx]))
        print("Output original model, sub adj: label={}".format(model.forward(sub_feat, norm_sub_adj)[new_idx].argmax()))
    elif test_model in ["GraphTransformer", "GAT", "GraphConv"]:
        print("Output original model, sub adj: {}".format(model.forward(sub_feat, sub_edge_index)[new_idx]))

    if dataset_name == "ogbn-arxiv":
        y_pred_orig = output.argmax(dim=1)[output_idx.index(target_node)]
    else:
        y_pred_orig = output.argmax(dim=1)[target_node]

    if dataset_name == "ogbn-arxiv":
        explanation = explainer(pyg_data.x[list(node_dict.keys())], sub_edge_index, index=new_idx)
    else:
        if test_model == "GCN":
            explanation = explainer(pyg_data.x, pyg_data.edge_index, index=target_node)
        else:
            explanation = explainer(pyg_data.x, pyg_data.edge_index, index=target_node)

    time_cost = time.time() - start

    # graph visualization
    subgraph = {
        "subgraph": None,
        "true_subgraph": None,
        "E_type": None,
    }

    # Check if the explanation was successfully generated
    flag = False
    if hasattr(explanation, "perturbs") and int(explanation.perturbs / 2) <= MAX_EDITS:
        flag = True
    if flag:
        if dataset_name == "ogbn-arxiv":
            edge_changes = analyze_edge_changes(sub_edge_index, explanation.stores[0]['cf'], new_idx)
            modified_sub_adj = sub_adj.clone()
            for (u, v) in edge_changes["added_edges"]:
                i, j = u, v
                modified_sub_adj[i, j] = 1
                modified_sub_adj[j, i] = 1
            for (u, v) in edge_changes["removed_edges"]:
                i, j = u, v
                modified_sub_adj[i, j] = 0
                modified_sub_adj[j, i] = 0
        else:
            edge_changes = analyze_edge_changes(pyg_data.edge_index, explanation.stores[0]['cf'], target_node)
            modified_sub_adj = sub_adj.clone()
            for (u, v) in edge_changes["added_edges"]:
                i, j = node_dict[u], node_dict[v]
                modified_sub_adj[i, j] = 1
                modified_sub_adj[j, i] = 1
            for (u, v) in edge_changes["removed_edges"]:
                i, j = node_dict[u], node_dict[v]
                modified_sub_adj[i, j] = 0
                modified_sub_adj[j, i] = 0

        changed_label = explanation.stores[0]['label']
        # print("Output original model, full adj: label={}".format(output[target_node].argmax()))
        # norm_modified_sub_adj = normalize_adj(modified_sub_adj)
        # print("Output original model, sub adj: label={}".format(model.forward(sub_feat, norm_modified_sub_adj)[new_idx].argmax()))
        # pyg_gcn.forward(explanation.stores[0]['x'], explanation.stores[0]['edge_index'])

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
        cf_example = [target_node, new_idx, modified_sub_adj, sub_adj, y_pred_orig, explanation.stores[0]['probability'],
                      sub_labels, explanation.perturbs / 2, sub_feat, flag]
    else:
        cf_example = [target_node, new_idx, sub_adj, sub_adj, y_pred_orig, output_prob, sub_labels, 0, sub_feat, flag]
    return subgraph, cf_example, time_cost


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
    explainer_method = "C2Explainer"
    heads_num = HEADS_NUM if TEST_MODEL in ["GraphTransformer", "GAT"] else None

    np.random.seed(SEED_NUM)
    torch.manual_seed(SEED_NUM)

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
    elif dataset_name == 'chameleon':
        # Create PyG Data object
        from torch_geometric.datasets import WikipediaNetwork
        chameleon_data = WikipediaNetwork(name="chameleon", root=dataset_path)
        pyg_data = chameleon_data[0]
        pyg_data.y = pyg_data.y.view(-1).long()
        # Create deeprobust Data object
        data = ChameleonDataset(chameleon_data)
        pyg_data.edge_index = data.pyg_data.edge_index
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
        edge_index, _ = dense_to_sparse(dense_adj)
        pre_output = gnn_model.forward(torch.tensor(features.toarray()), edge_index)
    elif test_model == 'GraphConv':
        file_path = os.path.join(model_save_path, 'graphConv_model.pth')
        gnn_model = load_GraphConv_model(file_path, data, nhid, dropout, device, lr, weight_decay, gcn_layer)
        dense_adj = torch.tensor(adj.toarray())
        norm_adj = normalize_adj(dense_adj)
        edge_index, edge_weight = dense_to_sparse(norm_adj)
        pre_output = gnn_model.forward(torch.tensor(features.toarray()), edge_index, edge_weight=edge_weight)
    elif test_model == 'GAT':
        file_path = os.path.join(model_save_path, 'gat_model.pth')
        gnn_model = load_GATNet_model(file_path, data, nhid, dropout, device, lr, weight_decay, gcn_layer,
                                      heads_num)
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
    # target_node_list = [80]
    print(f"Test nodes number: {len(target_node_list)}, incorrect: {len(target_node_list1)}")
    # target_node_list = target_node_list[101:110]

    ######################### GNN explainer generate  #########################
    # Get CF examples in test set
    start_0 = time.time()
    test_cf_examples = []
    cfexp_subgraph = {}
    time_list = []
    mis_cases = 0
    if test_model == 'GCN':
        pyg_gcn = GCNtoPYG(gnn_model, device, features, labels, gcn_layer)
    else:
        pyg_gcn = gnn_model
    for target_node in tqdm(target_node_list):
        # initialize C2Explainer, use subgraph mode
        explainer = C2Explainer(epochs=200, lr=0.1, silent_mode=True, undirected=True, subgraph_mode=False)

        # config Explainer：edge perturbation, do not change node feature
        explainer = Explainer(
            model=pyg_gcn,
            algorithm=explainer,
            explanation_type='model',
            node_mask_type=None,  # do not change node feature
            edge_mask_type='object',  # only edge perturbation
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='raw',
            )
        )

        subgraph, cf_example, time_cost = generate_c2explainer_subgraph(target_node, explainer, pyg_data, adj, features,
                                                                        labels,
                                                                        pre_output,
                                                                        gnn_model, pyg_gcn, device, idx_test, gcn_layer,
                                                                        with_bias,
                                                                        counterfactual_explanation_subgraph_path,
                                                                        test_model, dataset_name, heads_num,
                                                                        output_idx=idx_test)
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
