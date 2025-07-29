#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/24 09:27
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : graph_analysis.py
# @Software : PyCharm
# @Desc     :
"""
import os
import sys
import random
import time

import networkx as nx
import numpy as np
import torch
import pickle
import pandas as pd
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import classification_margin
from scipy.sparse import issparse
from tqdm import tqdm
import scipy.sparse as sp
from attack.GOttack.OrbitAttack import OrbitAttack
from attack.GOttack.orbit_table_generator import OrbitTableGenerator
from evasion_attack_subgraph.GOttack_subgraph.evasion_GOttack import set_up_surrogate_model, evasion_test_acc_GCN
from instance_level_subgraph.GNNExplainer_subgraph.generate_gnnexplainer_subgraph import gnnexplainer_subgraph
from model.gcn_model import GCN_model
from model.model_transfer import adj_to_edge_index, PyGCompatibleGCN, transfer_weights, dr_data_to_pyg_data
from subgraph_quantify.sematic_similarity.graph_embedding_vector import GATSimilarity, compute_graph_similarity
from subgraph_quantify.structual_similarity.graph_edit_distance import compute_graph_edit_distance
from subgraph_quantify.structual_similarity.maximum_commom_subgraph import maximum_common_subgraph
from utilty.attack_visualization import visualize_attack_subgraph, generate_timestamp_key
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import from_networkx, subgraph

from utilty.clean_subgraph_visualization import visualize_restricted_clean_subgraph
from utilty.maximum_common_graph_visualization import mx_com_graph_view


def select_test_nodes(attack_type, explanation_type, idx_test, ori_output, labels):
    """
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    :param attack_type:
    :param explanation_type:
    :param idx_test:
    :return:
    """
    node_list = []
    if attack_type is None:
        pass
    else:
        margin_dict = {}
        for idx in idx_test:
            margin = classification_margin(ori_output[idx], labels[idx])
            if margin < 0:  # only keep the nodes correctly classified
                continue
            margin_dict[idx] = margin
        sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)
        high = []
        low = []
        other = []
        for class_num in set(labels):
            class_num_sorted_margins = [x for x, y in sorted_margins if labels[x] == class_num]
            high += [x for x in class_num_sorted_margins[: 10]]
            low += [x for x in class_num_sorted_margins[-10:]]
            other_0 = [x for x in class_num_sorted_margins[10: -10]]
            other += np.random.choice(other_0, 20, replace=False).tolist()

        node_list += high + low + other
        node_list = [int(x) for x in node_list]

    return node_list


def evasion_test_acc_GCN(gnn_model, modified_adj, features, data, target_node):
    labels = data.labels
    ''' test on GCN '''
    output = gnn_model.predict(features=features, adj=modified_adj)
    probs = torch.exp(output[[target_node]])[0]
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    acc = acc_test.item()
    changed_label = output.argmax(1)[target_node]
    modified_labels = np.array(output.argmax(1))
    return acc, changed_label, modified_labels


def poison_test_acc_GCN(adj, features, data, target_node, device):
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
    acc = acc_test.item()
    changed_label = output.argmax(1)[target_node]
    modified_labels = np.array(output.argmax(1))
    modified_gcn_model = gcn
    return acc, changed_label, modified_labels, modified_gcn_model


def GOttack_Evasion_attack(gnn_model, data, attack_model, target_node_list, budget, features, adj, labels, test_model,
                           max_nodes, k_hop, evasion_attack_subgraph_path, verbose=False):
    miss = 0
    false_class_node = []
    attack_subgraph = {}
    for target_node in tqdm(target_node_list):
        # attack_model.attack(features, adj, labels, target_node, budget, verbose=verbose)
        attack_model.attack_e_minus(features, adj, labels, target_node, budget, verbose=verbose)
        modified_adj = attack_model.modified_adj  # get modified adj for one target node in a given budget, you can used in evasion and poisoning attack stages
        changed_label = None
        modified_labels = None
        if test_model == 'GCN':
            acc, changed_label, modified_labels = evasion_test_acc_GCN(gnn_model, modified_adj, features, data,
                                                                       target_node)
        else:
            raise Exception("Test model is not supported")

        if target_node in false_class_node or acc == 0:
            miss += 1
            attack_state = 'success'
        else:
            attack_state = 'fail'
        subgraph = visualize_attack_subgraph(
            modified_adj,
            adj,
            labels,
            features,
            changed_label,
            target_node,
            attack_state,
            k_hop=k_hop,
            max_nodes=max_nodes,
            title="Visualization for adversarial attack subgraph",
            pic_path=evasion_attack_subgraph_path
        )
        print("generate evasion attack subgraph for node {}".format(target_node))
        attack_subgraph[target_node] = {'at_subgraph': subgraph[0],
                                        'at_subgraph_visual': subgraph[0],
                                        'attack_state': attack_state,
                                        'original_label': labels[target_node],
                                        'changed_label': changed_label,
                                        'budget': budget,
                                        'modified_adj': modified_adj,
                                        'modified_features': None,
                                        'modified_labels': modified_labels,
                                        'modified_gcn_model': None,
                                        'E_type': subgraph[2]}
    return miss / len(target_node_list), attack_subgraph


def GOttack_Poison_attack(data, attack_model, target_node_list, budget, device, features, adj, labels, test_model,
                          max_nodes, k_hop, poison_attack_subgraph_path, verbose=False):
    miss = 0
    false_class_node = []
    attack_subgraph = {}
    for target_node in tqdm(target_node_list):
        attack_model.attack(features, adj, labels, target_node, budget, verbose=verbose)
        modified_adj = attack_model.modified_adj
        changed_label = None
        modified_labels = None
        if test_model == 'GCN':
            acc, changed_label, modified_labels, modified_gcn_model = poison_test_acc_GCN(modified_adj, features, data,
                                                                                          target_node, device)
        else:
            raise Exception("Test model is not supported")

        if target_node in false_class_node or acc == 0:
            miss += 1
            attack_state = 'success'
        else:
            attack_state = 'fail'
        subgraph = visualize_attack_subgraph(
            modified_adj,
            adj,
            labels,
            features,
            changed_label,
            target_node,
            attack_state,
            k_hop=k_hop,
            max_nodes=max_nodes,
            title="Visualization for adversarial attack subgraph",
            pic_path=poison_attack_subgraph_path
        )
        print("generate poison attack subgraph for node {}".format(target_node))
        attack_subgraph[target_node] = {'at_subgraph': subgraph[0],
                                        'at_subgraph_visual': subgraph[0],
                                        'attack_state': attack_state,
                                        'original_label': labels[target_node],
                                        'changed_label': changed_label,
                                        'budget': budget,
                                        'modified_adj': modified_adj,
                                        'modified_features': None,
                                        'modified_labels': modified_labels,
                                        'modified_gcn_model': modified_gcn_model}
    return miss / len(target_node_list), attack_subgraph


def attack_subgraph_generate(gnn_model, attack_method, attack_budget_list, target_node_list, data, features, adj,
                             labels, idx_train, idx_val, device, test_model, max_nodes, k_hop, path, attack_type):
    attack_subgraph = {}
    if attack_method == '1518':
        # Orbit attack(1518)
        df_orbit = OrbitTableGenerator(dataset_name).generate_orbit_table()
        for budget in attack_budget_list:  # out-layer given a fixed budget, it's an isolated test
            # Orbit attack(1518)
            surrogate = set_up_surrogate_model(features, adj, labels, idx_train, idx_val,
                                               device=device)  # 代理损失:gnn model
            model = OrbitAttack(surrogate, df_orbit, nnodes=adj.shape[0], device=device)  # initialize the attack model
            model = model.to(device)
            if attack_type == "Evasion":
                miss_percentage, attack_subgraph = GOttack_Evasion_attack(gnn_model, data, model, target_node_list,
                                                                          budget,
                                                                          features, adj, labels, test_model, max_nodes,
                                                                          k_hop, path)
            elif attack_type == "Poison":
                miss_percentage, attack_subgraph = GOttack_Poison_attack(data, model, target_node_list, budget, device,
                                                                         features, adj, labels, test_model, max_nodes,
                                                                         k_hop, path)

    else:
        pass
    return attack_subgraph


def clean_explanation_subgraph_generate(explainer, pyg_data, explainer_method, target_node_list, labels, features,
                                        attack_subgraph, explainer_edge_important_threshold, max_nodes,
                                        instance_level_explanation_subgraph_path):
    clean_explanation_subgraph = {}
    if explainer_method == "GNNExplainer":
        for target_node in tqdm(target_node_list):
            attack_subgraph_edge_num = len(attack_subgraph[target_node]['at_subgraph'].edges)
            subgraph = gnnexplainer_subgraph(explainer, pyg_data, target_node, labels, features, max_nodes,
                                             instance_level_explanation_subgraph_path, attack_subgraph_edge_num,
                                             threshold=explainer_edge_important_threshold, ex_type='clean')
            clean_explanation_subgraph[target_node] = subgraph
    return clean_explanation_subgraph


def attack_explanation_subgraph_generate(attack_type, explainer, attack_subgraph, explainer_method, target_node_list,
                                         labels, features, max_nodes, instance_level_explanation_subgraph_path,
                                         explainer_edge_important_threshold, device, pyg_data=None):
    attacked_explanation_subgraph = {}
    if explainer_method == "GNNExplainer":
        for target_node in tqdm(target_node_list):
            at_sbg_dt = attack_subgraph[target_node]
            modified_labels = at_sbg_dt['modified_labels']
            modified_adj = at_sbg_dt['modified_adj']
            modified_features = at_sbg_dt['modified_features'] if at_sbg_dt['modified_features'] else features
            modified_gcn_model = at_sbg_dt['modified_gcn_model']
            attack_subgraph_edge_num = len(attack_subgraph[target_node]['at_subgraph'].edges)
            subgraph = None
            if attack_type == "Evasion":
                # change pyg_data, features, labels
                attacked_pyg_data = dr_data_to_pyg_data(modified_adj, modified_features, modified_labels)
                subgraph = gnnexplainer_subgraph(explainer, attacked_pyg_data, target_node, modified_labels,
                                                 modified_features, max_nodes, instance_level_explanation_subgraph_path,
                                                 attack_subgraph_edge_num,
                                                 threshold=explainer_edge_important_threshold,
                                                 ex_type='attack')
            elif attack_type == "Poison":
                # change explainer, use clean graph
                start_time = time.time()
                modified_explainer = gnn_explainer_generate(modified_gcn_model, device, modified_features,
                                                            modified_labels)
                elapsed = time.time() - start_time
                print(f"GNN explainer newly trained in {elapsed:.4f}s!")
                subgraph = gnnexplainer_subgraph(modified_explainer, pyg_data, target_node, modified_labels,
                                                 modified_features, max_nodes, instance_level_explanation_subgraph_path,
                                                 attack_subgraph_edge_num,
                                                 threshold=explainer_edge_important_threshold,
                                                 ex_type='attack')

            attacked_explanation_subgraph[target_node] = subgraph

    return attacked_explanation_subgraph


def clean_subgraph_generate(target_node_list, adj, labels, features, max_nodes, k_hop, clean_subgraph_path):
    clean_subgraph = {}
    max_nodes = 50  # ensure comparison graph has more nodes
    k_hop = 2
    for target_node in tqdm(target_node_list):
        subgraph = visualize_restricted_clean_subgraph(
            adj,
            labels,
            features,
            target_node,
            k_hop=k_hop,
            max_nodes=max_nodes,
            title="Visualization for Clean Subgraph",
            pic_path=clean_subgraph_path
        )
        clean_subgraph[target_node] = subgraph
    return clean_subgraph


def generate_subgraph(attack_type, explanation_type, target_node_list, gnn_model, explainer, pyg_data, device,
                      test_model, attack_method, attack_budget_list, explainer_method, data, features, adj, labels,
                      idx_train, idx_val, explainer_edge_important_threshold, max_nodes, k_hop):
    # generate subgraph
    attack_subgraph = None
    clean_explanation_subgraph = None
    attacked_explanation_subgraph = None
    clean_subgraph = clean_subgraph_generate(target_node_list, adj, labels, features, max_nodes, k_hop,
                                             clean_subgraph_path)
    if attack_type == 'Evasion' and explanation_type == 'instance-level':
        # Evasion attack & instance-level explainer
        attack_subgraph = attack_subgraph_generate(gnn_model, attack_method, attack_budget_list, target_node_list, data,
                                                   features, adj, labels, idx_train, idx_val, device, test_model,
                                                   max_nodes, k_hop, evasion_attack_subgraph_path, attack_type)
        clean_explanation_subgraph = clean_explanation_subgraph_generate(explainer, pyg_data, explainer_method,
                                                                         target_node_list, labels, features,
                                                                         attack_subgraph,
                                                                         explainer_edge_important_threshold, max_nodes,
                                                                         instance_level_explanation_subgraph_path)
        attacked_explanation_subgraph = attack_explanation_subgraph_generate(attack_type, explainer, attack_subgraph,
                                                                             explainer_method, target_node_list,
                                                                             labels, features, max_nodes,
                                                                             instance_level_explanation_subgraph_path,
                                                                             explainer_edge_important_threshold, device)
    elif attack_type == 'Poison' and explanation_type == 'instance-level':
        # Poison attack & instance-level explainer
        attack_subgraph = attack_subgraph_generate(gnn_model, attack_method, attack_budget_list, target_node_list, data,
                                                   features, adj, labels, idx_train, idx_val, device, test_model,
                                                   max_nodes, k_hop, poison_attack_subgraph_path, attack_type)
        clean_explanation_subgraph = clean_explanation_subgraph_generate(explainer, pyg_data, explainer_method,
                                                                         target_node_list, labels, features,
                                                                         explainer_edge_important_threshold, max_nodes,
                                                                         instance_level_explanation_subgraph_path)
        # new modified GNN model will bring new modified GNN explainer
        modified_explainer = None  # rebuild for each node
        # poison attack use clean subgraph
        attacked_explanation_subgraph = attack_explanation_subgraph_generate(attack_type, modified_explainer,
                                                                             attack_subgraph,
                                                                             explainer_method, target_node_list,
                                                                             labels, features, max_nodes,
                                                                             instance_level_explanation_subgraph_path,
                                                                             explainer_edge_important_threshold, device,
                                                                             pyg_data=pyg_data)
    elif attack_type == 'Poison' and explanation_type == 'class-level':
        # Poison attack & class-level explainer
        pass
    elif attack_type == 'Evasion' and explanation_type == 'counterfactual':
        # Evasion attack & counterfactual explainer
        pass
    subgraph_data = {"clean_subgraph": clean_subgraph,
                     "attack_subgraph": attack_subgraph,
                     "clean_explanation_subgraph": clean_explanation_subgraph,
                     "attacked_explanation_subgraph": attacked_explanation_subgraph}
    return subgraph_data


def GCNtoPYG(gnn_model, device, features, labels):
    # initialize pyg gcn model
    pyg_gcn = PyGCompatibleGCN(
        in_channels=features.shape[1],
        hidden_channels=16,
        out_channels=labels.max().item() + 1,
        bias=True
    )
    pyg_gcn = pyg_gcn.to(device)

    # Initialize model (using deeprobust adjacency matrix)
    dr_trained_model = gnn_model
    pyg_gcn = transfer_weights(dr_trained_model, pyg_gcn)
    return pyg_gcn


def gnn_explainer_generate(gnn_model, device, features, labels):
    pyg_gcn = GCNtoPYG(gnn_model, device, features, labels)

    # Create explainer (using PyG-formatted data)
    explainer = Explainer(
        model=pyg_gcn,
        algorithm=GNNExplainer(
            epochs=200,  # 减少训练轮次
            # lr=0.1,  # 提高学习率
            log=False,  # 禁用日志
            # coeffs={'edge_size': 0.005, 'node_feat_size': 0.1}  # 添加正则化防止梯度爆炸
        ),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs'
        )
    )
    return explainer


def subgraph_quantify(attack_subgraph, explanation_subgraph, pyg_data, pyg_gnn_model, pic_name='at_vs_ex'):
    similarity_dict = {}
    for target_node, subgraph_data in tqdm(attack_subgraph.items()):
        # if target_node != 1697:
        #     continue
        similarity_dict[target_node] = {}
        print(f"current target node {target_node} ~~~~~~~~~~~~~~~~~~")
        at_subgraph = subgraph_data['at_subgraph']
        ex_subgraph = explanation_subgraph[target_node]
        attack_state = subgraph_data['attack_state']
        similarity_dict[target_node]["attack_state"] = attack_state
        edge_attack_type = subgraph_data['E_type']
        similarity_dict[target_node]["edge_attack_type"] = edge_attack_type

        # graph edit distance
        ged = compute_graph_edit_distance(at_subgraph, ex_subgraph)
        similarity_dict[target_node]["ged"] = ged

        # maximum common graph
        best_connected_common_subgraph, mapping, mcs = maximum_common_subgraph(at_subgraph, ex_subgraph,
                                                                               target_node=target_node)
        print("MCS Nodes:", best_connected_common_subgraph.nodes)
        print("Node Mapping:", mapping)
        mx_com_graph_view(target_node, at_subgraph, ex_subgraph, best_connected_common_subgraph, mapping,
                          graph_analysis_subgraph_path,
                          graph_one_name="Attack Subgraph",
                          graph_two_name="Explanation Subgraph", pic_name=pic_name)
        similarity_dict[target_node]["mcs"] = mcs

        # graph embedding vector
        start_time = time.time()
        similarity = compute_graph_similarity(at_subgraph, ex_subgraph, pyg_data, pyg_gnn_model)
        print(f"Similarity between at_subgraph and ex_subgraph: {similarity}")
        elapsed = time.time() - start_time
        print(f"graph embedding vector computed in {elapsed:.4f}s!")
        similarity_dict[target_node]["gev_at_ex"] = similarity

        similarity = compute_graph_similarity(at_subgraph, best_connected_common_subgraph, pyg_data, pyg_gnn_model)
        print(f"Similarity between at_subgraph and mcs: {similarity}")
        similarity_dict[target_node]["gev_at_mcs"] = similarity

        similarity = compute_graph_similarity(ex_subgraph, best_connected_common_subgraph, pyg_data, pyg_gnn_model)
        print(f"Similarity between at_subgraph and mcs: {similarity}")
        similarity_dict[target_node]["gev_ex_mcs"] = similarity

    return similarity_dict


if __name__ == "__main__":
    # target node for node classification on GNN model
    res = os.path.abspath(__file__)  # acquire absolute path of current file
    base_path = os.path.dirname(os.path.dirname(res))
    sys.path.insert(0, base_path)
    # sys.path.append(base_path + "/lib")

    ######################### initialize random state  #########################
    np.random.seed(102)

    ######################### initialize data experiment  #########################
    # general parameters
    dataset_name = 'cora'
    test_model = 'GCN'  # GSAGE, GCN, GIN
    device = "cpu"
    max_nodes = None
    k_hop = None

    # attack parameters
    attack_type = 'Evasion'  # ['Evasion', 'Poison']
    attack_budget_list = [5]  # [5,4,3,2,1]
    attack_method = '1518'  # ['1518'], 1518 means GOttack

    # explainer parameters
    explanation_type = 'instance-level'  # ['instance-level', 'class-level', 'counterfactual']
    explainer_method = 'GNNExplainer'  # ['GNNExplainer']
    explainer_edge_important_threshold = None  # explainer edge important threshold

    # create data results path
    # time_name = generate_timestamp_key()  # different files path for each test
    time_name = '20250716'  # for debug test
    # clean subgraph path
    clean_subgraph_path = base_path + f'/clean_subgraph/results_{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_{attack_budget_list}_{time_name}'
    if not os.path.exists(clean_subgraph_path):
        os.makedirs(clean_subgraph_path)
    # evasion attack subgraph path
    evasion_attack_subgraph_path = None
    if attack_type == 'Evasion':
        evasion_attack_subgraph_path = base_path + f'/evasion_attack_subgraph/results_{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_{attack_budget_list}_{time_name}'
        if not os.path.exists(evasion_attack_subgraph_path):
            os.makedirs(evasion_attack_subgraph_path)
    # poison attack subgraph path
    poison_attack_subgraph_path = None
    if attack_type == 'Poison':
        poison_attack_subgraph_path = base_path + f'/poison_attack_subgraph/results_{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_{attack_budget_list}_{time_name}'
        if not os.path.exists(poison_attack_subgraph_path):
            os.makedirs(poison_attack_subgraph_path)
    # instance level explanation subgraph path
    instance_level_explanation_subgraph_path = None
    if explanation_type == 'instance-level':
        instance_level_explanation_subgraph_path = base_path + f'/instance_level_subgraph/results_{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_{attack_budget_list}_{time_name}'
        if not os.path.exists(instance_level_explanation_subgraph_path):
            os.makedirs(instance_level_explanation_subgraph_path)
    # class level explanation subgraph path
    class_level_explanation_subgraph_path = None
    if explanation_type == 'class-level':
        class_level_explanation_subgraph_path = base_path + f'/class_level_subgraph/results_{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_{attack_budget_list}_{time_name}'
        if not os.path.exists(class_level_explanation_subgraph_path):
            os.makedirs(class_level_explanation_subgraph_path)
    # counterfactual explanation subgraph path
    counterfactual_explanation_subgraph_path = None
    if explanation_type == 'counterfactual':
        counterfactual_explanation_subgraph_path = base_path + f'/counterfactual_subgraph/results_{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_{attack_budget_list}_{time_name}'
        if not os.path.exists(counterfactual_explanation_subgraph_path):
            os.makedirs(counterfactual_explanation_subgraph_path)
    # graph analysis subgraph path
    graph_analysis_subgraph_path = base_path + f'/subgraph_quantify/results_{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_{attack_budget_list}_{time_name}'
    if not os.path.exists(graph_analysis_subgraph_path):
        os.makedirs(graph_analysis_subgraph_path)
    # dataset path
    dataset_path = base_path + '/dataset'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    ######################### loading deeprobust dataset  #########################
    # adjacency matrix is a high compressed sparse row format
    data = Dataset(root=dataset_path, name=dataset_name)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    # Create PyG Data object
    pyg_data = dr_data_to_pyg_data(adj, features, labels)

    # save subgraph results
    subgraph_data = None
    gnn_model = None

    ######################### GNN model generate  #########################
    ori_output = None
    if test_model == 'GCN':
        gnn_model, ori_output = GCN_model(adj, features, labels, device, idx_train, idx_val)
    file_path = os.path.join(graph_analysis_subgraph_path, 'gcn_model.pth')
    torch.save(gnn_model.state_dict(), file_path)

    ######################### select test nodes  #########################
    target_node_list = select_test_nodes(attack_type, explanation_type, idx_test, ori_output, labels)
    # target_node_list = target_node_list[0:20]

    ######################### GNN explainer generate  #########################
    explainer = gnn_explainer_generate(gnn_model, device, features, labels)

    ######################### generate subgraph  #########################
    subgraph_data = generate_subgraph(attack_type, explanation_type, target_node_list, gnn_model, explainer, pyg_data,
                                      device, test_model, attack_method, attack_budget_list, explainer_method, data,
                                      features, adj, labels, idx_train, idx_val, explainer_edge_important_threshold,
                                      max_nodes, k_hop)
    with open(graph_analysis_subgraph_path + "/subgraph_data.pickle", "wb") as fw:
        pickle.dump(subgraph_data, fw)

    ######################### subgraph quantify  #########################
    if not gnn_model:
        file_path = os.path.join(graph_analysis_subgraph_path, 'gcn_model.pth')
        gnn_model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        gnn_model.load_state_dict(torch.load(file_path))
        gnn_model.eval()
    if not subgraph_data:
        with open(graph_analysis_subgraph_path + "/subgraph_data.pickle", "rb") as fr:
            subgraph_data = pickle.load(fr)
    clean_subgraph = subgraph_data['clean_subgraph']
    attack_subgraph = subgraph_data['attack_subgraph']
    clean_explanation_subgraph = subgraph_data['clean_explanation_subgraph']
    attacked_explanation_subgraph = subgraph_data['attacked_explanation_subgraph']

    # # attack subgraph vs. clean explanation subgraph
    pyg_gnn_model = GCNtoPYG(gnn_model, device, features, labels)
    similarity_dict_1 = subgraph_quantify(attack_subgraph, clean_explanation_subgraph, pyg_data, pyg_gnn_model,
                                          pic_name='at_vs_cl-ex')

    # attack subgraph vs. attacked explanation subgraph
    similarity_dict_2 = subgraph_quantify(attack_subgraph, attacked_explanation_subgraph, pyg_data, pyg_gnn_model,
                                          pic_name='at_vs_at-ex')

    # Save all results
    result1 = pd.DataFrame(similarity_dict_1.values(),
                           columns=['attack_state', 'edge_attack_type', 'ged', 'mcs', 'gev_at_ex',
                                    'gev_at_mcs', 'gev_ex_mcs'],
                           index=similarity_dict_1.keys())
    result1.to_csv(graph_analysis_subgraph_path + '/result_{}_{}_test.csv'.format('all', 'at_vs_cl-ex'))
    result2 = pd.DataFrame(similarity_dict_2.values(),
                           columns=['attack_state', 'edge_attack_type', 'ged', 'mcs', 'gev_at_ex',
                                    'gev_at_mcs', 'gev_ex_mcs'],
                           index=similarity_dict_2.keys())
    result2.to_csv(graph_analysis_subgraph_path + '/result_{}_{}_test.csv'.format('all', 'at_vs_at-ex'))

    print("graph analysis done")
