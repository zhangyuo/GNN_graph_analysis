#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/24 09:27
# @Author   : **
# @Email    : **@**
# @File     : gnn_graph_analysis.py
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
from counterfactual_explanation_subgraph.CFExplainer_subgraph.cfexplainer_subgraph import \
    attack_cfexplanation_subgraph_generate
from evasion_attack_subgraph.GOttack_subgraph.evasion_GOttack import set_up_surrogate_model, evasion_test_acc_GCN
from instance_level_explanation_subgraph.GNNExplainer_subgraph.generate_gnnexplainer_subgraph import \
    gnnexplainer_subgraph
from model.GCN import GCN_model, adj_to_edge_index, PyGCompatibleGCN, transfer_weights, dr_data_to_pyg_data, GCNtoPYG
from subgraph_quantify.sematic_similarity.graph_embedding_vector import GATSimilarity, compute_graph_similarity
from subgraph_quantify.structual_similarity.graph_edit_distance import compute_graph_edit_distance
from subgraph_quantify.structual_similarity.maximum_commom_subgraph import maximum_common_subgraph
from utilty.attack_visualization import visualize_attack_subgraph, generate_timestamp_key
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.utils import from_networkx, subgraph, dense_to_sparse, k_hop_subgraph
from torch_geometric.transforms import RandomNodeSplit

from utilty.clean_subgraph_visualization import visualize_restricted_clean_subgraph
from utilty.maximum_common_graph_visualization import mx_com_graph_view
from config.config import HIDDEN_CHANNELS, DROPOUT, WITH_BIAS, DEVICE, GCN_LAYER
from utilty.utils import select_test_nodes, normalize_adj


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
                           evasion_attack_subgraph_path, verbose=False):
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
                                        'modified_gcn_model': gnn_model,
                                        'E_type': subgraph[2]}
    return miss / len(target_node_list), attack_subgraph


def GOttack_Poison_attack(data, attack_model, target_node_list, budget, device, features, adj, labels, test_model,
                          poison_attack_subgraph_path, verbose=False):
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
                             labels, idx_train, idx_val, device, test_model, path, attack_type, dataset_name):
    attack_subgraph = {}
    if attack_method == 'GOttack':
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
                                                                          features, adj, labels, test_model, path)
            elif attack_type == "Poison":
                miss_percentage, attack_subgraph = GOttack_Poison_attack(data, model, target_node_list, budget, device,
                                                                         features, adj, labels, test_model, path)

    else:
        pass
    return attack_subgraph


def clean_explanation_subgraph_generate(explainer, pyg_data, explainer_method, target_node_list, labels, features,
                                        attack_subgraph, instance_level_explanation_subgraph_path):
    clean_explanation_subgraph = {}
    if explainer_method == "GNNExplainer":
        for target_node in tqdm(target_node_list):
            attack_subgraph_edge_num = len(attack_subgraph[target_node]['at_subgraph'].edges)
            subgraph = gnnexplainer_subgraph(explainer, pyg_data, target_node, labels, features,
                                             instance_level_explanation_subgraph_path, attack_subgraph_edge_num,
                                             ex_type='clean')
            clean_explanation_subgraph[target_node] = subgraph
    return clean_explanation_subgraph


def attack_explanation_subgraph_generate(attack_type, explainer, attack_subgraph, explainer_method, target_node_list,
                                         labels, features, instance_level_explanation_subgraph_path,
                                         device, pyg_data=None):
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
                                                 modified_features, instance_level_explanation_subgraph_path,
                                                 attack_subgraph_edge_num,
                                                 ex_type='attack')
            elif attack_type == "Poison":
                # change explainer, use clean graph
                start_time = time.time()
                modified_explainer = gnn_explainer_generate(modified_gcn_model, device, modified_features,
                                                            modified_labels)
                elapsed = time.time() - start_time
                print(f"GNN explainer newly trained in {elapsed:.4f}s!")
                subgraph = gnnexplainer_subgraph(modified_explainer, pyg_data, target_node, modified_labels,
                                                 modified_features, instance_level_explanation_subgraph_path,
                                                 attack_subgraph_edge_num,
                                                 ex_type='attack')

            attacked_explanation_subgraph[target_node] = subgraph

    return attacked_explanation_subgraph


def clean_subgraph_generate(target_node_list, adj, labels, features, clean_subgraph_path, gcn_layer):
    clean_subgraph = {}
    max_nodes = 50  # ensure comparison graph has more nodes
    k_hop = gcn_layer
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
                      idx_train, idx_val, gcn_layer, dataset_name, clean_subgraph_path, evasion_attack_subgraph_path,
                      instance_level_explanation_subgraph_path, poison_attack_subgraph_path, pre_output, idx_test,
                      with_bias, counterfactual_explanation_subgraph_path):
    # generate subgraph
    attack_subgraph = None
    clean_explanation_subgraph = None
    attacked_explanation_subgraph = None
    clean_subgraph = None
    if attack_type == 'Evasion' and explanation_type == 'instance-level':
        clean_subgraph = clean_subgraph_generate(target_node_list, adj, labels, features, clean_subgraph_path,
                                                 gcn_layer)
        # Evasion attack & instance-level explainer
        attack_subgraph = attack_subgraph_generate(gnn_model, attack_method, attack_budget_list, target_node_list, data,
                                                   features, adj, labels, idx_train, idx_val, device, test_model,
                                                   evasion_attack_subgraph_path, attack_type, dataset_name)
        clean_explanation_subgraph = clean_explanation_subgraph_generate(explainer, pyg_data, explainer_method,
                                                                         target_node_list, labels, features,
                                                                         attack_subgraph,
                                                                         instance_level_explanation_subgraph_path)
        attacked_explanation_subgraph = attack_explanation_subgraph_generate(attack_type, explainer, attack_subgraph,
                                                                             explainer_method, target_node_list,
                                                                             labels, features,
                                                                             instance_level_explanation_subgraph_path,
                                                                             device)
    elif attack_type == 'Poison' and explanation_type == 'instance-level':
        # Poison attack & instance-level explainer
        attack_subgraph = attack_subgraph_generate(gnn_model, attack_method, attack_budget_list, target_node_list, data,
                                                   features, adj, labels, idx_train, idx_val, device, test_model,
                                                   poison_attack_subgraph_path, attack_type, dataset_name)
        clean_explanation_subgraph = clean_explanation_subgraph_generate(explainer, pyg_data, explainer_method,
                                                                         target_node_list, labels, features,
                                                                         instance_level_explanation_subgraph_path)
        # new modified GNN model will bring new modified GNN explainer
        modified_explainer = None  # rebuild for each node
        # poison attack use clean subgraph
        attacked_explanation_subgraph = attack_explanation_subgraph_generate(attack_type, modified_explainer,
                                                                             attack_subgraph,
                                                                             explainer_method, target_node_list,
                                                                             labels, features,
                                                                             instance_level_explanation_subgraph_path,
                                                                             device, pyg_data=pyg_data)
    elif attack_type == 'Poison' and explanation_type == 'class-level':
        # Poison attack & class-level explainer
        pass
    elif attack_type == 'Evasion' and explanation_type == 'counterfactual':
        # Evasion attack & counterfactual explainer
        with open(
                "/Users/*******/Documents/PycharmProject/GNN_graph_analysis/results/2025-08-07_E-/subgraph_quantify/Evasion_GOttack_instance-level_GNNExplainer_cora_budget[5]/subgraph_data.pickle",
                "rb") as fr:
            subgraph_data = pickle.load(fr)
        attack_subgraph = subgraph_data['attack_subgraph']
        attacked_explanation_subgraph = attack_cfexplanation_subgraph_generate(target_node_list, attack_subgraph,
                                                                               features, labels, gnn_model,
                                                                               device, idx_test, gcn_layer, with_bias,
                                                                               counterfactual_explanation_subgraph_path)

    subgraph_data = {"clean_subgraph": clean_subgraph,
                     "attack_subgraph": attack_subgraph,
                     "clean_explanation_subgraph": clean_explanation_subgraph,
                     "attacked_explanation_subgraph": attacked_explanation_subgraph}
    return subgraph_data


def gnn_explainer_generate(test_model, gnn_model, device, features, labels, gcn_layer, epoch=200):
    if test_model == "GCN":
        pyg_gcn = GCNtoPYG(gnn_model, device, features, labels, gcn_layer)
    else:
        pyg_gcn = gnn_model

    # Create explainer (using PyG-formatted data)
    explainer = Explainer(
        model=pyg_gcn,
        algorithm=GNNExplainer(
            epochs=epoch,  # 减少训练轮次
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


def pg_explainer_generate(test_model, gnn_model, device, features, labels, gcn_layer, pyg_data, data, epochs=30):
    if test_model == "GCN":
        pyg_gcn = GCNtoPYG(gnn_model, device, features, labels, gcn_layer)
    else:
        pyg_gcn = gnn_model
    # transform = RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)
    # pyg_data = transform(pyg_data)

    # explainer = PGExplainer(model, emb_dim=64, hidden_dim=32, lr=0.01, epochs=20, return_type='log_probs')

    # Create explainer (using PyG-formatted data)
    explainer = Explainer(
        model=pyg_gcn,
        algorithm=PGExplainer(
            epochs=epochs,  # 减少训练轮次
            # lr=0.1,  # 提高学习率
            log=False,  # 禁用日志
            # coeffs={'edge_size': 0.005, 'node_feat_size': 0.1}  # 添加正则化防止梯度爆炸
        ),
        explanation_type='phenomenon',  # PGExplainer only supports explanations of the 'phenomenon' type
        edge_mask_type='object',  # PGExplainer主要生成边掩码
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs'
        )
    )
    # PGExplainer needs to be trained based on nodes set after initializing
    train_indices = pyg_data.train_mask.nonzero(as_tuple=True)[0]  # 获取训练节点索引

    # 新增：过滤掉可能产生空子图的节点（如孤立节点）
    valid_train_indices = []
    for idx in train_indices:
        # 计算节点的度（邻居数）
        degree = (pyg_data.edge_index[0] == idx).sum().item()
        if degree > 0:  # 只保留有邻居的节点
            valid_train_indices.append(idx)
    train_indices = torch.tensor(valid_train_indices)

    if len(train_indices) == 0:
        raise ValueError("没有找到有效的训练节点（所有训练节点可能都是孤立的）")

    if test_model == "GCN":
        edge_index = pyg_data.edge_index
    else:
        dense_adj = torch.tensor(data.adj.toarray(), device=device)
        norm_adj = normalize_adj(dense_adj)
        edge_index, edge_weight = dense_to_sparse(norm_adj)

    for epoch in tqdm(range(epochs)):
        for index in train_indices:
            # Calculate the loss for each node and update the parameters of PGExplainer
            # 将索引转换为标量
            index_tensor = index.unsqueeze(0) if index.dim() == 0 else index
            if test_model == "GCN":
                loss = explainer.algorithm.train(
                    epoch,
                    pyg_gcn,
                    pyg_data.x,
                    edge_index,
                    target=pyg_data.y.to(torch.long),
                    index=index_tensor  # 传入标量
                )
            elif test_model in ["GraphTransformer", "GAT"]:
                loss = explainer.algorithm.train(
                    epoch,
                    pyg_gcn,
                    pyg_data.x,
                    edge_index,
                    edge_weight=edge_weight,
                    target=pyg_data.y.to(torch.long),
                    index=index_tensor  # 传入标量
                )
            elif test_model in ["GraphConv"]:
                loss = explainer.algorithm.train(
                    epoch,
                    pyg_gcn,
                    pyg_data.x,
                    edge_index,
                    edge_weight=edge_weight,
                    target=pyg_data.y.to(torch.long),
                    index=index_tensor  # 传入标量
                )
    return explainer


def pg_explainer_generate_batch(test_model, gnn_model, device, features, labels, gcn_layer, pyg_data, data, target_node_list, epochs=30):
    if test_model == "GCN":
        pyg_gcn = GCNtoPYG(gnn_model, device, features, labels, gcn_layer)
    else:
        pyg_gcn = gnn_model

    pyg_gcn.eval()
    pyg_gcn = pyg_gcn.to(device)
    pyg_data.x = pyg_data.x.to(device)
    pyg_data.edge_index = pyg_data.edge_index.to(device)
    pyg_data.y = pyg_data.y.to(device)

    # ⚡ 使用 Explainer 封装 PGExplainer
    explainer = Explainer(
        model=pyg_gcn,
        algorithm=PGExplainer(epochs=epochs, lr=0.01, emb_dim=128, hidden_dim=64),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs'
        )
    )

    # ⚡ 先训练 PGExplainer
    for epoch in range(epochs):
        total_loss = 0
        for target in target_node_list:
            # 抽取 k-hop 子图
            subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
                target, num_hops=gcn_layer, edge_index=pyg_data.edge_index, relabel_nodes=True
            )
            x_sub = pyg_data.x[subset]
            y_sub = pyg_data.y[subset]

            # train 方法更新内部 mask
            loss = explainer.algorithm.train(
                model=pyg_gcn,
                x=x_sub,
                epoch=epochs,
                edge_index=edge_index_sub,
                target=y_sub,
                index=mapping
            )
            total_loss += loss

        print(f"Epoch {epoch + 1}, Avg loss: {total_loss / len(target_node_list):.4f}")

    # # ⚡ 训练完毕后，再用 explainer() 对单节点解释
    # # 示例: target_node = target_node_list[0]
    # subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
    #     target_node_list[0], num_hops=gcn_layer, edge_index=pyg_data.edge_index, relabel_nodes=True
    # )
    # x_sub = pyg_data.x[subset]
    # y_sub = pyg_data.y[subset]
    #
    # explanation = explainer(
    #     x=x_sub,
    #     edge_index=edge_index_sub,
    #     target=y_sub,
    #     node_idx=mapping.tolist().index(0)  # target 在子图中的位置
    # )

    return explainer

def subgraph_quantify(attack_subgraph, explanation_subgraph, pyg_data, pyg_gnn_model, graph_analysis_subgraph_path,
                      target_node_list, target_node_list1,
                      pic_name='at_vs_ex', is_cf_explainer=False):
    similarity_dict = {}
    for target_node, subgraph_data in tqdm(attack_subgraph.items()):
        # if target_node != 1697:
        #     continue
        if is_cf_explainer and not explanation_subgraph[target_node]:
            continue
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

        if target_node in target_node_list:
            similarity_dict[target_node]["trained_gcn_test_state"] = "success"
        elif target_node in target_node_list1:
            similarity_dict[target_node]["trained_gcn_test_state"] = "failed"
        else:
            similarity_dict[target_node]["trained_gcn_test_state"] = "none"

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
    gcn_layer = GCN_LAYER

    # attack parameters
    attack_type = 'Evasion'  # ['Evasion', 'Poison']
    attack_budget_list = [5]  # [5,4,3,2,1]
    attack_method = 'GOttack'  # GOttack

    # explainer parameters
    explanation_type = 'instance-level'  # ['instance-level', 'class-level', 'counterfactual']
    explainer_method = 'GNNExplainer'  # ['GNNExplainer']

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
    explainer = gnn_explainer_generate(gnn_model, device, features, labels, gcn_layer)

    ######################### generate subgraph  #########################
    subgraph_data = generate_subgraph(attack_type, explanation_type, target_node_list, gnn_model, explainer, pyg_data,
                                      device, test_model, attack_method, attack_budget_list, explainer_method, data,
                                      features, adj, labels, idx_train, idx_val, gcn_layer, dataset_name,
                                      clean_subgraph_path,
                                      evasion_attack_subgraph_path, instance_level_explanation_subgraph_path,
                                      poison_attack_subgraph_path)
    with open(graph_analysis_subgraph_path + "/subgraph_data.pickle", "wb") as fw:
        pickle.dump(subgraph_data, fw)

    ######################### subgraph quantify  #########################
    if not gnn_model:
        file_path = os.path.join(graph_analysis_subgraph_path, 'gcn_model.pth')
        gnn_model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device,
                        with_bias=WITH_BIAS)
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
    pyg_gnn_model = GCNtoPYG(gnn_model, device, features, labels, gcn_layer)
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
