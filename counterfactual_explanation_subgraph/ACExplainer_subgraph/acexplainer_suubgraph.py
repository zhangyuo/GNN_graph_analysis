#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/8/27 20:19
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : acexplainer_suubgraph.py
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
import networkx as nx
from torch_geometric.utils import k_hop_subgraph, to_dense_adj
from tqdm import tqdm

from config.config import *
from explainer.ac_explanation.ac_explainer import ACExplainer
from explainer.ac_explanation.gcn_perturb import GCNPerturb
from model.GCN import GCN_model, dr_data_to_pyg_data, GCNtoPYG, load_GCN_model
from utilty.utils import safe_open, get_neighbourhood, normalize_adj, select_test_nodes
import torch.nn.functional as F


def generate_ac_explanation(target_node: int,
                            data,
                            pyg_data,
                            gnn_model,
                            output,
                            gcn_layer: int = 2,
                            attack_method: str = "GOttack",
                            top_t: int = 10,
                            device: str = "cuda",
                            nhid: int = 16,
                            dropout: float=0.5,
                            with_bias: bool=True):
    """
    生成AC-Explainer解释
    """
    # 1. 提取目标节点的l-hop子图
    node_idx, edge_index, mapping, _ = k_hop_subgraph(
        node_idx=target_node,
        num_hops=gcn_layer + 1, # 覆盖GCN的感受野
        edge_index=pyg_data.edge_index,
        relabel_nodes=True,
        num_nodes=pyg_data.edge_index.max() + 1
    )

    # 2. 获取攻击节点并映射到原始图索引
    attack_nodes = get_attack_nodes(target_node, pyg_data, attack_method, top_t)

    # 3. 构建扩展节点集合 (l-hop节点 + 攻击节点)
    extended_nodes = list(set(node_idx.tolist() + attack_nodes))
    extended_nodes.sort()

    # 4. 高效创建扩展子图的邻接矩阵和原始掩码矩阵
    # extended_adj, original_adj_mask = networkx_create_extended_adj(extended_nodes, pyg_data.edge_index)
    adj_dense = data.adj.toarray()
    extended_adj = adj_dense[np.ix_(extended_nodes,extended_nodes)]
    extended_adj = torch.tensor(extended_adj, dtype=torch.float, requires_grad=True)
    # 原始掩码矩阵与邻接矩阵相同
    original_adj_mask = extended_adj.clone()

    # 5. 提取扩展子图的特征
    extended_feat = pyg_data.x[extended_nodes]

    # 6. 找到目标节点在扩展子图中的新索引
    target_node_idx = extended_nodes.index(target_node)

    # test model log-probability output is same to original prediction output
    print("Output original model, full adj: {}".format(output[target_node]))
    norm_sub_adj = normalize_adj(extended_adj)
    print("Output original model, sub adj: {}".format(gnn_model.forward(extended_feat, norm_sub_adj)[target_node_idx]))

    # 7. 创建扰动模型
    perturb_model = GCNPerturb(
        gnn_model,
        nfeat=extended_feat.size(1),
        nhid=nhid,
        nclass=data.labels.max().item() + 1,
        extended_sub_adj=extended_adj,
        target_node=target_node_idx,
        original_adj_mask=original_adj_mask,
        dropout=dropout,
        gcn_layer=gcn_layer,
        with_bias=with_bias
    ).to(device)

    # 8. 创建解释器
    explainer = ACExplainer(
        model=perturb_model,
        extended_sub_adj=extended_adj,
        sub_feat=extended_feat,
        target_node=target_node_idx,
        device=device
    )

    # 10. 训练解释器
    result = explainer.train_explanation(epochs=200)

    if result is None:
        return {"error": "No valid counterfactual found"}

    # 11. 映射回原始图索引
    added_edges = []
    removed_edges = []

    delta_A = result["delta_A"]
    for i in range(delta_A.size(0)):
        for j in range(i + 1, delta_A.size(1)):
            if delta_A[i, j] > 0.5:  # 添加的边
                orig_i = extended_nodes[i]
                orig_j = extended_nodes[j]
                added_edges.append((orig_i, orig_j))
            elif delta_A[i, j] < -0.5:  # 删除的边
                orig_i = extended_nodes[i]
                orig_j = extended_nodes[j]
                removed_edges.append((orig_i, orig_j))

    return {
        "added_edges": added_edges,
        "removed_edges": removed_edges,
        "original_pred": result["original_pred"],
        "new_pred": result["new_pred"],
        "extended_nodes": extended_nodes
    }


def get_attack_nodes(target_node, pyg_data, method="GOttack", top_t=10):
    """获取攻击节点列表"""
    if method == "GOttack":
        # 这里应实现GOttack攻击方法，返回高影响力节点
        # 简化版：选择特征最相似的节点
        target_feat = pyg_data.x[target_node]
        similarities = []

        for i in range(pyg_data.edge_index.max() + 1):
            if i != target_node:
                sim = F.cosine_similarity(
                    target_feat.unsqueeze(0),
                    pyg_data.x[i].unsqueeze(0)
                )
                similarities.append((i, sim.item()))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [node for node, sim in similarities[:top_t]]

    else:
        # 其他攻击方法
        return list(range(top_t))  # 简化返回


def has_edge(edge_index, node_i, node_j):
    """检查两个节点之间是否存在边"""
    edges = edge_index.t().tolist()
    return [node_i, node_j] in edges or [node_j, node_i] in edges


def networkx_create_extended_adj(extended_nodes, edge_index):
    """
    使用NetworkX高效创建扩展邻接矩阵
    """
    # 创建NetworkX图对象
    G = nx.Graph()

    # 添加所有扩展节点
    G.add_nodes_from(extended_nodes)

    # 添加边
    # 过滤出只包含扩展节点的边
    node_to_idx = {node: idx for idx, node in enumerate(extended_nodes)}
    filtered_edges = []
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src in node_to_idx and dst in node_to_idx:
            filtered_edges.append((src, dst))

    # edges = edge_index.t().tolist()
    G.add_edges_from(filtered_edges)

    # 生成邻接矩阵
    adj_matrix = nx.adjacency_matrix(G, nodelist=extended_nodes)

    # 转换为PyTorch张量
    extended_adj = torch.tensor(adj_matrix.todense(), dtype=torch.float)

    # 原始掩码矩阵与邻接矩阵相同
    original_adj_mask = extended_adj.clone()

    return extended_adj, original_adj_mask


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
    explanation_type = "counterfactual"
    attack_method = ATTACK_METHOD
    attack_budget_list = ATTACK_BUDGET_LIST
    explainer_method = "ACExplainer"
    top_t = MAX_ATTACK_NODES_NUM


    np.random.seed(SEED_NUM)

    time_name = datetime.now().strftime("%Y-%m-%d")
    # counterfactual explanation subgraph path
    counterfactual_explanation_subgraph_path = base_path + f'/results/{time_name}/counterfactual_subgraph/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{attack_budget_list}'
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
    else:
        adj, features, labels = None, None, None
        idx_train, idx_val, idx_test = None, None, None

    ######################### Loading GCN model  #########################
    model_save_path = f'{base_path}/model_save/{test_model}/{dataset_name}/{gcn_layer}-layer/'
    file_path = os.path.join(model_save_path, 'gcn_model.pth')
    gnn_model = load_GCN_model(file_path, features, labels, nhid, dropout, device, lr, weight_decay,
                               with_bias, gcn_layer)
    dense_adj = torch.tensor(adj.toarray())
    norm_adj = normalize_adj(dense_adj)
    pre_output = gnn_model.forward(torch.tensor(features.toarray()), norm_adj)

    # Create PyG Data object
    pyg_data = dr_data_to_pyg_data(adj, features, labels)

    ######################### select test nodes  #########################
    target_node_list, target_node_list1 = select_test_nodes(attack_type, explanation_type, idx_test, pre_output, labels)
    target_node_list = target_node_list + target_node_list1
    target_node_list = target_node_list[100:101]

    ######################### GNN explainer generate  #########################
    # Get CF examples in test set
    start = time.time()
    test_cf_examples = []
    cfexp_subgraph = {}
    for target_node in tqdm(target_node_list):
        cf_example = generate_ac_explanation(target_node, data, pyg_data, gnn_model, pre_output, gcn_layer,
                                                       attack_method, top_t, device, nhid, dropout, with_bias)
        print(cf_example)
        print("Time for {} epochs of one example: {:.4f}min".format(200, (time.time() - start) / 60))
    #     cfexp_subgraph[target_node] = None
    #     test_cf_examples.append(cf_example)
    # print("Total time elapsed: {:.4f}s".format((time.time() - start) / 60))
    # print("Number of CF examples found: {}/{}".format(len(test_cf_examples), len(target_node_list)))
    #
    # with open(counterfactual_explanation_subgraph_path + "/cfexp_subgraph.pickle", "wb") as fw:
    #     pickle.dump(cfexp_subgraph, fw)
    #
    # # Save CF examples in test set
    # with safe_open(
    #         "../results/{}/{}/{}_cf_examples_gcnlayer{}_lr{}_beta{}_mom{}_epochs{}_seed{}".format(DATA_NAME,
    #                                                                                               OPTIMIZER,
    #                                                                                               DATA_NAME,
    #                                                                                               GCN_LAYER,
    #                                                                                               LEARNING_RATE,
    #                                                                                               BETA,
    #                                                                                               N_Momentum,
    #                                                                                               NUM_EPOCHS,
    #                                                                                               SEED_NUM), "wb") as f:
    #     pickle.dump(test_cf_examples, f)
