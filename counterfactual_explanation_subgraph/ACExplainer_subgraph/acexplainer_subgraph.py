#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/8/27 20:19
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : acexplainer_subgraph.py
# @Software : PyCharm
# @Desc     :
"""
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(
    os.path.dirname(os.path.dirname(res)))  # acquire the parent path of current file's parent path
sys.path.insert(0, base_path)
from model.GAT import load_GATNet_model
from model.GraphConv import load_GraphConv_model
import time
from datetime import datetime

import torch
import numpy as np
from deeprobust.graph.data import Dataset
import networkx as nx
from torch_geometric.utils import k_hop_subgraph, to_dense_adj, dense_to_sparse
from tqdm import tqdm

from attack.GOttack.OrbitAttack import OrbitAttack
from attack.GOttack.orbit_table_generator import OrbitTableGenerator
from config.config import *
from explainer.ac_explanation.ac_explainer import ACExplainer
from model.GCN import GCN_model, dr_data_to_pyg_data, GCNtoPYG, load_GCN_model
from utilty.cfexplanation_visualization import visualize_cfexp_subgraph
from utilty.utils import safe_open, get_neighbourhood, normalize_adj, select_test_nodes, CPU_Unpickler, BAShapesDataset, \
    TreeCyclesDataset, LoanDecisionDataset
import torch.nn.functional as F
from evasion_attack_subgraph.GOttack_subgraph.evasion_GOttack import set_up_surrogate_model
from model.GraphTransformer import load_GraphTransforer_model
from deeprobust.graph.defense import GCN


def generate_acexplainer_subgraph(df_orbit,
                                  target_node: int,
                                  data,
                                  pyg_data,
                                  gnn_model,
                                  surrogate,
                                  output,
                                  gcn_layer: int = 2,
                                  attack_method: str = "GOttack",
                                  top_t: int = 10,
                                  device: str = "cuda",
                                  nhid: int = 16,
                                  dropout: float = 0.5,
                                  with_bias: bool = True,
                                  test_model: str = "GCN",
                                  heads: int = 2,
                                  dataset_name: str = "cora"):
    """
    生成AC-Explainer解释
    """
    start = time.time()
    # 1. 提取目标节点的l-hop子图
    node_index, edge_index, mapping, _ = k_hop_subgraph(
        node_idx=target_node,
        num_hops=gcn_layer + 1,  # 覆盖GCN的感受野
        edge_index=pyg_data.edge_index,
        relabel_nodes=True,
        num_nodes=pyg_data.edge_index.max() + 1
    )

    # 2. 获取攻击节点并映射到原始图索引
    attack_model = OrbitAttack(surrogate, df_orbit, nnodes=data.adj.shape[0],
                               device=device, top_t=top_t, gcn_layer=gcn_layer)  # initialize the attack model
    attack_nodes = get_attack_nodes(attack_model, df_orbit, target_node, data, pyg_data, attack_method, top_t)

    node_index = node_index.tolist()

    # 3. 构建扩展节点集合 (l-hop节点 + 攻击节点)
    extended_nodes = list(set(node_index + attack_nodes))
    extended_nodes.sort()

    # 4. 高效创建扩展子图的邻接矩阵和原始掩码矩阵
    # extended_adj, original_adj_mask = networkx_create_extended_adj(extended_nodes, pyg_data.edge_index)
    adj_dense = data.adj.toarray()
    extended_adj = adj_dense[np.ix_(extended_nodes, extended_nodes)]
    extended_adj = torch.tensor(extended_adj, dtype=torch.float, requires_grad=True)

    # 5. 提取扩展子图的特征
    extended_feat = pyg_data.x[extended_nodes]

    # 6. 找到目标节点在扩展子图中的新索引
    target_node_idx = extended_nodes.index(target_node)
    node_dict = {int(orig_id): idx for idx, orig_id in enumerate(extended_nodes)}
    node_index_1, _, _, _ = k_hop_subgraph(
        node_idx=target_node,
        num_hops=1,
        edge_index=pyg_data.edge_index,
        relabel_nodes=True,
        num_nodes=pyg_data.edge_index.max() + 1
    )
    node_index_1 = node_index_1.tolist()
    node_num_l_hop = [node_index_1, attack_nodes, node_dict]

    # test model log-probability output is same to original prediction output
    # print("Output original model, full adj: {}".format(output[target_node]))
    # norm_sub_adj = normalize_adj(extended_adj)
    # print("Output original model, sub adj: {}".format(gnn_model.forward(extended_feat, norm_sub_adj)[target_node_idx]))

    # 7. 创建解释器
    explainer = ACExplainer(
        model=gnn_model,
        target_node=target_node,
        node_idx=target_node_idx,
        node_num_l_hop=node_num_l_hop,
        extended_sub_adj=extended_adj,
        sub_feat=extended_feat,
        sub_labels=data.labels[extended_nodes],
        y_pred_orig=output.argmax(dim=1)[target_node],
        nclass=data.labels.max().item() + 1,
        nhid=nhid,
        dropout=dropout,
        lambda_pred=LAMBDA_PRED,
        lambda_dist=LAMBDA_DIST,
        lambda_plau=LAMBDA_PLAU,
        epoch=NUM_EPOCHS_AC,
        optimizer=OPTIMIZER_AC,
        n_momentum=N_Momentum_AC,
        lr=LEARNING_RATE_AC,
        top_k=MAX_EDITS,
        tau_plus=TAU_PLUS,
        tau_minus=TAU_MINUS,
        α1=α1,
        α2=α2,
        α3=α3,
        α4=α4,
        tau_c=TAU_C,
        device=device,
        gcn_layer=gcn_layer,
        with_bias=with_bias,
        test_model=test_model,
        heads=heads,
        dataset_name=dataset_name
    )

    # 8. 训练解释器
    result = explainer.explain()

    time_cost = time.time() - start

    # 9. 映射回原始图索引
    added_edges = []
    removed_edges = []

    delta_A = result["delta_A"]
    for i in range(delta_A.size(0)):
        for j in range(i + 1, delta_A.size(1)):
            if delta_A[i, j] > TAU_PLUS and extended_adj[i, j] == 0:  # 添加的边
                orig_i = extended_nodes[i]
                orig_j = extended_nodes[j]
                added_edges.append((orig_i, orig_j))
            elif delta_A[i, j] < TAU_MINUS and extended_adj[i, j] == 1:  # 删除的边
                orig_i = extended_nodes[i]
                orig_j = extended_nodes[j]
                removed_edges.append((orig_i, orig_j))

    # 10. generate subgraph
    subgraph = {
        "subgraph": None,
        "true_subgraph": None,
        "E_type": None,
    }
    sub_labels = data.labels[extended_nodes]
    if result['success']:
        modified_sub_adj = result["cf_adj"].detach().numpy()
        changed_label = result["new_pred"].item()
        subgraph, true_subgraph, E_type = visualize_cfexp_subgraph(
            modified_sub_adj,
            extended_adj.detach().numpy(),
            data.labels,
            sub_labels,
            extended_feat.numpy(),
            changed_label,
            target_node_idx,
            cfexp_name='ACExplanation',
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

    return {
        "success": result['success'],
        "target_node": target_node,
        "new_idx": target_node_idx,
        "added_edges": added_edges,
        "removed_edges": removed_edges,
        "explanation_size": len(added_edges) + len(removed_edges),
        "plau_loss": result["plau_loss"],
        "original_pred": result["original_pred"],
        "new_pred": result["new_pred"],
        "extended_adj": extended_adj,
        "cf_adj": result["cf_adj"],
        "extended_feat": extended_feat,
        "sub_labels": sub_labels
    }, time_cost, subgraph


def get_attack_nodes(attack_model, df_orbit, target_node, data, pyg_data, method="GOttack", top_t=10):
    """获取攻击节点列表"""
    if method == "GOttack":
        # 实现GOttack攻击方法，返回高影响力节点
        # 1. Feature Similarity > 0.1
        matching_index = df_orbit.index[df_orbit['two_Orbit_type'] == '1518'].tolist()
        if len(matching_index) < top_t:
            matching_index += df_orbit.index[df_orbit['two_Orbit_type'] == '1519'].tolist()
        similarities = []
        for i in matching_index:
            if i != target_node:
                sim = F.cosine_similarity(
                    pyg_data.x[target_node].unsqueeze(0),
                    pyg_data.x[i].unsqueeze(0)
                )
                if sim > 0.1:
                    similarities.append((i, sim.item()))

        similarities.sort(key=lambda x: x[1], reverse=True)
        high_sim_node = [node for node, sim in similarities]

        # Domain Rules
        pass

        # High Impact Prioritization
        attack_model.attack_cf(data.features, data.adj, data.labels, target_node, top_t, high_sim_node=high_sim_node)
        best_edges = attack_model.best_edge_list
        print("best edges: ", len(best_edges))

        attck_nodes = [node[1] for node in best_edges]
        attck_nodes = list(set(attck_nodes))

        return attck_nodes

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
    explainer_method = "ACExplainer"
    top_t = MAX_ATTACK_NODES_NUM
    heads_num = HEADS_NUM if TEST_MODEL in ["GraphTransformer", "GAT"] else None

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
    elif test_model == 'GAT':
        file_path = os.path.join(model_save_path, 'gat_model.pth')
        gnn_model = load_GATNet_model(file_path, data, nhid, dropout, device, lr, weight_decay, gcn_layer, heads_num)
        dense_adj = torch.tensor(adj.toarray())
        norm_adj = normalize_adj(dense_adj)
        edge_index, edge_weight = dense_to_sparse(norm_adj)
        pre_output = gnn_model.forward(torch.tensor(features.toarray()), edge_index, edge_weight=edge_weight)

    if gcn_layer != 2:
        # surrogate = set_up_surrogate_model(features, adj, labels, idx_train, idx_val, device=device)  # 代理损失:gnn model
        surrogate = gnn_model
    else:
        surrogate = gnn_model

    if test_model != "GCN":
        file_path = base_path + f"/model_save/surrogate/{dataset_name}/"
        if os.path.exists(file_path):
            surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16, dropout=0,
                            with_relu=False, with_bias=False, device=device)
            surrogate.load_state_dict(torch.load(file_path + 'surrogate.model', map_location=device))
            surrogate.to(device)
        else:
            os.makedirs(file_path)
            surrogate = set_up_surrogate_model(features, adj, labels, idx_train, idx_val, device=device)
            torch.save(surrogate.state_dict(), file_path + 'surrogate.model')

    ######################### select test nodes  #########################
    target_node_list, target_node_list1 = select_test_nodes(dataset_name, attack_type, idx_test, pre_output, labels)
    target_node_list = target_node_list + target_node_list1
    target_node_list.sort()
    print(f"Test nodes number: {len(target_node_list)}, incorrect: {len(target_node_list1)}")
    # target_node_list = target_node_list[10:20]

    ######################### GNN explainer generate  #########################
    df_orbit = OrbitTableGenerator(dataset_name).generate_orbit_table()
    # Get CF examples in test set
    start_0 = time.time()
    test_cf_examples = []
    cfexp_subgraph = {}
    time_list = []
    mis_cases = 0
    for target_node in tqdm(target_node_list):
        cf_example, time_cost, subgraph = generate_acexplainer_subgraph(df_orbit, target_node, data, pyg_data,
                                                                        gnn_model,
                                                                        surrogate, pre_output, gcn_layer, attack_method,
                                                                        top_t,
                                                                        device, nhid, dropout, with_bias, test_model,
                                                                        heads_num, dataset_name)
        # print(cf_example)
        print("Time for {} epochs of one example: {:.4f}s".format(NUM_EPOCHS_AC, time_cost))
        time_list.append(time_cost)
        cfexp_subgraph[target_node] = subgraph if cf_example['success'] else None
        test_cf_examples.append({"data": cf_example, "time_cost": time_cost})
        if cf_example['success']:
            mis_cases += 1
    print("Total time elapsed: {:.4f}min".format((time.time() - start_0) / 60))
    print("Number of CF examples found: {}/{}".format(mis_cases, len(target_node_list)))

    with open(counterfactual_explanation_subgraph_path + "/cfexp_subgraph.pickle", "wb") as fw:
        pickle.dump(cfexp_subgraph, fw)

    # Save CF examples in test set
    with open(
            counterfactual_explanation_subgraph_path + f"/{DATA_NAME}_cf_examples_gcnlayer{GCN_LAYER}_lr{LEARNING_RATE}_epochs{NUM_EPOCHS_AC}_seed{SEED_NUM}",
            "wb") as f:
        pickle.dump(test_cf_examples, f)
