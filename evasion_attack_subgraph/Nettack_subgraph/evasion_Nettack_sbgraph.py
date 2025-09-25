#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/19 11:38
# @Author   : **
# @Email    : **@**
# @File     : evasion_GOttack.py
# @Software : PyCharm
# @Desc     :
"""
import os
import pickle
import random
import sys
res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(
    os.path.dirname(os.path.dirname(res)))  # acquire the parent path of current file's parent path
sys.path.insert(0, base_path)
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import dense_to_sparse, to_undirected, k_hop_subgraph
from counterfactual_explanation_subgraph.ACExplainer_subgraph.acexplainer_subgraph import evaluate_test_data
from model.GAT import load_GATNet_model
from model.GraphConv import load_GraphConv_model
from model.GraphTransformer import load_GraphTransforer_model


import time
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from deeprobust.graph.targeted_attack import Nettack
from tqdm import tqdm
from attack.GOttack.OrbitAttack import OrbitAttack
from model.GCN import dr_data_to_pyg_data, load_GCN_model
from utilty.attack_visualization import visualize_restricted_attack_subgraph
from attack.GOttack.orbit_table_generator import OrbitTableGenerator
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import classification_margin
from config.config import *
from utilty.utils import CPU_Unpickler, BAShapesDataset, TreeCyclesDataset, LoanDecisionDataset, normalize_adj, \
    select_test_nodes, OGBNArxivDataset, compute_deg_diff, compute_motif_viol, edge_index_to_adj, tensor_to_sparse, \
    tensor_to_numpy
from evasion_attack_subgraph.GOttack_subgraph.evasion_GOttack import set_up_surrogate_model
import torch.nn.functional as F

warnings.filterwarnings("ignore")

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
    attack_method = "Nettack"
    attack_budget_list = ATTACK_BUDGET_LIST
    top_t = ATTACK_BUDGET_LIST[0]
    heads_num = HEADS_NUM if TEST_MODEL in ["GraphTransformer", "GAT"] else None
    tau_c = TAU_C

    np.random.seed(SEED_NUM)
    torch.manual_seed(SEED_NUM)

    time_name = datetime.now().strftime("%Y-%m-%d")
    # counterfactual explanation subgraph path
    counterfactual_explanation_subgraph_path = base_path + f'/results/{time_name}/attack_subgraph_{test_model}/{attack_type}_{attack_method}_{dataset_name}_budget{attack_budget_list}-{SEED_NUM}'
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

    if gcn_layer != 2:
        surrogate = set_up_surrogate_model(features, adj, labels, idx_train, idx_val, device=device)  # 代理损失:gnn model
        # surrogate = gnn_model
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
    if dataset_name == "ogbn-arxiv":
        idx_test = target_node_id
    target_node_list, target_node_list1 = select_test_nodes(dataset_name, attack_type, idx_test, pre_output, labels)
    target_node_list = target_node_list + target_node_list1
    target_node_list.sort()
    print(f"Test nodes number: {len(target_node_list)}, incorrect: {len(target_node_list1)}")
    # target_node_list = target_node_list[110:130]

    ######################### attack subgraph generate  #########################
    start_0 = time.time()
    time_list = []
    test_cf_examples = []
    mis_cases = 0
    for num_idx, target_node in enumerate(tqdm(target_node_list)):
        if num_idx == 15:
            pass
        print(num_idx)
        start_time = time.time()

        # construct new subgraph: l+1 hop and attack nodes
        new_idx_map_tgt_node = None
        if dataset_name == "ogbn-arxiv":
            # 1. 提取目标节点的l-hop子图
            node_index, edge_index, mapping, _ = k_hop_subgraph(
                node_idx=target_node,
                num_hops=gcn_layer + 1,  # 覆盖GCN的感受野
                edge_index=pyg_data.edge_index,
                relabel_nodes=True,
                num_nodes=pyg_data.edge_index.max() + 1
            )

            # random sampling other nodes
            num_global_nodes = int(pyg_data.num_nodes) if hasattr(pyg_data, 'num_nodes') else int(pyg_data.x.size(0))
            all_nodes = torch.arange(num_global_nodes, device=node_index.device)

            mask = all_nodes != target_node
            candidate_nodes = all_nodes[mask]

            sample_size = min(2000, candidate_nodes.size(0))
            random_nodes = candidate_nodes[torch.randperm(candidate_nodes.size(0))[:sample_size]]

            # because of big graph, sampling part of orbit nodes
            new_nodes = torch.cat([node_index, random_nodes], dim=0)
            new_nodes = torch.unique(new_nodes, sorted=False)
            # 记录目标节点在新子图中的索引（用于后续 reference）
            if isinstance(target_node, torch.Tensor):
                target_node_val = int(target_node.item())
            else:
                target_node_val = int(target_node)
            tgt_pos_mask = (new_nodes == target_node_val).nonzero(as_tuple=True)[0]
            tgt_node_map_new_idx = int(tgt_pos_mask.item()) if tgt_pos_mask.numel() > 0 else None

            # 为了重标号，构建 full->new mapping array（vectorized，避免 Python 循环）
            mapping_array = torch.full((num_global_nodes,), -1, dtype=torch.long, device=new_nodes.device)
            mapping_array[new_nodes] = torch.arange(new_nodes.size(0), dtype=torch.long, device=new_nodes.device)

            # 过滤全图 edge_index：只保留两端都在 new_nodes 的边
            global_edge_index = pyg_data.edge_index.to(new_nodes.device)  # [2, E]

            # 建立一个布尔掩码，标记哪些节点在 new_nodes 中
            node_mask = torch.zeros(num_global_nodes, dtype=torch.bool, device=new_nodes.device)
            node_mask[new_nodes] = True

            # 过滤边：要求两端节点都在 new_nodes 中
            mask = node_mask[global_edge_index[0]] & node_mask[global_edge_index[1]]
            sub_edge_index = global_edge_index[:, mask]

            # 重标号为 [0 .. n_sub-1]
            sub_edge_index = mapping_array[sub_edge_index]  # shape [2, E_sub]
            # 可选：如果你想保证无自环/无重复，可进行 remove_self_loops/ coalesce（视需要）

            # 生成稠密邻接
            n_sub = new_nodes.size(0)
            data.adj = edge_index_to_adj(sub_edge_index, n_sub)

            # data.features: features 对应 new_nodes（转换为 scipy csr）
            data.features = tensor_to_sparse(pyg_data.x[new_nodes.cpu()])

            # data.labels: numpy array 对应 new_nodes
            data.labels = tensor_to_numpy(pyg_data.y[new_nodes.cpu()])

            # 记录新的 mapping（可选），方便从 new-subgraph-index 映射回全图 node id
            # new_idx_map_tgt_node: 如果你要从新索引找全局 id
            new_idx_map_tgt_node = {int(i): int(new_nodes[i].item()) for i in range(n_sub)}
            target_node = tgt_node_map_new_idx

            node_index_old = node_index.tolist()
            node_index_tensor = torch.tensor(node_index_old, dtype=torch.long, device=mapping_array.device)

            # 映射为新子图 ID
            node_index = mapping_array[node_index_tensor].cpu()

        attack_model = Nettack(surrogate, nnodes=data.adj.shape[0], attack_structure=True, attack_features=False)  # initialize the attack model
        attack_model = attack_model.to(device)
        attack_model.attack(data.features, data.adj, data.labels, target_node,ll_cutoff=0.1, n_perturbations=top_t)

        edited_edges = attack_model.structure_perturbations
        cf_adj = attack_model.modified_adj
        cf_adj = torch.tensor(cf_adj.toarray())
        extended_adj = data.adj
        extended_adj = torch.tensor(extended_adj.toarray())
        added_edges = []
        removed_edges = []

        for index, (u, v) in enumerate(edited_edges):
            if extended_adj[u,v].item() == 0.0:
                added_edges.append(edited_edges[index])
            else:
                removed_edges.append(edited_edges[index])

        norm_adj = normalize_adj(cf_adj)
        if test_model == "GCN":
            if dataset_name == "ogbn-arxiv":
                feat_dense = data.features.toarray()
                feat_tensor = torch.tensor(feat_dense, dtype=torch.float32, device=device)
                y_new_output = gnn_model.forward(feat_tensor, norm_adj)
            else:
                y_new_output = gnn_model.forward(pyg_data.x, norm_adj)
        else:
            edge_index, edge_weight = dense_to_sparse(norm_adj)
            y_new_output = gnn_model.forward(pyg_data.x, edge_index, edge_weight=edge_weight)

        if dataset_name == "ogbn-arxiv":
            target_node_label = pre_output[idx_test.index(target_node_val)].argmax().item()
        else:
            target_node_label = pre_output[target_node].argmax().item()

        new_idx_label = y_new_output[target_node].argmax().item()

        L_plau = α2 * compute_deg_diff(extended_adj, cf_adj) + α3 * compute_motif_viol(extended_adj, cf_adj, tau_c)

        if new_idx_label != target_node_label:
            print("find counterfactual explanation")
            cf_example = {
                "success": True,
                "target_node": target_node,
                "new_idx": target_node,
                "added_edges": added_edges,
                "removed_edges": removed_edges,
                "explanation_size": len(edited_edges),
                "original_pred": target_node_label,
                "new_pred": new_idx_label,
                # "extended_adj": extended_adj,
                "cf_adj": cf_adj if dataset_name!='ogbn-arxiv' else None,
                # "extended_feat": pyg_data.x,
                # "sub_labels": pyg_data.y
                "y_new_output": y_new_output[target_node],
                "new_idx_map_tgt_node": new_idx_map_tgt_node,
                "L_plau": L_plau
            }
        else:
            print("Don't find counterfactual explanation")
            cf_example = {
                "success": False,
                "target_node": target_node,
                "new_idx": target_node,
                "added_edges": added_edges,
                "removed_edges": removed_edges,
                "explanation_size": len(edited_edges),
                "original_pred": target_node_label,
                "new_pred": new_idx_label,
                # "extended_adj": extended_adj,
                "cf_adj": cf_adj if dataset_name!='ogbn-arxiv' else None,
                # "extended_feat": pyg_data.x,
                # "sub_labels": pyg_data.y
                "y_new_output": y_new_output[target_node],
                "new_idx_map_tgt_node": new_idx_map_tgt_node,
                "L_plau": L_plau
            }
        time_cost = time.time() - start_time

        print("Time for one example: {:.4f}s".format(time_cost))
        time_list.append(time_cost)
        test_cf_examples.append({"data": cf_example, "time_cost": time_cost})
        if cf_example['success']:
            mis_cases += 1

    print("Total time elapsed: {:.4f}min".format((time.time() - start_0) / 60))
    print("Number of CF examples found: {}/{}".format(mis_cases, len(target_node_list)))

    # Save results
    with open(
            counterfactual_explanation_subgraph_path + f"/{DATA_NAME}_cf_examples_gcnlayer{GCN_LAYER}_lr{LEARNING_RATE}_seed{SEED_NUM}",
            "wb") as f:
        pickle.dump(test_cf_examples, f)
