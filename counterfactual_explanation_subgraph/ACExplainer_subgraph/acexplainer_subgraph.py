#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/8/27 20:19
# @Author   : **
# @Email    : **@**
# @File     : acexplainer_subgraph.py
# @Software : PyCharm
# @Desc     :
"""
import os
import pickle
import sys
import warnings

import pandas as pd
import scipy as sp
from torch_geometric.loader import NeighborSampler

warnings.filterwarnings("ignore")
res = os.path.abspath(__file__)  # acquire absolute path of current file
base_path = os.path.dirname(
    os.path.dirname(os.path.dirname(res)))  # acquire the parent path of current file's parent path
sys.path.insert(0, base_path)
from model.GAT import load_GATNet_model
from model.GraphConv import load_GraphConv_model
import time
from datetime import datetime
from ogb.nodeproppred import PygNodePropPredDataset
import torch
import numpy as np
from deeprobust.graph.data import Dataset
import networkx as nx
from torch_geometric.utils import k_hop_subgraph, to_dense_adj, dense_to_sparse, to_undirected
from tqdm import tqdm

from attack.GOttack.OrbitAttack import OrbitAttack
from attack.GOttack.orbit_table_generator import OrbitTableGenerator
from config.config import *
from explainer.ac_explanation.ac_explainer import ACExplainer
from model.GCN import GCN_model, dr_data_to_pyg_data, GCNtoPYG, load_GCN_model
from utilty.cfexplanation_visualization import visualize_cfexp_subgraph
from utilty.utils import safe_open, get_neighbourhood, normalize_adj, select_test_nodes, CPU_Unpickler, BAShapesDataset, \
    TreeCyclesDataset, LoanDecisionDataset, OGBNArxivDataset, edge_index_to_adj, tensor_to_sparse, tensor_to_numpy
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
                                  dataset_name: str = "cora",
                                  output_idx=None):
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

    # construct new subgraph: l+1 hop and attack nodes
    new_idx_map_tgt_node = None
    if dataset_name == "ogbn-arxiv":
        # because of big graph, sampling part of orbit nodes
        df_orbit = df_orbit.loc[df_orbit['two_Orbit_type'] == '1518']
        df_orbit = df_orbit.sample(n=1000, random_state=102)  # 固定随机种子，结果可复现
        # df_orbit = pd.DataFrame(df_orbit.to_numpy()[node_index], columns=df_orbit.columns)
        orbit_nodes_series = df_orbit['node_number']
        orbit_nodes = torch.tensor(orbit_nodes_series.astype(int).values, dtype=torch.long, device=node_index.device)
        new_nodes = torch.cat([node_index, orbit_nodes], dim=0)
        new_nodes = torch.unique(new_nodes, sorted=False)
        # 记录目标节点在新子图中的索引（用于后续 reference）
        if isinstance(target_node, torch.Tensor):
            target_node_val = int(target_node.item())
        else:
            target_node_val = int(target_node)
        tgt_pos_mask = (new_nodes == target_node_val).nonzero(as_tuple=True)[0]
        tgt_node_map_new_idx = int(tgt_pos_mask.item()) if tgt_pos_mask.numel() > 0 else None

        # 为了重标号，构建 full->new mapping array（vectorized，避免 Python 循环）
        num_global_nodes = int(pyg_data.num_nodes) if hasattr(pyg_data, 'num_nodes') else int(pyg_data.x.size(0))
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

        # 把 df_orbit 的 node_number 转成 tensor
        orbit_old_ids = torch.tensor(
            df_orbit['node_number'].astype(int).values,
            dtype=torch.long,
            device=mapping_array.device
        )
        # 用 mapping_array 查新 ID
        orbit_new_ids = mapping_array[orbit_old_ids].cpu().numpy()
        # 把 df_orbit 里的 node_number 更新为新的索引
        df_orbit = df_orbit.copy()
        df_orbit['node_number'] = orbit_new_ids
        df_orbit.index = orbit_new_ids

        node_index_old = node_index.tolist()
        node_index_tensor = torch.tensor(node_index_old, dtype=torch.long, device=mapping_array.device)

        # 映射为新子图 ID
        node_index = mapping_array[node_index_tensor].cpu()

    # 2. 获取攻击节点并映射到原始图索引
    attack_model = OrbitAttack(surrogate, df_orbit, nnodes=data.adj.shape[0],
                               device=device, top_t=top_t, gcn_layer=gcn_layer)  # initialize the attack model
    attack_nodes = get_attack_nodes(attack_model, df_orbit, target_node, data, attack_method, top_t)

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
    if dataset_name == "ogbn-arxiv":
        extended_global_nodes = [new_idx_map_tgt_node[i] for i in extended_nodes]
        extended_feat = pyg_data.x[extended_global_nodes]
    else:
        extended_feat = pyg_data.x[extended_nodes]

    # 6. 找到目标节点在扩展子图中的新索引
    target_node_idx = extended_nodes.index(target_node)
    node_dict = {int(orig_id): idx for idx, orig_id in enumerate(extended_nodes)}
    if dataset_name == "ogbn-arxiv":
        node_index_1, _, _, _ = k_hop_subgraph(
            node_idx=target_node,
            num_hops=1,
            edge_index=sub_edge_index,
            relabel_nodes=True,
            num_nodes=sub_edge_index.max() + 1
        )
    else:
        node_index_1, _, _, _ = k_hop_subgraph(
            node_idx=target_node,
            num_hops=1,
            edge_index=pyg_data.edge_index,
            relabel_nodes=True,
            num_nodes=pyg_data.edge_index.max() + 1
        )
    node_index_1 = node_index_1.tolist()
    node_num_l_hop = [node_index_1, attack_nodes, node_dict]

    # # test model log-probability output is same to original prediction output
    # if dataset_name == "ogbn-arxiv":
    #     print("Output original model, full adj: {}".format(output[output_idx.index(target_node_val)]))
    #     norm_sub_adj = normalize_adj(extended_adj)
    #     print("Output original model, sub adj: {}".format(
    #         gnn_model.forward(extended_feat, norm_sub_adj)[target_node_idx]))
    # else:
    #     print("Output original model, full adj: {}".format(output[target_node]))
    #     norm_sub_adj = normalize_adj(extended_adj)
    #     print("Output original model, sub adj: {}".format(gnn_model.forward(extended_feat, norm_sub_adj)[target_node_idx]))

    # 7. 创建解释器
    if dataset_name == "ogbn-arxiv":
        y_pred_orig = output.argmax(dim=1)[output_idx.index(target_node_val)]
    else:
        y_pred_orig = output.argmax(dim=1)[target_node]

    explainer = ACExplainer(
        model=gnn_model,
        target_node=target_node,
        node_idx=target_node_idx,
        node_num_l_hop=node_num_l_hop,
        extended_sub_adj=extended_adj,
        sub_feat=extended_feat,
        sub_labels=data.labels[extended_nodes],
        y_pred_orig=y_pred_orig,
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
        "sub_labels": sub_labels,
        "new_idx_map_tgt_node": new_idx_map_tgt_node
    }, time_cost, subgraph


def get_attack_nodes(attack_model, df_orbit, target_node, data, method="GOttack", top_t=10):
    """获取攻击节点列表"""
    if method == "GOttack":
        # 实现GOttack攻击方法，返回高影响力节点
        # 1. Feature Similarity > 0.1
        matching_index = df_orbit.index[df_orbit['two_Orbit_type'] == '1518'].tolist()
        if len(matching_index) < top_t:
            matching_index += df_orbit.index[df_orbit['two_Orbit_type'] == '1519'].tolist()

        similarities = []
        feat_target = torch.tensor(data.features[target_node].todense(), dtype=torch.float32).flatten()
        for i in matching_index:
            if i != target_node:
                feat_i = torch.tensor(data.features[i].todense(), dtype=torch.float32).flatten()
                sim = F.cosine_similarity(
                    feat_target.unsqueeze(0),  # Add batch dimension
                    feat_i.unsqueeze(0),  # Add batch dimension
                    dim=1
                )
                sim = sim.item()
                if sim > 0.1:
                    similarities.append((i, sim))

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


def sample_subgraph_edges_by_hop(edge_index, target_node, hop_neighbors={1: 10, 2: 5}):
    """
    使用 NeighborSampler 替代手写采样函数
    返回采样后的 sampled_edge_index 和 sampled_nodes
    edge_index: [2, num_edges]
    target_node: int，目标节点编号
    hop_neighbors: dict {hop层: 每层采样邻居数}
    """
    # 按 hop 顺序生成 sizes
    max_hop = max(hop_neighbors.keys())
    sizes = [hop_neighbors.get(h + 1, 0) for h in range(max_hop)]

    sampler = NeighborSampler(
        edge_index,
        sizes=sizes,
        batch_size=1,
        shuffle=False,
        num_nodes=edge_index.max().item() + 1
    )

    for batch_size, n_id, adjs in sampler:
        all_edges = []
        for edge_index_hop, _, _ in adjs:
            if edge_index_hop.numel() > 0:
                all_edges.append(edge_index_hop)
        if len(all_edges) > 0:
            sampled_edge_index = torch.cat(all_edges, dim=1)
            sampled_nodes = n_id
        else:
            sampled_edge_index = torch.empty((2, 0), dtype=torch.long)
            sampled_nodes = n_id
        break  # 只取目标节点 batch

    # 保证 sampled_nodes 是一维
    sampled_nodes = sampled_nodes.view(-1)

    # 保证目标节点在子图中
    target_node_tensor = torch.tensor([target_node], dtype=sampled_nodes.dtype, device=sampled_nodes.device)
    if (sampled_nodes == target_node_tensor).sum() == 0:
        sampled_nodes = torch.cat([sampled_nodes, target_node_tensor])

    return sampled_edge_index, sampled_nodes


def evaluate_test_data(gnn_model, data, pyg_data, gcn_layer):
    logits = []
    target_node_id = []
    num_count = 0
    for idx, node_id in enumerate(tqdm(data.idx_test)):
        # if idx > 2000:
        #     break
        if num_count > 2000:
            break
        target_node = node_id.item()
        node_index, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=target_node,
            num_hops=gcn_layer + 1,
            edge_index=pyg_data.edge_index,
            relabel_nodes=True,
            num_nodes=pyg_data.edge_index.max() + 1
        )
        if len(node_index) < 500:
            print(len(node_index))
            target_node_id.append(target_node)
            num_count += 1
        else:
            continue

        # sampled_edge_index, sampled_nodes = sample_subgraph_edges_by_hop(
        #     sub_edge_index, target_node, hop_neighbors={1: 10, 2: 5}
        # )

        # if sampled_edge_index is None or sampled_nodes is None:
        #     continue

        # 子图特征 & 标签
        x_sub = pyg_data.x[node_index].to(device)

        # 子图邻接矩阵 (dense)
        sub_adj = to_dense_adj(sub_edge_index, max_num_nodes=x_sub.size(0)).squeeze(0).to(device)
        norm_sub_adj = normalize_adj(sub_adj)

        # 前向传播
        out = gnn_model(x_sub, norm_sub_adj)

        # mapping 是原始 node_id 在子图中的位置
        # mapping = (sampled_nodes == target_node).nonzero(as_tuple=True)[0]
        logit = out[mapping]  # 单节点预测
        logits.append(logit.squeeze(0).cpu())

    output = torch.stack(logits, dim=0)  # [num_test_nodes, num_classes]
    return output, target_node_id


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
    counterfactual_explanation_subgraph_path = base_path + f'/results/{time_name}/counterfactual_subgraph_{test_model}/{attack_type}_{attack_method}_{explanation_type}_{explainer_method}_{dataset_name}_budget{[MAX_EDITS]}-{SEED_NUM}'
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
    if dataset_name == "ogbn-arxiv":
        idx_test = target_node_id
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
                                                                        heads_num, dataset_name,
                                                                        output_idx=idx_test)
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
