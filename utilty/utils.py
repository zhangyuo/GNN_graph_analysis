import io
import os
import errno
import pickle

import torch
import numpy as np
import pandas as pd
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph
from deeprobust.graph.utils import classification_margin
from deeprobust.graph.data import Dataset
import scipy.sparse as sp
from torch_geometric.utils import to_undirected
import torch.nn.functional as F


def select_test_nodes(dataset_name, attack_type, idx_test, ori_output, labels):
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
    node_list1 = []
    sample_num = {
        "cora": 10,
        "BA-SHAPES": 10,
        "TREE-CYCLES": 20,
        "Loan-Decision": 20,
        "ogbn-arxiv": 1
    }[dataset_name]
    if attack_type is None:
        pass
    else:
        margin_dict_correct = {}
        margin_dict_incorrect = {}
        for num, idx in enumerate(idx_test):
            if dataset_name == "ogbn-arxiv":
                margin = classification_margin(ori_output[num], labels[idx])
            else:
                margin = classification_margin(ori_output[idx], labels[idx])
            if margin < 0:  # only keep the nodes correctly classified
                margin_dict_incorrect[idx] = margin
            else:
                margin_dict_correct[idx] = margin

        sorted_margins = sorted(margin_dict_correct.items(), key=lambda x: x[1], reverse=True)
        high = []
        low = []
        other = []
        # sample_num = 1

        for class_num in set(labels):
            class_num_sorted_margins = [x for x, y in sorted_margins if labels[x] == class_num]
            high += [x for x in class_num_sorted_margins[: sample_num]]
            low += [x for x in class_num_sorted_margins[-sample_num:]]
            other_0 = [x for x in class_num_sorted_margins[sample_num: -sample_num]]
            if len(other_0) > 0:
                try:
                    other += np.random.choice(other_0, 2 * sample_num, replace=False).tolist()
                except:
                    other += np.random.choice(other_0, 2 * sample_num, replace=True).tolist()
            else:
                other += np.random.choice(other_0, len(other_0), replace=False).tolist()
            # if len(other_0) > 20:
            #     other += np.random.choice(other_0, sample_num, replace=True).tolist()
            # elif len(other_0) > 0:
            #     other += np.random.choice(other_0, len(other_0), replace=True).tolist()
            # else:
            #     print(f"Warning: Classification other_0 number of class {class_num} is empty，Skip sampling")

        node_list += high + low + other
        node_list = [int(x) for x in node_list]
        node_list = list(set(node_list))

        sorted_margins = sorted(margin_dict_incorrect.items(), key=lambda x: x[1], reverse=True)
        high = []
        low = []
        other = []
        # sample_num = 1
        for class_num in set(labels):
            class_num_sorted_margins = [x for x, y in sorted_margins if labels[x] == class_num]
            high += [x for x in class_num_sorted_margins[: sample_num]]
            low += [x for x in class_num_sorted_margins[-sample_num:]]
            other_0 = [x for x in class_num_sorted_margins[sample_num: -sample_num]]
            if len(other_0) > 0:
                try:
                    other += np.random.choice(other_0, 2 * sample_num, replace=False).tolist()
                except:
                    other += np.random.choice(other_0, 2 * sample_num, replace=True).tolist()
            else:
                other += np.random.choice(other_0, len(other_0), replace=False).tolist()
            # if len(other_0) > 20:
            #     other += np.random.choice(other_0, sample_num, replace=True).tolist()
            # elif len(other_0) > 0:
            #     other += np.random.choice(other_0, len(other_0), replace=True).tolist()
            # else:
            #     print(f"Warning: Misclassification other_0 number of class {class_num} is empty，Skip sampling")

        node_list1 += high + low + other
        node_list1 = [int(x) for x in node_list1]
        node_list1 = list(set(node_list1))

    return node_list, node_list1


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def safe_open(path, w):
    ''' Open "path" for writing, creating any parent directories as needed.'''
    mkdir_p(os.path.dirname(path))
    return open(path, w)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_degree_matrix(adj):
    return torch.diag(sum(adj))


def normalize_adj(adj):
    # Normalize adjacancy matrix according to reparam trick in GCN paper
    A_tilde = adj + torch.eye(adj.shape[0])
    D_tilde = get_degree_matrix(A_tilde)
    # Raise to power -1/2, set all infs to 0s
    D_tilde_exp = D_tilde ** (-1 / 2)
    D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

    # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
    norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
    return norm_adj


# def get_neighbourhood(node_idx, edge_index, n_hops, features, labels):
# 	edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index[0])     # Get all nodes involved
# 	edge_subset_relabel = subgraph(edge_subset[0], edge_index[0], relabel_nodes=True)       # Get relabelled subset of edges
# 	sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
# 	sub_feat = features[edge_subset[0], :]
# 	sub_labels = labels[edge_subset[0]]
# 	new_index = np.array([i for i in range(len(edge_subset[0]))])
# 	node_dict = dict(zip(edge_subset[0].numpy(), new_index))        # Maps orig labels to new
# 	# print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
# 	return sub_adj, sub_feat, sub_labels, node_dict

def get_neighbourhood(target_node, edge_index, features, labels, gcn_layer):
    # generate explanation for target node from specified explainer
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=target_node,
        num_hops=gcn_layer + 1,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=edge_index.max() + 1
    )

    sub_adj = to_dense_adj(sub_edge_index).squeeze()
    sub_feat = features[subset, :]
    sub_feat = torch.tensor(sub_feat.toarray(), dtype=torch.float)
    sub_labels = labels[subset]
    node_dict = {int(orig_id): idx for idx, orig_id in enumerate(subset.tolist())}
    # print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
    return sub_adj, sub_edge_index, sub_feat, sub_labels, node_dict


def create_symm_matrix_from_vec(vector, n_rows):
    matrix = torch.zeros(n_rows, n_rows)
    idx = torch.tril_indices(n_rows, n_rows)
    matrix[idx[0], idx[1]] = vector
    symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
    return symm_matrix


def create_vec_from_symm_matrix(matrix, P_vec_size):
    idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
    vector = matrix[idx[0], idx[1]]
    return vector


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def get_S_values(pickled_results, header):
    df_prep = []
    for example in pickled_results:
        if example != []:
            df_prep.append(example[0])
    return pd.DataFrame(df_prep, columns=header)


def redo_dataset_pgexplainer_format(dataset, train_idx, test_idx):
    dataset.data.train_mask = index_to_mask(train_idx, size=dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(test_idx[len(test_idx)], size=dataset.data.num_nodes)


def accuracy(pred, labels):
    """计算准确率"""
    pred_labels = torch.tensor(pred, dtype=int)
    labels = torch.tensor(labels, dtype=int)
    correct = (pred_labels == labels).float().sum()  # 统计正确预测数
    return correct / len(labels)  # 返回准确率


# def compute_deg_diff(orig_sub_adj, edited_sub_adj):
#     orig_degrees = torch.sum(orig_sub_adj, dim=1)
#     new_degrees = torch.sum(edited_sub_adj, dim=1)
#     deg_diff = torch.sum(
#         torch.abs(new_degrees - orig_degrees) / (1 + orig_degrees)
#     )
#     return deg_diff

def compute_deg_diff(orig_sub_adj, edited_sub_adj):
    orig_degrees = torch.sum(orig_sub_adj)
    new_degrees = torch.sum(edited_sub_adj)
    deg_diff = torch.sum(
        torch.abs(new_degrees - orig_degrees) / (1 + orig_degrees)
    )
    return deg_diff


def compute_motif_viol(orig_sub_adj, edited_sub_adj, tau_c):
    orig_cluster_coef = clustering_coefficient(orig_sub_adj)
    new_cluster_coef = clustering_coefficient(edited_sub_adj)
    motif_violation = torch.sum(
        torch.clamp(torch.abs(new_cluster_coef - orig_cluster_coef) - tau_c, min=0.0)
    )
    return motif_violation


def compute_feat_sim(target_node_feat, edited_node_feat):
    feat_sim = F.cosine_similarity(
        target_node_feat.unsqueeze(0),
        edited_node_feat.unsqueeze(0)
    )
    return feat_sim


def clustering_coefficient(adj_tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    使用 PyTorch 近似计算无向图的局部聚类系数（向量化实现）。
    注意：这是对传统聚类系数的一种近似，主要用于训练和损失计算。
    """
    # 计算每个节点的度
    degrees = torch.sum(adj_tensor, dim=1)

    # 计算 A²，其对角线元素是节点邻居之间存在的路径数（每条边被计算两次）
    A_squared = torch.mm(adj_tensor, adj_tensor)
    # 节点i的邻居之间实际存在的边数近似为 (A_squared[i, i] - degrees[i]) / 2.0
    # 减 degrees[i] 是因为邻接矩阵对角线（自环）也被计算在内，通常需要减去
    # 这里简化处理，直接使用 A_squared 的对角线
    triangles = torch.diag(A_squared) / 2.0  # 更精确的计算可能需要调整

    # 计算可能存在的最大边数 k*(k-1)/2
    max_possible_edges = degrees * (degrees - 1) / 2.0

    # 避免除以零：对于度小于2的节点，聚类系数设为0
    # clustering_coeffs = torch.zeros_like(degrees, dtype=torch.float32)
    # valid_mask = (degrees > 1)
    # clustering_coeffs[valid_mask] = triangles[valid_mask] / max_possible_edges[valid_mask]

    clustering_coeffs = triangles / (max_possible_edges + eps)
    clustering_coeffs = torch.where(degrees > 1, clustering_coeffs, torch.zeros_like(clustering_coeffs))

    return clustering_coeffs


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class BAShapesDataset(Dataset):
    def __init__(self, pyg_data, test_model=None):
        self.name = 'BA-SHAPES'
        self.num_nodes = pyg_data.num_nodes
        self.num_features = pyg_data.num_node_features

        # 提取关键数据组件
        self.adj = self.edge_index_to_adj(pyg_data.edge_index)
        self.features = efficient_tensor_to_csr(pyg_data.x)
        self.labels = pyg_data.y.numpy()

        # 创建训练/验证/测试掩码
        # if test_model == "GAT":
        #     transform = RandomNodeSplit(split="train_rest", num_val=140, num_test=200)
        #     data = transform(pyg_data)
        #     valid_nodes = np.arange(self.num_nodes)
        #     self.idx_train = valid_nodes[data.train_mask]
        #     self.idx_val  = valid_nodes[data.val_mask]
        #     self.idx_test = valid_nodes[data.test_mask]
        # else:
        self.idx_train = self._create_mask(0.1)
        self.idx_val = self._create_mask(0.1, exclude=self.idx_train)
        self.idx_test = self._create_mask(0.8, exclude=np.concatenate([self.idx_train, self.idx_val]))

    def edge_index_to_adj(self, edge_index):
        """将 PyG 的 edge_index 转换为邻接矩阵"""
        import scipy.sparse as sp
        row, col = edge_index
        adj = sp.coo_matrix((np.ones(row.shape[0], dtype=np.float32), (row, col)),
                            shape=(self.num_nodes, self.num_nodes))
        return adj.tocsr()

    def _create_mask(self, ratio, exclude=None):
        """创建数据分割掩码"""
        valid_nodes = np.arange(self.num_nodes)
        if exclude is not None:
            valid_nodes = np.setdiff1d(valid_nodes, exclude)
        return np.random.choice(valid_nodes, size=int(ratio * self.num_nodes), replace=False)


class TreeCyclesDataset(Dataset):
    def __init__(self, pyg_data):
        self.name = 'TREE-CYCLES'
        self.num_nodes = pyg_data.num_nodes
        self.num_features = pyg_data.num_node_features

        # 提取关键数据组件
        self.adj = self.edge_index_to_adj(pyg_data.edge_index)
        self.features = efficient_tensor_to_csr(pyg_data.x)
        self.labels = pyg_data.y.numpy()

        # 创建训练/验证/测试掩码
        self.idx_train = self._create_mask(0.2)
        self.idx_val = self._create_mask(0.1, exclude=self.idx_train)
        self.idx_test = self._create_mask(0.7, exclude=np.concatenate([self.idx_train, self.idx_val]))

    def edge_index_to_adj(self, edge_index):
        """将 PyG 的 edge_index 转换为邻接矩阵"""
        import scipy.sparse as sp
        row, col = edge_index
        adj = sp.coo_matrix((np.ones(row.shape[0], dtype=np.float32), (row, col)),
                            shape=(self.num_nodes, self.num_nodes))
        return adj.tocsr()

    def _create_mask(self, ratio, exclude=None):
        """创建数据分割掩码"""
        valid_nodes = np.arange(self.num_nodes)
        if exclude is not None:
            valid_nodes = np.setdiff1d(valid_nodes, exclude)
        return np.random.choice(valid_nodes, size=int(ratio * self.num_nodes), replace=False)


class LoanDecisionDataset(Dataset):
    def __init__(self, pyg_data):
        self.name = 'Loan-Decision'
        self.num_nodes = pyg_data.num_nodes
        self.num_features = pyg_data.num_node_features

        # 提取关键数据组件
        self.adj = self.edge_index_to_adj(pyg_data.edge_index)
        self.features = efficient_tensor_to_csr(pyg_data.x)
        self.labels = pyg_data.y.numpy()

        # 创建训练/验证/测试掩码
        self.idx_train = self._create_mask(0.2)
        self.idx_val = self._create_mask(0.1, exclude=self.idx_train)
        self.idx_test = self._create_mask(0.7, exclude=np.concatenate([self.idx_train, self.idx_val]))

    def edge_index_to_adj(self, edge_index):
        """将 PyG 的 edge_index 转换为邻接矩阵"""
        import scipy.sparse as sp
        row, col = edge_index
        adj = sp.coo_matrix((np.ones(row.shape[0], dtype=np.float32), (row, col)),
                            shape=(self.num_nodes, self.num_nodes))
        return adj.tocsr()

    def _create_mask(self, ratio, exclude=None):
        """创建数据分割掩码"""
        valid_nodes = np.arange(self.num_nodes)
        if exclude is not None:
            valid_nodes = np.setdiff1d(valid_nodes, exclude)
        return np.random.choice(valid_nodes, size=int(ratio * self.num_nodes), replace=False)


class OGBNArxivDataset(Dataset):
    def __init__(self, ogbn_arxiv_data):
        self.pyg_data = ogbn_arxiv_data[0]
        self.name = 'ogbn-arxiv'
        self.num_nodes = self.pyg_data.num_nodes
        self.num_features = self.pyg_data.num_node_features

        # 提取关键数据组件
        edge_set = set((u.item(), v.item()) for u, v in self.pyg_data.edge_index.t())
        is_symmetric = all((v, u) in edge_set for (u, v) in edge_set)
        print(f"Edge index is symmetric: {is_symmetric}")
        if not is_symmetric:
            # self.pyg_data.orgi_edge_index = self.pyg_data.edge_index
            self.pyg_data.edge_index = to_undirected(self.pyg_data.edge_index)

        self.adj = self.edge_index_to_adj(self.pyg_data.edge_index)
        # self.orgi_adj = self.edge_index_to_adj(self.pyg_data.orgi_edge_index)
        self.features = efficient_tensor_to_csr(self.pyg_data.x)
        self.labels = self.pyg_data.y.view(-1).long().numpy()

        # 创建训练0.54-90941/验证0.18-29799/测试掩码0.28-48302
        split_idx = ogbn_arxiv_data.get_idx_split()
        self.idx_train = split_idx["train"]
        self.idx_val = split_idx["valid"]
        self.idx_test = split_idx["test"]

        # node year
        self.node_years = self.pyg_data.node_year.numpy().flatten()

    def edge_index_to_adj(self, edge_index):
        """将 PyG 的 edge_index 转换为邻接矩阵"""
        import scipy.sparse as sp
        row, col = edge_index
        adj = sp.coo_matrix((np.ones(row.shape[0], dtype=np.float32), (row, col)),
                            shape=(self.num_nodes, self.num_nodes))
        return adj.tocsr()

    def _create_mask(self, ratio, exclude=None):
        """创建数据分割掩码"""
        valid_nodes = np.arange(self.num_nodes)
        if exclude is not None:
            valid_nodes = np.setdiff1d(valid_nodes, exclude)
        return np.random.choice(valid_nodes, size=int(ratio * self.num_nodes), replace=False)

    def get_pyg_data(self):
        return self.pyg_data


def efficient_tensor_to_csr(features):
    # 获取Tensor数据
    features_np = features.detach().cpu().numpy()

    # 直接创建CSR矩阵
    return sp.csr_matrix(features_np)


import torch
import scipy.sparse as sp


# 1. adj: PyG edge_index -> scipy.sparse
def edge_index_to_adj(edge_index, num_nodes):
    # COO 格式
    row, col = edge_index
    data = torch.ones(row.size(0))
    adj = sp.coo_matrix(
        (data.numpy(), (row.numpy(), col.numpy())),
        shape=(num_nodes, num_nodes)
    )
    return adj.tocsr()


# 2. features: torch.Tensor -> scipy.sparse
def tensor_to_sparse(features):
    if torch.is_tensor(features):
        features = features.numpy()
    return sp.csr_matrix(features)


# 3. labels: torch.Tensor -> numpy
def tensor_to_numpy(labels):
    if torch.is_tensor(labels):
        labels = labels.numpy()
    return labels
