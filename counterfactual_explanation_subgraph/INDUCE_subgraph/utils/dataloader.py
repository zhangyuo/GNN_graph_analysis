# load data
import io

from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.utils import to_dense_adj, remove_self_loops, to_undirected
from torch_geometric.utils import add_self_loops
from torch_geometric.datasets import TUDataset
from scipy.sparse import csr_matrix, issparse
import networkx as nx
import pickle
from utils.utils import *
from deeprobust.graph.data import Dataset
from torch_geometric.data import Data
import torch
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset


def label_process(labels, adj):
    deg = get_degree_matrix(adj)
    for node in range(adj.shape[0]):
        neighbour = torch.nonzero(adj[node]).transpose(0, 1)
        nn_deg = [int(deg[nn][nn].item()) for nn in neighbour[0]]
        maxi = np.max(nn_deg)
        if maxi < 17:
            labels[node] = torch.tensor(0)
        else:
            labels[node] = torch.tensor(1)
    return labels


def efficient_tensor_to_csr(features):
    # 获取Tensor数据
    features_np = features.detach().cpu().numpy()

    # 直接创建CSR矩阵
    return sp.csr_matrix(features_np)


class ChameleonDataset(Dataset):
    def __init__(self, chameleon_data):
        self.pyg_data = chameleon_data[0]
        self.name = 'chameleon'
        self.num_nodes = self.pyg_data.num_nodes
        self.num_features = self.pyg_data.num_node_features

        # 提取关键数据组件
        edge_set = set((u.item(), v.item()) for u, v in self.pyg_data.edge_index.t())
        is_symmetric = all((v, u) in edge_set for (u, v) in edge_set)
        print(f"Edge index is symmetric: {is_symmetric}")
        if not is_symmetric:
            edge_index = self.pyg_data.edge_index
            edge_index, _ = remove_self_loops(edge_index)  # chameleon has self-loop edges (e.g., (u,u) or (v,v))
            self.pyg_data.edge_index = to_undirected(edge_index)

        self.adj = self.edge_index_to_adj(self.pyg_data.edge_index)
        # self.orgi_adj = self.edge_index_to_adj(self.pyg_data.orgi_edge_index)
        self.features = efficient_tensor_to_csr(self.pyg_data.x)
        self.labels = self.pyg_data.y.view(-1).long().numpy()

        # 创建训练/验证/测试掩码
        self.idx_train = self._create_mask(0.50)
        self.idx_val = self._create_mask(0.25, exclude=self.idx_train)
        self.idx_test = self._create_mask(0.25, exclude=np.concatenate([self.idx_train, self.idx_val]))

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


class DataLoader:
    def __init__(self, dataset, self_loops=False):
        self.data = dataset
        self.self_loops = self_loops

    def loadData(self):
        file = open('../data/gnn_explainer/{}.pickle'.format(self.data[:4]), 'rb')
        data = pickle.load(file)
        file.close()

        return data

    def loadCora(self):
        data = Dataset(root='../data/gnn_explainer/', name=self.data)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        # Create PyG Data object
        pyg_data = dr_data_to_pyg_data(adj, features, labels)

        _data = {
            "adj": adj,
            "feat": features,
            "labels": labels,
            "train_idx": idx_train,
            "test_idx": idx_test
        }
        return _data, adj, features, labels, pyg_data

    def loadChameleon(self):
        # Create PyG Data object
        data = WikipediaNetwork(name=self.data, root='../data/gnn_explainer/')
        pyg_data = data[0]
        pyg_data.y = pyg_data.y.view(-1).long()
        # Create deeprobust Data object
        data = ChameleonDataset(data)
        pyg_data.edge_index = data.pyg_data.edge_index
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

        _data = {
            "adj": adj,
            "feat": features,
            "labels": labels,
            "train_idx": idx_train,
            "test_idx": idx_test
        }
        return _data, adj, features, labels, pyg_data

    def loadBAShapes(self, test_model=''):
        # Create PyG Data object
        with open('../data/gnn_explainer/' + "BAShapes.pickle", "rb") as f:
            pyg_data = CPU_Unpickler(f).load()
            if test_model == "GAT":
                # because of no features of nodes
                pyg_data.x = F.one_hot(pyg_data.y).float()
        data = BAShapesDataset(pyg_data)
        # Create deeprobust Data object
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

        _data = {
            "adj": adj,
            "feat": features,
            "labels": labels,
            "train_idx": idx_train,
            "test_idx": idx_test
        }
        return _data, adj, features, labels, pyg_data

    def loadTreeCycles(self, test_model=''):
        # Create PyG Data object
        with open('../data/gnn_explainer/' + "/TreeCycle.pickle", "rb") as f:
            pyg_data = CPU_Unpickler(f).load()
            if test_model == "GAT":
                # because of no features of nodes
                pyg_data.x = F.one_hot(pyg_data.y).float()
        # Create deeprobust Data object
        data = TreeCyclesDataset(pyg_data)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

        _data = {
            "adj": adj,
            "feat": features,
            "labels": labels,
            "train_idx": idx_train,
            "test_idx": idx_test
        }
        return _data, adj, features, labels, pyg_data

    def loadLoanDecision(self):
        # Create PyG Data object
        with open('../data/gnn_explainer/' + "/LoanDecision.pickle", "rb") as f:
            pyg_data = CPU_Unpickler(f).load()
        # Create deeprobust Data object
        data = LoanDecisionDataset(pyg_data)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

        _data = {
            "adj": adj,
            "feat": features,
            "labels": labels,
            "train_idx": idx_train,
            "test_idx": idx_test,
            "data": data
        }
        return _data, adj, features, labels, pyg_data

    def loadOgbnArxiv(self):
        # Create PyG Data object
        ogbn_arxiv_data = PygNodePropPredDataset(name="ogbn-arxiv", root='../data/gnn_explainer/')
        pyg_data = ogbn_arxiv_data[0]
        pyg_data.edge_index = to_undirected(pyg_data.edge_index)
        pyg_data.y = pyg_data.y.view(-1).long()
        # Create deeprobust Data object
        data = OGBNArxivDataset(ogbn_arxiv_data)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

        _data = {
            "adj": adj,
            "feat": features,
            "labels": labels,
            "train_idx": idx_train,
            "test_idx": idx_test
        }
        return _data, adj, features, labels, pyg_data

    def preprocessData(self):
        pyg_data = None
        if self.data == "cora":
            data = self.loadCora()[0]
            pyg_data = self.loadCora()[4]
        elif self.data == "chameleon":
            data = self.loadChameleon()[0]
            pyg_data = self.loadChameleon()[4]
        elif self.data == "BA-SHAPES":
            data = self.loadBAShapes()[0]
            pyg_data = self.loadBAShapes()[4]
        elif self.data == "TREE-CYCLES":
            data = self.loadTreeCycles()[0]
            pyg_data = self.loadTreeCycles()[4]
        elif self.data == "Loan-Decision":
            data = self.loadLoanDecision()[0]
            pyg_data = self.loadLoanDecision()[4]
        elif self.data == "ogbn-arxiv":
            arxiv_data = self.loadOgbnArxiv()
            data = arxiv_data[0]
            pyg_data = arxiv_data[4]
        else:
            data = self.loadData()

        try:
            adj = torch.Tensor(data["adj"]).squeeze()  # Does not include self loops
        except:
            dense_array = data["adj"].toarray()
            adj = torch.from_numpy(dense_array).float()

        if (self.self_loops):  # add self loops
            adj.fill_diagonal_(1)

        try:
            features = torch.Tensor(data["feat"]).squeeze()
        except:
            features = pyg_data.x.squeeze()
        # print("from data: ", features.shape, features[675]) - all 1s - dim 10
        labels = torch.tensor(data["labels"], dtype=torch.long).squeeze()
        idx_train = torch.tensor(data["train_idx"])
        idx_test = torch.tensor(data["test_idx"])
        # returns tuple(edge_index, edge_attributes)
        if self.data == "ogbn-arxiv":
            edge_index = [pyg_data.edge_index]
        else:
            edge_index = dense_to_sparse(adj)  # [0]

        if self.data == "ogbn-arxiv":
            norm_adj = adj  # big dataset normalizes failed in cpu environment
        else:
            norm_adj = normalize_adj(adj)  # According to reparam trick from GCN paper

        g = Graph(adj, features, labels, idx_train, idx_test, edge_index, norm_adj)
        return g


def adj_to_edge_index(adj):
    """
    transfer adjacency matrix in deeprobust data to edge_index in pyg data
    :param adj:
    :return:
    """
    coo_adj = sp.coo_matrix(adj)
    # 使用np.vstack提高效率
    edge_array = np.vstack([coo_adj.row, coo_adj.col])
    return torch.tensor(edge_array, dtype=torch.long)


def dr_data_to_pyg_data(adj, features, labels):
    """
    transfer deeprobust data to pyg data
    :return:
    """
    features_dense = features.toarray() if issparse(features) else features

    pyg_data = Data(
        x=torch.tensor(features_dense, dtype=torch.float),
        edge_index=adj_to_edge_index(adj),
        # adj=torch.tensor(adj.toarray(), dtype=torch.float) if str(type(adj)) != "<class 'torch.Tensor'>" else adj,
        y=torch.tensor(labels)
    )
    return pyg_data


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
            self.pyg_data.edge_index = to_undirected(
                self.pyg_data.edge_index)  # ogbn-arxiv has no self-loop edges, function of to_undirected can't delete self-loop by itself

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
