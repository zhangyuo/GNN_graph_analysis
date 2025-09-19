import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from deeprobust.graph.defense import GraphConvolution
from torch.nn.parameter import Parameter
from torch_geometric.nn import TransformerConv, GraphConv, GATConv, DenseGATConv
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from utilty.utils import get_degree_matrix, create_symm_matrix_from_vec, create_vec_from_symm_matrix


class GCNCoraPerturb(nn.Module):
    """
    2-layer GCN used in GNN Explainer cora tasks
    """

    def __init__(self, nfeat, nhid, nclass, adj, dropout, beta,gcn_layer, with_bias, test_model, dataset_name, heads, edge_additions=False):
        super(GCNCoraPerturb, self).__init__()
        self.adj = adj
        self.nclass = nclass
        self.beta = beta
        self.num_nodes = self.adj.shape[0]
        self.gcn_layer = gcn_layer
        self.edge_additions = edge_additions  # are edge additions included in perturbed matrix
        self.heads = heads
        self.model_name = test_model
        self.dataset_name=dataset_name

        # P_hat needs to be symmetric ==> learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
        self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2) + self.num_nodes  # # 上三角元素数量

        if self.edge_additions:
            self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size)))
        else:
            self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))
        self.reset_parameters()

        if self.model_name == "GCN":
            if self.gcn_layer == 3:
                self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
                self.gc2 = GraphConvolution(nhid, nhid, with_bias=with_bias)
                self.gc3 = GraphConvolution(nhid, nclass, with_bias=with_bias)
                self.lin = nn.Linear(nhid + nhid + nclass, nclass)
            else:
                self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
                self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        elif self.model_name == "GraphTransformer":
            self.layers = nn.ModuleList()
            self.layers.append(TransformerConv(nfeat, nhid, heads=heads, dropout=dropout, edge_dim=1))
            for _ in range(self.gcn_layer - 2):
                self.layers.append(
                    TransformerConv(nhid * heads, nhid, heads=heads, dropout=dropout, edge_dim=1))
            self.layers.append(TransformerConv(nhid * heads, nclass, heads=1, dropout=dropout, edge_dim=1))
        elif self.model_name == "GraphConv":
            self.layers = nn.ModuleList()
            self.layers.append(GraphConv(nfeat, nhid, aggr="add"))
            for _ in range(self.gcn_layer - 2):
                self.layers.append(GraphConv(nhid, nhid, aggr="add"))
            self.layers.append(GraphConv(nhid, nclass, aggr="add"))
        elif self.model_name == "GAT":
            self.layers = nn.ModuleList()
            self.layers.append(GATConv(nfeat, nhid, heads=heads, dropout=dropout, edge_dim=1))
            for _ in range(self.gcn_layer - 2):
                self.layers.append(GATConv(nhid * heads, nhid, heads=heads, dropout=dropout, edge_dim=1))
            if self.dataset_name == "Cora":
                self.layers.append(GATConv(nhid * heads, nclass, heads=1, concat=False, dropout=dropout, edge_dim=1))
            else:
                self.layers.append(GATConv(nhid * heads, nclass, heads=1, dropout=dropout, edge_dim=1))
        self.dropout = dropout

    def reset_parameters(self, eps=10 ** -4):
        # Think more about how to initialize this
        # eps = 20
        with torch.no_grad():
            if self.edge_additions:
                adj_vec = create_vec_from_symm_matrix(self.adj, self.P_vec_size).numpy()
                for i in range(len(adj_vec)):
                    if i < 1:
                        adj_vec[i] = adj_vec[i] - eps
                    else:
                        adj_vec[i] = adj_vec[i] + eps
                torch.add(self.P_vec, torch.FloatTensor(adj_vec))  # self.P_vec is all 0s
            else:
                torch.sub(self.P_vec, eps)
                # init_range = 0.9
                # min_value = 0.1
                # self.P_vec.data.uniform_(min_value, init_range)

    def get_mask_parameters(self) -> nn.Parameter:
        """获取可训练的掩码参数"""
        return self.P_vec

    def forward(self, x, sub_adj):
        self.sub_adj = sub_adj
        # Same as normalize_adj in utils.py except includes P_hat in A_tilde
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)  # Ensure symmetry 向量→对称矩阵

        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True

        if self.edge_additions:  # Learn new adj matrix directly
            # 扰动后邻接矩阵
            A_tilde = F.sigmoid(self.P_hat_symm) + torch.eye(self.num_nodes)  # Use sigmoid to bound P_hat in [0,1]
        else:  # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
            A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(
                self.num_nodes)  # Use sigmoid to bound P_hat in [0,1]

        D_tilde = get_degree_matrix(A_tilde).detach()  # Don't need gradient of this 度矩阵
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)  # 归一化邻接矩阵

        if self.model_name == "GCN":
            if self.gcn_layer == 3:
                x1 = F.relu(self.gc1(x, norm_adj))
                x1 = F.dropout(x1, self.dropout, training=self.training)
                x2 = F.relu(self.gc2(x1, norm_adj))
                x2 = F.dropout(x2, self.dropout, training=self.training)
                x3 = self.gc3(x2, norm_adj)
                x = self.lin(torch.cat((x1, x2, x3), dim=1))
                return F.log_softmax(x, dim=1)
            else:
                x1 = F.relu(self.gc1(x, norm_adj))
                x1 = F.dropout(x1, self.dropout, training=self.training)
                x2 = self.gc2(x1, norm_adj)
                return F.log_softmax(x2, dim=1)
        elif self.model_name in ["GraphTransformer", "GAT"]:
            edge_index, _ = dense_to_sparse(norm_adj)
            edge_index = edge_index.to(x.device)
            edge_attr = norm_adj[edge_index[0], edge_index[1]].view(-1, 1)  # for return grad in backward
            edge_attr = edge_attr.requires_grad_(True)
            for conv in self.layers[:-1]:
                x = conv(x, edge_index, edge_attr=edge_attr)
                if self.model_name == "GAT" and self.dataset_name == "BA-SHAPES":
                    x = F.elu(x)
                else:
                    x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # 最后一层
            x = self.layers[-1](x, edge_index, edge_attr=edge_attr)
            return F.log_softmax(x, dim=1)
        elif self.model_name in ["GraphConv"]:
            edge_index, _ = dense_to_sparse(norm_adj)
            edge_index = edge_index.to(x.device)
            edge_attr = norm_adj[edge_index[0], edge_index[1]].view(-1, 1)  # for return grad in backward
            edge_attr = edge_attr.requires_grad_(True)
            for conv in self.layers[:-1]:
                x = conv(x, edge_index, edge_weight=edge_attr)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # 最后一层
            x = self.layers[-1](x, edge_index, edge_weight=edge_attr)
            return F.log_softmax(x, dim=1)

    def forward_prediction(self, x):
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions  双模式预测机制

        self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()  # threshold P_hat 二值化阈值

        if self.edge_additions:
            A_tilde = self.P + torch.eye(self.num_nodes)
        else:
            A_tilde = self.P * self.adj + torch.eye(self.num_nodes)  # 离散化邻接矩阵

        D_tilde = get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        if self.model_name == "GCN":
            if self.gcn_layer == 3:
                x1 = F.relu(self.gc1(x, norm_adj))
                x1 = F.dropout(x1, self.dropout, training=self.training)
                x2 = F.relu(self.gc2(x1, norm_adj))
                x2 = F.dropout(x2, self.dropout, training=self.training)
                x3 = self.gc3(x2, norm_adj)
                x = self.lin(torch.cat((x1, x2, x3), dim=1))
                return F.log_softmax(x, dim=1), self.P
            else:
                x1 = F.relu(self.gc1(x, norm_adj))
                x1 = F.dropout(x1, self.dropout, training=self.training)
                x2 = self.gc2(x1, norm_adj)
                return F.log_softmax(x2, dim=1), self.P
        elif self.model_name in ["GraphTransformer", "GAT"]:
            edge_index, edge_weight = dense_to_sparse(norm_adj)
            edge_index = edge_index.to(x.device)
            edge_attr = edge_weight.view(-1, 1)  # [num_edges, 1]
            for conv in self.layers[:-1]:
                x = conv(x, edge_index, edge_attr=edge_attr)
                if self.model_name == "GAT" and self.dataset_name == "BA-SHAPES":
                    x = F.elu(x)
                else:
                    x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # 最后一层
            x = self.layers[-1](x, edge_index, edge_attr=edge_attr)
            return F.log_softmax(x, dim=1), self.P
        elif self.model_name in ["GraphConv", "GAT"]:
            edge_index, edge_weight = dense_to_sparse(norm_adj)
            edge_index = edge_index.to(x.device)
            edge_attr = edge_weight.view(-1, 1)  # [num_edges, 1]
            for conv in self.layers[:-1]:
                x = conv(x, edge_index, edge_weight=edge_attr)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # 最后一层
            x = self.layers[-1](x, edge_index, edge_weight=edge_attr)
            return F.log_softmax(x, dim=1), self.P

    def loss(self, output, y_pred_orig, y_pred_new_actual):
        """
        反事实损失函数
        """
        pred_same = (y_pred_new_actual == y_pred_orig).float()  # 当预测未改变时梯度为0（停止优化）

        # Need dim >=2 for F.nll_loss to work
        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)

        if self.edge_additions:
            cf_adj = self.P
        else:
            cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig)  # 预测差异损失（负号表示最大化差异）
        loss_graph_dist = sum(sum(abs(cf_adj - self.adj))) / 2  # Number of edges changed (symmetrical) 图结构变化量（边改变数）

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist  # 复合损失, β：平衡预测改变与图修改量的超参数
        return loss_total, loss_pred, loss_graph_dist, cf_adj
