#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/8/27 20:17
# @Author   : **
# @Email    : **@**
# @File     : gcn_perturb.py
# @Software : PyCharm
# @Desc     :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from deeprobust.graph.defense import GraphConvolution
import numpy as np

from utilty.utils import normalize_adj, get_degree_matrix, compute_deg_diff, compute_motif_viol, compute_feat_sim
from config.config import TEST_MODEL
from torch_geometric.nn import GINConv, SAGEConv, APPNP, GraphConv, TransformerConv, GATConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.utils import dense_to_sparse


class SignedMaskPerturbation(nn.Module):
    def __init__(self,
                 extended_sub_adj: torch.Tensor,
                 node_idx: int,
                 node_num_l_hop: list,
                 top_k: int = 5,
                 tau_plus: float = 0.5,
                 tau_minus: float = -0.5,
                 test_model: str = "GCN",
                 dataset_name: str = "cora"):
        """
        AC-Explainer的带符号掩码扰动模块
        参数:
            extended_sub_adj: 扩展后的子图邻接矩阵 [n, n]
            node_idx: 目标节点在扩展子图中的索引
            top_k: 保留的最大边数 (默认5)
            tau_plus: 添加边的阈值 (默认0.5)
            tau_minus: 删除边的阈值 (默认-0.5)
        """
        super().__init__()
        self.extended_sub_adj = extended_sub_adj  # 克隆扩展子图邻接矩阵，避免影响原数据
        self.node_idx = node_idx  # 目标节点在子图中的索引
        self.tau_plus = tau_plus  # 添加边的阈值
        self.tau_minus = tau_minus  # 删除边的阈值
        self.node_num_l_hop = node_num_l_hop
        self.top_k = top_k  # 保留的最大边修改数量
        self.n_nodes = extended_sub_adj.size(0)  # 扩展子图中的节点数
        self.plan_added_node_idx = []
        self.plan_deleted_node_idx = []
        self.test_model = test_model
        self.dataset_name = dataset_name

        # 初始化带符号的掩码参数
        self.M = self._initialize_mask()

    def _initialize_mask(self) -> nn.Parameter:
        """根据目标节点和扩展子图初始化掩码"""
        eps = 10 ** -4
        mask_init_values = []
        mask_index = 0

        [node_index, attack_nodes, node_dict] = self.node_num_l_hop
        attack_nodes_idx = [node_dict[ad] for ad in attack_nodes]
        lhop_node_index = [node_dict[ni] for ni in node_index]

        # 遍历extended_sub_adj中所有现有边
        # sub_adj = self.extended_sub_adj[lhop_node_index, :][:, lhop_node_index]
        # init_value = -0.8  # GCN:-0.5 GraphConv:-1.0 GraphTransformer: -0.8
        init_value = {
            "GCN": {"cora": -0.5, "BA-SHAPES": -0.5, "TREE-CYCLES": -0.5, "Loan-Decision": -0.5, "ogbn-arxiv": -0.5},
            "GraphTransformer": {"cora": -0.8, "BA-SHAPES": -0.6, "TREE-CYCLES": -0.8, "Loan-Decision": -0.8, "ogbn-arxiv": -0.8},
            "GraphConv": {"cora": -1.0, "BA-SHAPES": -1.0, "TREE-CYCLES": -1.0, "Loan-Decision": -1.0, "ogbn-arxiv": -1.0},
            "GAT": {"cora": -0.8, "BA-SHAPES": -0.8, "TREE-CYCLES": -0.8, "Loan-Decision": -0.8, "ogbn-arxiv": -0.8}
        }[self.test_model][self.dataset_name]
        init_value += 0.01*torch.randn(1).item()
        ones_indices = torch.nonzero(self.extended_sub_adj == 1)
        non_diagonal_ones = ones_indices[ones_indices[:, 0] != ones_indices[:, 1]].tolist()
        for i in range(len(non_diagonal_ones)):
            if non_diagonal_ones[i][0] in lhop_node_index and non_diagonal_ones[i][1] in lhop_node_index and \
                    non_diagonal_ones[i][0] < non_diagonal_ones[i][1]:
                # 现有边初始化为小负数 (倾向删除)
                mask_init_values.append(init_value)
                self.plan_deleted_node_idx.append([mask_index, non_diagonal_ones[i]])
                mask_index += 1
        # 遍历所有attack_nodes，针对无现有边场景倾向添加，但需要抑制加边
        # init_value = 0.8  # GCN：0.4 GraphConv:0.55 GraphTransformer: 0.8
        init_value = {
            "GCN": {"cora": 0.4, "BA-SHAPES": 0.8, "TREE-CYCLES": 0.4, "Loan-Decision": 0.4, "ogbn-arxiv": 0.4},
            "GraphTransformer": {"cora": 0.8, "BA-SHAPES": 0.6, "TREE-CYCLES": 0.8, "Loan-Decision": 0.8, "ogbn-arxiv": 0.8},
            "GraphConv": {"cora": 0.55, "BA-SHAPES": 0.55, "TREE-CYCLES": 0.55, "Loan-Decision": 0.55, "ogbn-arxiv": 0.55},
            "GAT": {"cora": 0.8, "BA-SHAPES": 0.8, "TREE-CYCLES": 0.8, "Loan-Decision": 0.8, "ogbn-arxiv": 0.8}
        }[self.test_model][self.dataset_name]
        init_value -= 0.01 * torch.randn(1).item()
        for i in attack_nodes_idx:
            if i != self.node_idx:
                mask_init_values.append(init_value)
                self.plan_added_node_idx.append([mask_index, [self.node_idx, i]])
                mask_index += 1

        return nn.Parameter(torch.tensor(mask_init_values, dtype=torch.float32))

    def _apply_discretization(self, M_e: torch.Tensor) -> torch.Tensor:
        """
        应用TopK稀疏化 (仅保留梯度最大的k个扰动)
        应用三值离散化：-1(删除), 0(不变), +1(添加)
        """
        with torch.no_grad():  # 离散化操作不需要梯度
            # 应用TopK稀疏化 (仅保留梯度最大的k个扰动)
            abs_values = torch.abs(M_e)  # 计算连续掩码的绝对值（衡量扰动强度
            top_k_M_e = M_e
            if len(M_e) > self.top_k:
                # 找出绝对值最大的top_k个索引
                # keep_k = min(self.top_k, int(0.5 * len(M_e)))
                topk_indices = torch.topk(abs_values, self.top_k).indices
                sparse_mask = torch.zeros_like(M_e)  # 创建全0掩码
                sparse_mask[topk_indices] = 1  # 仅将top_k个位置设为1
                top_k_M_e = M_e * sparse_mask  # 应用稀疏掩码

            full_mask = torch.zeros_like(self.extended_sub_adj)
            for data in self.plan_added_node_idx + self.plan_deleted_node_idx:
                full_mask[data[1][0], data[1][1]] = top_k_M_e[data[0]]
                full_mask[data[1][1], data[1][0]] = top_k_M_e[data[0]]
            # edge_idx = 0
            # for i in range(self.n_nodes):
            #     if i != self.node_idx and i in (self.plan_added_node_idx + self.plan_deleted_node_idx):
            #         full_mask[self.node_idx, i] = top_k_M_e[edge_idx]
            #         full_mask[i, self.node_idx] = top_k_M_e[edge_idx]
            #         edge_idx += 1

            # 计算离散值(使用torch.where进行三值化)
            delta_A = torch.where(
                full_mask > self.tau_plus,
                1,
                torch.where(
                    full_mask < self.tau_minus,
                    -1,
                    0
                )
            )

        return delta_A

    def train_forward(self) -> torch.Tensor:
        """
        训练模式：返回连续近似的mask（保持梯度)
        使用直通梯度估计器保持可微性
        """
        full_mask = torch.zeros_like(self.extended_sub_adj)
        for data in self.plan_added_node_idx + self.plan_deleted_node_idx:
            full_mask[data[1][0], data[1][1]] = self.M[data[0]]
            full_mask[data[1][1], data[1][0]] = self.M[data[0]]
        # edge_idx = 0
        # for i in range(self.n_nodes):
        #     if i != self.node_idx and i in (self.plan_added_node_idx + self.plan_deleted_node_idx):
        #         full_mask[self.node_idx, i] = self.M[edge_idx]
        #         full_mask[i, self.node_idx] = self.M[edge_idx]
        #         edge_idx += 1
        return full_mask

    def predict_forward(self) -> torch.Tensor:
        """预测模式：返回完全离散的mask（无梯度）"""
        M_e = torch.tanh(self.M)
        delta_A = self._apply_discretization(M_e)
        return delta_A

    def build_perturbed_adj(self, adj, delta_A):
        perturbed_adj = torch.where(
            delta_A == 1,  # 条件：如果 delta_A 指示“增加边”
            torch.ones_like(adj),  # 则对应位置设为 1
            torch.where(
                delta_A == -1,  # 否则，如果 delta_A 指示“删除边”
                torch.zeros_like(adj),  # 则对应位置设为 0
                adj  # 否则（delta_A == 0），保持原邻接矩阵的值不变
            )
        )
        return perturbed_adj

    def ste_perturbed_adj(self, adj, full_mask):
        # 1. 通过 tanh 将 full_mask 映射到 [-1, 1] 区间，这是一个可微操作
        continuous_mask = torch.tanh(full_mask)  # 保持梯度流
        # 2. 在前向传播中，根据 continuous_mask 的值进行离散决策
        with torch.no_grad():
            # 创建与 continuous_mask 同形状的矩阵，存放离散决策
            discrete_decision = torch.where(continuous_mask > 0.5,  # 条件1：大于0.5
                                            torch.ones_like(continuous_mask),  # 满足条件1：置1（添加边）
                                            torch.where(continuous_mask < -0.5,  # 条件2：小于-0.5
                                                        -torch.ones_like(continuous_mask),  # 满足条件2：置-1（删除边）
                                                        torch.zeros_like(continuous_mask)))  # 否则：置0（不变）
            # 根据离散决策生成扰动后的邻接矩阵
            perturbed_adj_discrete = torch.where(discrete_decision > 0.5,
                                                 torch.ones_like(adj),
                                                 torch.where(discrete_decision < -0.5,
                                                             torch.zeros_like(adj),
                                                             adj))

        # 3. 关键步骤：使用直通估计器连接前向离散决策和反向连续梯度
        perturbed_adj = perturbed_adj_discrete + (continuous_mask - continuous_mask.detach())

        return perturbed_adj


class GNNPerturb(nn.Module):
    def __init__(self,
                 nfeat: int,
                 nhid: int,
                 nclass: int,
                 extended_sub_adj: torch.Tensor,
                 sub_feat: torch.Tensor,
                 node_idx: int,
                 node_num_l_hop: list,
                 dropout: float = 0.5,
                 lambda_pred: float = 1.0,
                 lambda_dist: float = 0.5,
                 lambda_plau: float = 0.2,
                 top_k: int = 5,
                 tau_plus: float = 0.5,
                 tau_minus: float = -0.5,
                 α1: float = 0.1,
                 α2: float = 0.1,
                 α3: float = 0.1,
                 α4: float = 0.5,
                 tau_c: float = 0.1,
                 gcn_layer: int = 2,
                 with_bias: bool = True,
                 test_model: str = "GCN",
                 heads: int = 2,
                 dataset_name: str = "cora"):
        super().__init__()

        self.gcn_layer = gcn_layer
        self.lambda_pred = lambda_pred  # 预测损失权重
        self.lambda_dist = lambda_dist  # 稀疏损失权重
        self.lambda_plau = lambda_plau  # 现实性损失权重
        self.dropout = dropout
        self.extended_sub_adj = extended_sub_adj
        self.sub_feat = sub_feat
        self.num_nodes = self.extended_sub_adj.shape[0]
        self.node_idx = node_idx
        self.node_num_l_hop = node_num_l_hop
        self.α1 = α1
        self.α2 = α2
        self.α3 = α3
        self.α4 = α4
        self.tau_c = tau_c
        self.model_name = test_model
        self.dataset_name = dataset_name
        self.heads = heads

        # 扰动层
        print(f"Input extended_sub_adj.requires_grad: {extended_sub_adj.requires_grad}")
        self.perturb_layer = SignedMaskPerturbation(extended_sub_adj, node_idx, node_num_l_hop, top_k, tau_plus,
                                                    tau_minus, test_model, dataset_name)

        # GCN层定义
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

    def forward(self, x: torch.Tensor, sub_adj: torch.Tensor) -> torch.Tensor:
        """训练模式：使用连续扰动矩阵"""
        self.sub_adj = sub_adj
        self.full_mask = self.perturb_layer.train_forward()

        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True

        # Use tanh to bound full mask in [-1,1]
        scale = 1.0
        perturbed_adj = self.perturb_layer.ste_perturbed_adj(self.sub_adj, scale * self.full_mask)
        A_tilde = perturbed_adj + torch.eye(self.num_nodes)

        D_tilde = get_degree_matrix(A_tilde).detach()  # Don't need gradient of this 度矩阵
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)  # 归一化邻接矩阵

        return self._gcn_forward(x, norm_adj)

    def forward_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """预测模式：使用离散扰动矩阵"""
        self.delta_A = self.perturb_layer.predict_forward()

        A_tilde = self.perturb_layer.build_perturbed_adj(self.extended_sub_adj, self.delta_A) + torch.eye(
            self.num_nodes)  # 离散化邻接矩阵

        D_tilde = get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        return self._gcn_forward(x, norm_adj)

    def _gcn_forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
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
            edge_index, edge_weight = dense_to_sparse(norm_adj)
            edge_index = edge_index.to(x.device)
            edge_attr = edge_weight.view(-1, 1)  # [num_edges, 1]
            edge_attr.requires_grad_(True)
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
            edge_index, edge_weight = dense_to_sparse(norm_adj)
            edge_index = edge_index.to(x.device)
            edge_attr = edge_weight.view(-1, 1)  # [num_edges, 1]
            edge_attr.requires_grad_(True)
            for conv in self.layers[:-1]:
                x = conv(x, edge_index, edge_weight=edge_attr)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # 最后一层
            x = self.layers[-1](x, edge_index, edge_weight=edge_attr)
            return F.log_softmax(x, dim=1)

    def get_mask_parameters(self) -> nn.Parameter:
        """获取可训练的掩码参数"""
        return self.perturb_layer.M

    def compute_losses(self,
                       output: torch.Tensor,
                       y_pred_orig: torch.Tensor,
                       y_pred_new_actual: torch.Tensor) -> tuple:
        """计算多目标损失函数"""

        # 预测损失 (鼓励翻转预测)
        pred_loss = -F.nll_loss(
            output[self.node_idx].unsqueeze(0),
            y_pred_orig.unsqueeze(0)
        ) * (y_pred_new_actual == y_pred_orig.unsqueeze(0)).float()
        # pred_loss = F.cross_entropy(output[self.node_idx].unsqueeze(0), y_pred_orig.unsqueeze(0))
        # # 添加鼓励预测改变的项，使用更激进的策略
        # if y_pred_new_actual == y_pred_orig:
        #     # 如果预测没有改变，增加更大的损失以鼓励改变
        #     pred_loss += 5.0 * pred_loss
        # # 添加注意力一致性损失，鼓励注意力权重的稀疏性
        # attention_loss = 0
        # for name, module in self.named_modules():
        #     if hasattr(module, 'att') and module.att is not None:
        #         # 鼓励注意力权重稀疏化
        #         attention_loss += torch.mean(torch.abs(module.att.weight))

        # 稀疏损失 (L0范数)
        # cf_adj = self.perturb_layer.build_perturbed_adj(self.extended_sub_adj, self.delta_A)
        # cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient
        # # dist_loss = sum(sum(abs(cf_adj - self.extended_sub_adj))) / 2
        # deletion_mask = (self.delta_A == -1) & (self.extended_sub_adj == 1)
        # num_deletions = deletion_mask.sum() / 2
        # addition_mask = (self.delta_A == 1) & (self.extended_sub_adj == 0)
        # num_additions = addition_mask.sum() / 2

        cf_adj = self.perturb_layer.ste_perturbed_adj(self.extended_sub_adj, self.full_mask)
        if self.lambda_dist == 0:
            dist_loss = torch.tensor(0.0)
        else:
            dist_loss = torch.sum(torch.abs(cf_adj - self.extended_sub_adj)) / 2

        # 现实性损失
        if self.lambda_plau == 0:
            plau_loss = torch.tensor(0.0)
        else:
            plau_loss = self.compute_plausibility_loss()

        # 加权总损失
        total_loss = self.lambda_pred * pred_loss + self.lambda_dist * dist_loss + self.lambda_plau * plau_loss

        return total_loss, pred_loss, dist_loss, plau_loss, cf_adj, self.delta_A, self.perturb_layer, self.full_mask

    def compute_plausibility_loss(self) -> torch.Tensor:
        """计算现实性损失"""
        loss_components = torch.tensor(0.0)
        loss_components_1 = torch.tensor(0.0)
        loss_components_2 = torch.tensor(0.0)
        loss_components_3 = torch.tensor(0.0)
        loss_components_4 = torch.tensor(0.0)

        # 1. 特征相似度惩罚 (仅对添加边)
        # add_mask = (self.delta_A > 0.5)
        # if add_mask.sum() > 0:
        #     # 计算特征相似度
        #     target_feat = self.sub_feat[self.node_idx]
        #     for i in range(self.extended_sub_adj.size(0)):
        #         if add_mask[self.node_idx, i]:
        #             feat_sim = compute_feat_sim(target_feat, self.sub_feat[i])
        #             loss_components_1 = loss_components_1 + (1 - feat_sim) * self.α1
        #     loss_components_1 = loss_components_1 / add_mask.sum()

        # 2. 度分布惩罚
        orig_sub_adj = self.extended_sub_adj
        # edited_sub_adj = self.perturb_layer.build_perturbed_adj(self.extended_sub_adj, self.delta_A)
        # deg_diff = compute_deg_diff(orig_sub_adj, edited_sub_adj)

        cf_adj_soft = self.perturb_layer.ste_perturbed_adj(self.extended_sub_adj, self.full_mask)
        deg_diff = compute_deg_diff(orig_sub_adj, cf_adj_soft)

        loss_components_2 = deg_diff * self.α2

        # 3. penalty of clustering coefficients drastic changes
        # motif_violation = compute_motif_viol(orig_sub_adj, edited_sub_adj, self.tau_c)

        motif_violation = compute_motif_viol(orig_sub_adj, cf_adj_soft, self.tau_c)

        loss_components_3 = motif_violation * self.α3

        # 4. domain-specific constraint
        # publish_year = None
        # violation_count = 0
        # if add_mask.sum() > 0 and publish_year:
        #     target_year = publish_year[self.node_idx]
        #     for i in range(self.extended_sub_adj.size(0)):
        #         if add_mask[self.node_idx, i]:
        #             year_i = publish_year[i]
        #             # 如果节点i的年份早于目标节点，但存在从i到j的边，则违反规则
        #             if (year_i < target_year) and self.extended_sub_adj[i, self.node_idx]:
        #                 violation_count += 1
        #             # 同样检查相反情况
        #             elif (target_year < year_i) and self.extended_sub_adj[self.node_idx, i]:
        #                 violation_count += 1
        #     sem_cost = violation_count / add_mask.sum()
        #     loss_components_4 = loss_components_4 + torch.tensor(sem_cost, dtype=float) * self.α4

        loss_components = loss_components_1 + loss_components_2 + loss_components_3 + loss_components_4

        return loss_components
