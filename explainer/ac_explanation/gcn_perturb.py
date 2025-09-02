#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/8/27 20:17
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : gcn_perturb.py
# @Software : PyCharm
# @Desc     :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from deeprobust.graph.defense import GraphConvolution

from utilty.utils import normalize_adj


class SignedMaskPerturbation(nn.Module):
    def __init__(self,
                 extended_sub_adj: torch.Tensor,
                 target_node: int,
                 original_adj_mask: torch.Tensor,
                 tau_plus: float = 0.5,
                 tau_minus: float = -0.5,
                 top_k: int = 5):
        """
        AC-Explainer的带符号掩码扰动模块
        参数:
            extended_sub_adj: 扩展后的子图邻接矩阵 [n, n]
            target_node: 目标节点在扩展子图中的索引
            original_adj_mask: 原始图中边存在的掩码矩阵
            tau_plus: 添加边的阈值 (默认0.5)
            tau_minus: 删除边的阈值 (默认-0.5)
            top_k: 保留的最大边数 (默认5)
        """
        super().__init__()
        self.extended_sub_adj = extended_sub_adj  # 克隆扩展子图邻接矩阵，避免影响原数据
        self.target_node = target_node  # 目标节点在子图中的索引
        self.original_adj_mask = original_adj_mask  # 原始图中边存在的掩码矩阵 (1表示边存在)
        self.tau_plus = tau_plus  # 添加边的阈值
        self.tau_minus = tau_minus  # 删除边的阈值
        self.top_k = top_k  # 保留的最大边修改数量
        self.n_nodes = extended_sub_adj.size(0)  # 扩展子图中的节点数

        # 初始化带符号的掩码参数
        self.M = self.initialize_mask()

    def initialize_mask(self) -> nn.Parameter:
        """根据目标节点和扩展子图初始化掩码"""
        mask_init_values = []

        # 遍历扩展子图中的所有节点（除了目标节点自己）
        for i in range(self.n_nodes):
            if i == self.target_node:
                continue

            # 检查在原始图中是否存在边
            if self.original_adj_mask[self.target_node, i]:
                # 现有边初始化为小负数 (倾向删除)
                mask_init_values.append(-0.2)
            else:
                # 新边初始化为小正数 (倾向添加)
                mask_init_values.append(0.2)

        # 转换为可训练参数--将列表转换为PyTorch张量，并封装为可学习参数(Parameter)
        return nn.Parameter(torch.tensor(mask_init_values, dtype=torch.float32))

    def apply_discretization(self, M_cont: torch.Tensor) -> torch.Tensor:
        """
        应用三值离散化：-1(删除), 0(不变), +1(添加)
        使用直通梯度估计器保持可微性
        """
        # 计算离散值(使用torch.where进行三值化)
        with torch.no_grad():  # 离散化操作不需要梯度
            discrete_M = torch.where(
                M_cont > self.tau_plus,  # 如果大于正阈值
                torch.ones_like(M_cont),  # 则置为1 (添加边)
                torch.where(
                    M_cont < self.tau_minus,  # 如果小于负阈值
                    -torch.ones_like(M_cont),  # 则置为-1 (删除边)
                    torch.zeros_like(M_cont)  # 否则置为0 (不变)
                )
            )

        # 应用TopK稀疏化 (仅保留梯度最大的k个扰动)
        abs_values = torch.abs(M_cont)  # 计算连续掩码的绝对值（衡量扰动强度）
        if len(M_cont) > self.top_k:
            # 找出绝对值最大的top_k个索引
            topk_indices = torch.topk(abs_values, self.top_k).indices
            sparse_mask = torch.zeros_like(M_cont) # 创建全0掩码
            sparse_mask[topk_indices] = 1 # 仅将top_k个位置设为1
            discrete_M = discrete_M * sparse_mask # 应用稀疏掩码

        # 直通梯度估计器: 保持前向离散，反向传播连续梯度
        return discrete_M + (M_cont - M_cont.detach())

    def forward(self) -> torch.Tensor:
        """生成扰动后的邻接矩阵"""
        # 应用tanh约束到[-1, 1]
        M_cont = torch.tanh(self.M)
        # M_cont.requires_grad = True
        print(f"M_cont.requires_grad: {M_cont.requires_grad}")  # 应为 True

        # 应用离散化处理
        M_disc = self.apply_discretization(M_cont)
        print(f"M_disc.requires_grad: {M_disc.requires_grad}")  # 应为 True

        # 创建与计算图连接的掩码矩阵
        full_mask = torch.zeros_like(self.extended_sub_adj) + 0.0 * M_cont.sum()

        # 准备索引和值
        row_indices, col_indices = [], []
        for i in range(self.n_nodes):
            if i != self.target_node:
                row_indices.extend([self.target_node, i])
                col_indices.extend([i, self.target_node])

        rows = torch.tensor(row_indices, dtype=torch.long, device=M_disc.device)
        cols = torch.tensor(col_indices, dtype=torch.long, device=M_disc.device)
        values = M_disc.repeat_interleave(2)

        # 离散操作（前向传播）
        with torch.no_grad():
            full_mask_discrete = full_mask.clone()
            full_mask_discrete[rows, cols] = values.detach()

        # 连续近似（梯度传播）
        continuous_values = M_cont.repeat_interleave(2)
        continuous_mask = torch.zeros_like(full_mask)
        continuous_mask[rows, cols] = continuous_values

        # 直通估计器
        full_mask_ste = full_mask_discrete + (continuous_mask - continuous_mask.detach())

        # 生成扰动邻接矩阵
        perturbed_adj = torch.where(
            full_mask_ste > 0.5,
            torch.ones_like(self.extended_sub_adj),
            torch.where(
                full_mask_ste < -0.5,
                torch.zeros_like(self.extended_sub_adj),
                self.extended_sub_adj
            )
        )
        return perturbed_adj

    def get_perturbation_matrix(self) -> torch.Tensor:
        """获取ΔA矩阵 (-1, 0, 1)"""
        with torch.no_grad(): # 不需要梯度
            M_cont = torch.tanh(self.M)
            # 离散化过程（同apply_discretization，但不使用STE）
            discrete_M = torch.where(
                M_cont > self.tau_plus,
                torch.ones_like(M_cont),
                torch.where(
                    M_cont < self.tau_minus,
                    -torch.ones_like(M_cont),
                    torch.zeros_like(M_cont)
                )
            )

            # 创建全尺寸扰动矩阵
            full_delta = torch.zeros_like(self.extended_sub_adj)
            edge_idx = 0
            for i in range(self.n_nodes):
                if i != self.target_node:
                    full_delta[self.target_node, i] = discrete_M[edge_idx]
                    full_delta[i, self.target_node] = discrete_M[edge_idx]
                    edge_idx += 1

            return full_delta


class GCNPerturb(nn.Module):
    def __init__(self,
                 gnn_model,
                 nfeat: int,
                 nhid: int,
                 nclass: int,
                 extended_sub_adj: torch.Tensor,
                 target_node: int,
                 original_adj_mask: torch.Tensor,
                 dropout: float = 0.5,
                 gcn_layer: int = 2,
                 with_bias: bool = True):
        super().__init__()
        print(f"Input extended_sub_adj.requires_grad: {extended_sub_adj.requires_grad}")
        self.perturb_layer = SignedMaskPerturbation(
            extended_sub_adj, target_node, original_adj_mask
        )

        # GCN层定义
        self.gcn_layer = gcn_layer
        self.dropout = dropout
        self.extended_sub_adj = extended_sub_adj

        if self.gcn_layer == 3:
            self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
            self.gc2 = GraphConvolution(nhid, nhid, with_bias=with_bias)
            self.gc3 = GraphConvolution(nhid, nclass, with_bias=with_bias)
            self.lin = nn.Linear(nhid + nhid + nclass, nclass)
        else:
            self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
            self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)

        # 继承原模型参数
        # self.model = gnn_model
        self.load_state_dict(gnn_model.state_dict(), strict=False)
        self.training = False

        # Freeze weights from original model in cf_model 冻结原始参数，仅训练扰动矩阵
        for name, param in self.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False
        for name, param in gnn_model.named_parameters():
            print("orig model requires_grad: ", name, param.requires_grad)
        for name, param in self.named_parameters():
            print("cf model requires_grad: ", name, param.requires_grad)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 生成扰动后的邻接矩阵
        adj_perturbed = self.perturb_layer()  # 作用是调用 SignedMaskPerturbation 模块的 forward 方法，生成一个扰动后的邻接矩阵

        # 添加自环并归一化
        # adj_perturbed = adj_perturbed.fill_diagonal_(1) # 添加自环
        # adj_normalized = self.normalize_adj(adj_perturbed) # 对称归一化
        adj_normalized = normalize_adj(adj_perturbed)

        if self.gcn_layer == 3:
            x1 = F.relu(self.gc1(x, adj_normalized))
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x2 = F.relu(self.gc2(x1, adj_normalized))
            x2 = F.dropout(x2, self.dropout, training=self.training)
            x3 = self.gc3(x2, adj_normalized)
            x = self.lin(torch.cat((x1, x2, x3), dim=1))
            return F.log_softmax(x, dim=1)
        else:
            x1 = F.relu(self.gc1(x, adj_normalized))
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x2 = self.gc2(x1, adj_normalized)
            return F.log_softmax(x2, dim=1)

    def normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """对称归一化邻接矩阵 D^{-1/2} A D^{-1/2}"""
        rowsum = torch.sum(adj, dim=1) # 计算每个节点的度
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten() # 计算度的-1/2次方
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0. # 将无穷大值（度为零的节点）置零
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt) # 构建度矩阵的逆平方根（对角矩阵）
        # 计算归一化后的邻接矩阵
        return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    def get_mask_parameters(self) -> nn.Parameter:
        """获取可训练的掩码参数"""
        return self.perturb_layer.M
