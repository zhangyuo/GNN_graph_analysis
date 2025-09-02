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
        self.extended_sub_adj = extended_sub_adj.clone().detach()
        self.target_node = target_node
        self.original_adj_mask = original_adj_mask
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.top_k = top_k
        self.n_nodes = extended_sub_adj.size(0)

        # 初始化带符号的掩码参数
        self.M = self.initialize_mask()

    def initialize_mask(self) -> nn.Parameter:
        """根据目标节点和扩展子图初始化掩码"""
        mask_init_values = []

        # 遍历扩展子图中的所有节点
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

        # 转换为可训练参数
        return nn.Parameter(torch.tensor(mask_init_values, dtype=torch.float32))

    def apply_discretization(self, M_cont: torch.Tensor) -> torch.Tensor:
        """
        应用三值离散化：-1(删除), 0(不变), +1(添加)
        使用直通梯度估计器保持可微性
        """
        # 计算离散值
        with torch.no_grad():
            discrete_M = torch.where(
                M_cont > self.tau_plus,
                torch.ones_like(M_cont),
                torch.where(
                    M_cont < self.tau_minus,
                    -torch.ones_like(M_cont),
                    torch.zeros_like(M_cont)
                )
            )

        # 应用TopK稀疏化 (仅保留梯度最大的k个扰动)
        abs_values = torch.abs(M_cont)
        if len(M_cont) > self.top_k:
            topk_indices = torch.topk(abs_values, self.top_k).indices
            sparse_mask = torch.zeros_like(M_cont)
            sparse_mask[topk_indices] = 1
            discrete_M = discrete_M * sparse_mask

        # 直通梯度估计
        return discrete_M + (M_cont - M_cont.detach())

    def forward(self) -> torch.Tensor:
        """生成扰动后的邻接矩阵"""
        # 应用tanh约束到[-1, 1]
        M_cont = torch.tanh(self.M)

        # 应用离散化处理
        M_disc = self.apply_discretization(M_cont)

        # 创建全尺寸掩码矩阵
        full_mask = torch.zeros_like(self.extended_sub_adj)

        # 填充掩码值到目标节点的所有可能边
        edge_idx = 0
        for i in range(self.n_nodes):
            if i != self.target_node:
                full_mask[self.target_node, i] = M_disc[edge_idx]
                full_mask[i, self.target_node] = M_disc[edge_idx]
                edge_idx += 1

        # 应用扰动 (删除: -1, 添加: +1)
        perturbed_adj = self.extended_sub_adj.clone()

        # 处理边删除 (M_disc = -1)
        del_mask = (full_mask < -0.5) & (self.extended_sub_adj > 0)
        perturbed_adj[del_mask] = 0

        # 处理边添加 (M_disc = +1)
        add_mask = (full_mask > 0.5) & (self.extended_sub_adj == 0)
        perturbed_adj[add_mask] = 1

        return perturbed_adj

    def get_perturbation_matrix(self) -> torch.Tensor:
        """获取ΔA矩阵 (-1, 0, 1)"""
        with torch.no_grad():
            M_cont = torch.tanh(self.M)
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
                 nfeat: int,
                 nhid: int,
                 nclass: int,
                 extended_sub_adj: torch.Tensor,
                 target_node: int,
                 original_adj_mask: torch.Tensor,
                 dropout: float = 0.5):
        super().__init__()
        self.perturb_layer = SignedMaskPerturbation(
            extended_sub_adj, target_node, original_adj_mask
        )

        # GCN层定义
        self.gc1 = nn.Linear(nfeat, nhid)
        self.gc2 = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.extended_sub_adj = extended_sub_adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 生成扰动后的邻接矩阵
        adj_perturbed = self.perturb_layer()

        # 添加自环并归一化
        adj_perturbed = adj_perturbed.fill_diagonal_(1)
        adj_normalized = self.normalize_adj(adj_perturbed)

        # GCN前向传播
        x = F.relu(self.gc1(torch.mm(adj_normalized, x)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(torch.mm(adj_normalized, x))

        return F.log_softmax(x, dim=1)

    def normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """对称归一化邻接矩阵"""
        rowsum = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    def get_mask_parameters(self) -> nn.Parameter:
        """获取可训练的掩码参数"""
        return self.perturb_layer.M