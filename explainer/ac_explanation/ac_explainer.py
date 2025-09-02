#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/8/27 20:16
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : ac_explainer.py
# @Software : PyCharm
# @Desc     :
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utilty.utils import normalize_adj


class ACExplainer:
    def __init__(self,
                 model: nn.Module,
                 extended_sub_adj: torch.Tensor,
                 sub_feat: torch.Tensor,
                 target_node: int,
                 lambda_pred: float = 1.0,
                 lambda_dist: float = 0.5,
                 lambda_plau: float = 0.2,
                 device: str = "cuda"):
        # 将模型和数据移动到指定设备
        self.model = model.to(device)
        self.original_adj = extended_sub_adj.to(device)
        self.features = sub_feat.to(device)
        self.target_node = target_node
        self.device = device

        # 损失权重
        self.lambda_pred = lambda_pred  # 预测损失权重
        self.lambda_dist = lambda_dist  # 稀疏损失权重
        self.lambda_plau = lambda_plau  # 现实性损失权重

        # 存储原始预测
        with torch.no_grad():
            # norm_sub_adj = normalize_adj(self.original_adj)
            # output = self.f(self.features, norm_sub_adj)
            output = self.model(self.features)
            # 获取目标节点的原始预测类别
            self.orig_pred = output.argmax(dim=1)[target_node].item()

        # 优化器设置
        self.optimizer = optim.Adam([self.model.get_mask_parameters()], lr=0.01)

    def compute_losses(self, output: torch.Tensor, delta_A: torch.Tensor) -> tuple:
        """计算多目标损失函数"""
        # 预测损失 (鼓励翻转预测)
        current_pred = output.argmax(dim=1)[self.target_node]
        # pred_loss = -F.nll_loss(
        #     output[self.target_node].unsqueeze(0),
        #     torch.tensor([self.orig_pred], device=self.device)
        # ) * (torch.argmax(output[self.target_node]) == torch.tensor([self.orig_pred], device=self.device)).float()
        pred_loss = -F.nll_loss(
            output[self.target_node].unsqueeze(0),  # shape: [1, n_class]
            torch.tensor([self.orig_pred], device=self.device)  # shape: [1]
        )

        # 稀疏损失 (L0范数)
        n_edits = torch.sum(torch.abs(delta_A) > 0.5).float()
        dist_loss = n_edits / len(self.model.get_mask_parameters())

        # 现实性损失
        plau_loss = self.compute_plausibility_loss(delta_A)

        # 加权总损失
        total_loss = (
                self.lambda_pred * pred_loss +
                self.lambda_dist * dist_loss +
                self.lambda_plau * plau_loss
        )

        return total_loss, pred_loss, dist_loss, plau_loss, current_pred.item()

    def compute_plausibility_loss(self, delta_A: torch.Tensor) -> torch.Tensor:
        """计算现实性损失"""
        loss_components = torch.tensor(0.0, device=self.device)

        # 1. 特征相似度惩罚 (仅对添加边)
        add_mask = (delta_A > 0.5)
        if add_mask.sum() > 0:
            # 计算特征相似度
            target_feat = self.features[self.target_node]

            for i in range(self.original_adj.size(0)):
                if add_mask[self.target_node, i]:
                    feat_sim = F.cosine_similarity(
                        target_feat.unsqueeze(0),
                        self.features[i].unsqueeze(0)
                    )
                    loss_components = loss_components + (1 - feat_sim) * 0.1  # α1 = 0.1

        # 2. 度分布惩罚
        orig_degrees = torch.sum(self.original_adj, dim=1)
        new_degrees = torch.sum(self.original_adj + delta_A, dim=1)
        deg_diff = torch.abs(new_degrees - orig_degrees) / (1 + orig_degrees)
        loss_components += torch.mean(deg_diff) * 0.2  # α2 = 0.2

        return loss_components

    def train_explanation(self, epochs: int = 100) -> dict:
        """训练解释器"""
        best_loss = float('inf')
        best_delta_A = None
        best_pred = None
        no_improve = 0

        # self.model.train()
        for epoch in range(epochs):
            print(f"\n######## epoch: {epoch + 1} #############")
            self.optimizer.zero_grad()

            # 前向传播
            output = self.model(self.features)
            print("Output requires_grad:", output.requires_grad)  # 应该为 True
            print("Output grad_fn:", output.grad_fn)  # 不应为 None！
            if output.grad_fn is None:
                print("严重错误: 模型的前向传播未生成计算图！")
                # 进一步检查输入特征是否需要梯度（通常不需要，但模型参数需要）
                print("Features requires_grad:", self.features.requires_grad)
                # 检查扰动参数
                mask_params = self.model.get_mask_parameters()
                print("Mask params requires_grad:", mask_params.requires_grad)
                print("Mask params grad_fn:", mask_params.grad_fn)
            delta_A = self.model.perturb_layer.get_perturbation_matrix()

            # 计算损失
            total_loss, pred_loss, dist_loss, plau_loss, current_pred = self.compute_losses(output, delta_A)
            if total_loss.grad_fn is None:
                print("警告: total_loss 没有梯度函数，计算图可能已中断!")
                # 这里可以添加更详细的调试信息
                print(f"pred_loss.requires_grad: {pred_loss.requires_grad}")
                print(f"dist_loss.requires_grad: {dist_loss.requires_grad}")
                print(f"plau_loss.requires_grad: {plau_loss.requires_grad}")

            # 调试：检查梯度信息
            print(f"total_loss.requires_grad: {total_loss.requires_grad}")
            print(f"total_loss.grad_fn: {total_loss.grad_fn}")

            # 反向传播
            total_loss.backward()

            # 检查掩码参数是否有梯度
            mask_grad = self.model.get_mask_parameters().grad
            print(f"梯度范数: {torch.norm(mask_grad).item() if mask_grad is not None else '无梯度'}")

            self.optimizer.step()

            print('Node idx: {}'.format(self.target_node),
                  'Epoch: {:04d}'.format(epoch + 1),
                  'loss: {:.4f}'.format(total_loss.item()),
                  'pred loss: {:.4f}'.format(pred_loss.item()),
                  'dist loss: {:.4f}'.format(dist_loss.item()),
                  'plau loss: {:.4f}'.format(plau_loss.item()), )
            print('orig pred: {}, new pred nondiff: {}'.format(self.orig_pred, output.argmax(dim=1)[self.target_node]))

            # 早停检查
            if current_pred != self.orig_pred and total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_delta_A = delta_A.detach().clone()
                best_pred = current_pred
                no_improve = 0
            elif current_pred != self.orig_pred:
                no_improve += 1

            if no_improve > 20:  # 提前停止
                break

        # 返回结果
        if best_delta_A is not None:
            return {
                "delta_A": best_delta_A,
                "original_pred": self.orig_pred,
                "new_pred": best_pred
            }
        return None

    def minimality_pruning(self, delta_A: torch.Tensor) -> torch.Tensor:
        """最小化剪枝算法"""
        current_delta = delta_A.clone()

        # 获取所有扰动边
        edge_indices = torch.nonzero(current_delta != 0)
        if len(edge_indices) == 0:
            return current_delta

        # 按梯度敏感度排序
        importance_scores = self.compute_edge_importance(current_delta)
        sorted_indices = torch.argsort(importance_scores.flatten())

        # 迭代移除边
        for idx in sorted_indices:
            i, j = edge_indices[idx]

            # 临时移除该边
            temp_delta = current_delta.clone()
            temp_delta[i, j] = 0
            temp_delta[j, i] = 0

            # 检查预测是否仍翻转
            with torch.no_grad():
                temp_adj = self.original_adj + temp_delta
                temp_adj = temp_adj.fill_diagonal_(1)
                temp_normalized = self.model.normalize_adj(temp_adj)

                # 手动计算GCN输出
                x = F.relu(self.model.gc1(torch.mm(temp_normalized, self.features)))
                x = F.dropout(x, self.model.dropout, training=False)
                x = self.model.gc2(torch.mm(temp_normalized, x))
                temp_output = F.log_softmax(x, dim=1)
                temp_pred = temp_output.argmax(dim=1)[self.target_node].item()

                if temp_pred != self.orig_pred:
                    current_delta = temp_delta

        return current_delta

    def compute_edge_importance(self, delta_A: torch.Tensor) -> torch.Tensor:
        """计算边重要性分数 (基于梯度敏感度)"""
        self.model.zero_grad()

        # 计算预测损失
        output = self.model(self.features)
        # pred_loss = -F.nll_loss(
        #     output[self.target_node].unsqueeze(0),
        #     torch.tensor(self.orig_pred, device=self.device)
        # ) * (output[self.target_node].unsqueeze(0) == self.orig_pred).float()
        pred_loss = -F.nll_loss(
            output[self.target_node].unsqueeze(0),
            torch.tensor([self.orig_pred], device=self.device)
        )

        # 反向传播获取梯度
        pred_loss.backward()
        mask_grad = self.model.get_mask_parameters().grad.abs()

        # 重建全尺寸梯度矩阵
        grad_matrix = torch.zeros_like(delta_A)
        edge_idx = 0
        for i in range(self.original_adj.size(0)):
            if i != self.target_node:
                grad_matrix[self.target_node, i] = mask_grad[edge_idx]
                grad_matrix[i, self.target_node] = mask_grad[edge_idx]
                edge_idx += 1

        return grad_matrix
