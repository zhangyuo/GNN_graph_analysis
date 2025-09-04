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
from torch.nn.utils import clip_grad_norm_

from utilty.utils import normalize_adj
from explainer.ac_explanation.gcn_perturb import GNNPerturb


class ACExplainer:
    def __init__(self,
                 model: nn.Module,
                 target_node: int,
                 node_idx: int,
                 extended_sub_adj: torch.Tensor,
                 sub_feat: torch.Tensor,
                 sub_labels: torch.Tensor,
                 y_pred_orig: torch.Tensor,
                 nclass: int,
                 nhid: int = 16,
                 dropout: float = 0.5,
                 lambda_pred: float = 1.0,
                 lambda_dist: float = 0.5,
                 lambda_plau: float = 0.2,
                 epoch: int = 200,
                 optimizer: str = 'SGD',
                 n_momentum: float = 0.9,
                 lr: float = 0.01,
                 top_k: int = 5,
                 tau_plus: float = 0.5,
                 tau_minus: float = -0.5,
                 α1: float = 0.1,
                 α2: float = 0.1,
                 α3: float = 0.1,
                 α4: float = 0.5,
                 tau_c: float = 0.1,
                 device: str = "cuda",
                 gcn_layer: int = 2,
                 with_bias: bool = True):
        # 将模型和数据移动到指定设备
        self.model = model.to(device)
        self.model.eval()
        self.extended_sub_adj = extended_sub_adj.to(device)
        self.sub_feat = sub_feat.to(device)

        self.n_hid = nhid
        self.dropout = dropout
        self.sub_labels = sub_labels
        self.y_pred_orig = y_pred_orig

        # 损失权重
        self.lambda_pred = lambda_pred  # 预测损失权重
        self.lambda_dist = lambda_dist  # 稀疏损失权重
        self.lambda_plau = lambda_plau  # 现实性损失权重

        self.num_classes = nclass
        self.target_node = target_node
        self.node_idx = node_idx
        self.epoch = epoch
        self.optimizer_type = optimizer
        self.n_momentum = n_momentum
        self.lr = lr
        self.top_k = top_k
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.α1 = α1
        self.α2 = α2
        self.α3 = α3
        self.α4 = α4
        self.tau_c = tau_c
        self.device = device
        self.gcn_layer = gcn_layer
        self.with_bias = with_bias

        # 创建扰动模型
        self.cf_model = GNNPerturb(
            nfeat=self.sub_feat.size(1),
            nhid=self.n_hid,
            nclass=self.num_classes,
            extended_sub_adj=self.extended_sub_adj,
            sub_feat=self.sub_feat,
            node_idx=self.node_idx,
            dropout=dropout,
            lambda_pred=self.lambda_pred,
            lambda_dist=self.lambda_dist,
            lambda_plau=self.lambda_plau,
            top_k=self.top_k,
            tau_plus=self.tau_plus,
            tau_minus=self.tau_minus,
            α1=self.α1,
            α2=self.α2,
            α3=self.α3,
            α4=self.α4,
            tau_c=self.tau_c,
            gcn_layer=gcn_layer,
            with_bias=with_bias
        ).to(device)

        # 继承原模型参数
        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

        # Freeze weights from original model in cf_model 冻结原始参数，仅训练扰动矩阵
        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            print("orig model requires_grad: ", name, param.requires_grad)
        for name, param in self.cf_model.named_parameters():
            print("cf model requires_grad: ", name, param.requires_grad)

        # 优化器设置
        if self.optimizer_type == "SGD" and self.n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=self.lr)
        elif self.optimizer_type == "SGD" and self.n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=self.lr, nesterov=True,
                                          momentum=self.n_momentum)
        elif self.optimizer_type == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=self.lr)
        elif self.optimizer_type == "Adam":
            self.cf_optimizer = optim.Adam([self.cf_model.get_mask_parameters()], lr=self.lr)

    def explain(self) -> dict:
        """训练解释器"""
        best_loss = float('inf')
        best_delta_A = None
        best_pred = None
        best_cf_adj = None
        no_improve = 0

        self.cf_model.eval()  # 反事实模型g训练阶段采用评估模式，冻结dropout和batchnorm

        for epoch in range(self.epoch):
            # print(f"\n######## epoch: {epoch + 1} #############")
            self.cf_optimizer.zero_grad()

            # 前向传播
            output = self.cf_model.forward(self.sub_feat, self.extended_sub_adj)  # 可微分预测
            output_actual = self.cf_model.forward_prediction(self.sub_feat)  # 离散预测

            y_pred_new = torch.argmax(output[self.node_idx])
            y_pred_new_actual = torch.argmax(output_actual[self.node_idx])

            # 计算损失
            total_loss, pred_loss, dist_loss, plau_loss, cf_adj, delta_A = self.cf_model.compute_losses(output,
                                                                                                        self.y_pred_orig,
                                                                                                        y_pred_new_actual)

            # 反向传播
            total_loss.backward()
            clip_grad_norm_(self.cf_model.parameters(), 2.0)  # 裁剪梯度幅度
            self.cf_optimizer.step()

            print('Target node: {}'.format(self.target_node),
                  'New idx: {}'.format(self.node_idx),
                  'Epoch: {:04d}'.format(epoch + 1),
                  'loss: {:.4f}'.format(total_loss.item()),
                  'pred loss: {:.4f}'.format(pred_loss.item()),
                  'dist loss: {:.4f}'.format(dist_loss.item()),
                  'plau loss: {:.4f}'.format(plau_loss.item()))
            print('Output: {}\n'.format(output[self.node_idx].data),
                  'Output nondiff: {}\n'.format(output_actual[self.node_idx].data),
                  'orig pred: {}, new pred: {}, new pred nondiff: {}'.format(self.y_pred_orig, y_pred_new,
                                                                             y_pred_new_actual))
            print(" ")
            # 早停检查
            if y_pred_new_actual != self.y_pred_orig and total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_delta_A = delta_A.detach().clone()
                best_pred = y_pred_new_actual
                best_cf_adj = cf_adj
                no_improve = 0
            elif y_pred_new_actual != self.y_pred_orig:
                no_improve += 1

            if no_improve > 20:  # 提前停止
                break

        # 返回结果
        if best_delta_A is not None:
            return {
                "delta_A": best_delta_A,
                "cf_adj": best_cf_adj,
                "original_pred": self.y_pred_orig,
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
                x = F.relu(self.model.gc1(torch.mm(temp_normalized, self.sub_feat)))
                x = F.dropout(x, self.model.dropout, training=False)
                x = self.model.gc2(torch.mm(temp_normalized, x))
                temp_output = F.log_softmax(x, dim=1)
                temp_pred = temp_output.argmax(dim=1)[self.node_idx].item()

                if temp_pred != self.y_pred_orig:
                    current_delta = temp_delta

        return current_delta

    def compute_edge_importance(self, delta_A: torch.Tensor) -> torch.Tensor:
        """计算边重要性分数 (基于梯度敏感度)"""
        self.model.zero_grad()

        # 计算预测损失
        output = self.model(self.sub_feat)
        # pred_loss = -F.nll_loss(
        #     output[self.node_idx].unsqueeze(0),
        #     torch.tensor(self.y_pred_orig, device=self.device)
        # ) * (output[self.node_idx].unsqueeze(0) == self.y_pred_orig).float()
        pred_loss = -F.nll_loss(
            output[self.node_idx].unsqueeze(0),
            torch.tensor([self.y_pred_orig], device=self.device)
        )

        # 反向传播获取梯度
        pred_loss.backward()
        mask_grad = self.model.get_mask_parameters().grad.abs()

        # 重建全尺寸梯度矩阵
        grad_matrix = torch.zeros_like(delta_A)
        edge_idx = 0
        for i in range(self.original_adj.size(0)):
            if i != self.node_idx:
                grad_matrix[self.node_idx, i] = mask_grad[edge_idx]
                grad_matrix[i, self.node_idx] = mask_grad[edge_idx]
                edge_idx += 1

        return grad_matrix
