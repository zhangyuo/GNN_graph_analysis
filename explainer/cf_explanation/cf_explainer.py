# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

from utilty.utils import get_degree_matrix, normalize_adj
from explainer.cf_explanation.gcn_perturb import GCNCoraPerturb


class CFExplainer:
    """
    CF Explainer class, returns counterfactual subgraph
    """

    def __init__(self, model, sub_adj, sub_feat, n_hid, dropout,
                 sub_labels, y_pred_orig, num_classes, beta, device, gcn_layer, with_bias, test_model, dataset_name, heads):
        super(CFExplainer, self).__init__()
        self.model = model
        self.model.eval()
        self.sub_adj = sub_adj
        self.sub_feat = sub_feat
        self.n_hid = n_hid
        self.dropout = dropout
        self.sub_labels = sub_labels
        self.y_pred_orig = y_pred_orig
        self.beta = beta
        self.num_classes = num_classes
        self.device = device
        self.gcn_layer = gcn_layer
        self.with_bias = with_bias
        self.heads = heads
        self.test_model = test_model
        self.dataset_name=dataset_name

        # Instantiate CF model class, load weights from original model
        self.cf_model = GCNCoraPerturb(self.sub_feat.shape[1], n_hid, self.num_classes, self.sub_adj, dropout, beta,
                                       gcn_layer, with_bias=with_bias, test_model=test_model, dataset_name=dataset_name,heads=heads)  # 加载可扰动模型

        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)  # 继承原模型参数

        # Freeze weights from original model in cf_model 冻结原始参数，仅训练扰动矩阵
        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias") or name.endswith("att_src") or name.endswith(
                    "att_dst") or name.endswith("att_edge"):
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            print("orig model requires_grad: ", name, param.requires_grad)
        for name, param in self.cf_model.named_parameters():
            print("cf model requires_grad: ", name, param.requires_grad)

    def explain(self, cf_optimizer, node_idx, new_idx, lr, n_momentum, num_epochs):
        self.node_idx = node_idx
        self.new_idx = new_idx

        self.x = self.sub_feat
        self.A_x = self.sub_adj
        self.D_x = get_degree_matrix(self.A_x)

        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum)
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)

        best_cf_example = None
        best_loss = np.inf
        num_cf_examples = 0
        best_cf_example_1 = None
        best_loss_1 = np.inf
        for epoch in tqdm(range(num_epochs)):
            new_example, loss_total = self.train(epoch)
            if new_example[-1] and loss_total < best_loss:
                # success
                best_cf_example = new_example
                best_loss = loss_total
                num_cf_examples += 1
            elif not new_example[-1] and loss_total < best_loss_1:
                # failed
                best_cf_example_1 = new_example
                best_loss_1 = loss_total
        print("{} CF examples for node_idx = {}".format(num_cf_examples, self.node_idx))
        print(" ")
        if num_cf_examples > 0:
            return best_cf_example
        else:
            return best_cf_example_1

    def train(self, epoch):
        t = time.time()
        self.cf_model.eval()  # 反事实模型g训练阶段采用评估模式，冻结dropout和batchnorm
        self.cf_optimizer.zero_grad()  # 清空上一轮的梯度（避免累积）

        # output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        # output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
        # 前向传播（包含连续和二值化预测）
        output = self.cf_model.forward(self.x, self.A_x)  # 可微分预测
        output_actual, self.P = self.cf_model.forward_prediction(self.x)  # 离散预测

        # Need to use new_idx from now on since sub_adj is reindexed
        y_pred_new = torch.argmax(output[self.new_idx])
        y_pred_new_actual = torch.argmax(output_actual[self.new_idx])

        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(output[self.new_idx], self.y_pred_orig,
                                                                            y_pred_new_actual)  # 计算复合损失
        loss_total.backward()  # 计算当前梯度

        # 检查掩码参数梯度
        # print("P_vec.grad:", self.cf_model.get_mask_parameters().grad)
        if self.cf_model.get_mask_parameters().grad is not None:
            print(f"Mask grad norm: {self.cf_model.get_mask_parameters().grad.norm().item()}")
        else:
            print("Mask grad is None")

        clip_grad_norm(self.cf_model.parameters(), 2.0)  # 裁剪梯度幅度
        self.cf_optimizer.step()  # 根据梯度更新参数
        print('Node idx: {}'.format(self.node_idx),
              'New idx: {}'.format(self.new_idx),
              'Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.4f}'.format(loss_total.item()),
              'pred loss: {:.4f}'.format(loss_pred.item()),
              'graph loss: {:.4f}'.format(loss_graph_dist.item()))
        print('Output: {}\n'.format(output[self.new_idx].data),
              'Output nondiff: {}\n'.format(output_actual[self.new_idx].data),
              'orig pred: {}, new pred: {}, new pred nondiff: {}'.format(self.y_pred_orig, y_pred_new,
                                                                         y_pred_new_actual))
        print(" ")
        if y_pred_new_actual != self.y_pred_orig:
            # header = ["node_idx", "new_idx", "cf_adj", "sub_adj", "y_pred_orig", "y_pred_new", "y_pred_new_actual",
            #             "label", "num_nodes", "loss_total", "loss_pred", "loss_graph_dist"]
            success = True
            cf_stats = [self.node_idx,
                        self.new_idx,
                        cf_adj.detach().numpy(),
                        self.sub_adj.detach().numpy(),
                        self.y_pred_orig.item(),
                        y_pred_new_actual.item(),
                        self.sub_labels,
                        loss_graph_dist.item(),
                        self.sub_feat,
                        success]
        else:
            success = False
            cf_stats = [self.node_idx,
                        self.new_idx,
                        cf_adj.detach().numpy(),
                        self.sub_adj.detach().numpy(),
                        self.y_pred_orig.item(),
                        y_pred_new_actual.item(),
                        self.sub_labels,
                        loss_graph_dist.item(),
                        self.sub_feat,
                        success]

        return (cf_stats, loss_total.item())
