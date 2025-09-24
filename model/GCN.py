#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/20 15:22
# @Author   : **
# @Email    : **@**
# @File     : GCN.py
# @Software : PyCharm
# @Desc     :
"""
import math
import os
import sys
import time

import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.nn.parameter import Parameter
from scipy.sparse import issparse, csr_matrix
import scipy.sparse as sp
from deeprobust.graph.data import Dataset
import numpy as np
from torch_geometric.utils import k_hop_subgraph, add_self_loops, subgraph, get_laplacian
import torch.nn.functional as F
from deeprobust.graph.defense import GCN, GraphConvolution
from config.config import *
from utilty.utils import normalize_adj, get_neighbourhood


class GCN_extend(GCN):
    def __init__(self, nfeat, nhid, nclass, dropout, device, lr, weight_decay, with_bias=True, gcn_layer=2):
        super().__init__(nfeat=nfeat,
                         nhid=nhid,
                         nclass=nclass,
                         dropout=dropout,
                         device=device,
                         lr=lr,
                         weight_decay=weight_decay,
                         with_bias=with_bias)
        if gcn_layer == 3:
            self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
            self.gc2 = GraphConvolution(nhid, nhid, with_bias=with_bias)
            self.gc3 = GraphConvolution(nhid, nclass, with_bias=with_bias)
            self.lin = nn.Linear(nhid + nhid + nclass, nclass, bias=with_bias)  # 拼接多层输出
        else:
            self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
            self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.gcn_layer=gcn_layer

    def forward(self, x, adj):
        if self.gcn_layer == 3:
            x1 = F.relu(self.gc1(x, adj))  # 第一层卷积+激活
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x2 = F.relu(self.gc2(x1, adj))  # 第二层卷积+激活
            x2 = F.dropout(x2, self.dropout, training=self.training)
            x3 = self.gc3(x2, adj)  # 第三层无激活
            x = self.lin(torch.cat((x1, x2, x3), dim=1))  # 特征拼接
            return F.log_softmax(x, dim=1)  # 分类输出，输出层使用log_softmax适配NLLLoss
        else:
            x1 = F.relu(self.gc1(x, adj))  # 第一层卷积+激活
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x2 = self.gc2(x1, adj)  # 第二层卷积
            return F.log_softmax(x2, dim=1)  # 分类输出，输出层使用log_softmax适配NLLLoss

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

    def fit_extended(self, features, adj, labels, idx_train, idx_val, train_iters=GCN_EPOCHS, patience=500):
        """扩展训练方法（使用自定义优化器）"""
        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # # 添加学习率调度器
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='max',
        #     factor=0.5,
        #     patience=50,
        #     verbose=True
        # )

        best_val_acc = 0.0
        best_model_state = None

        # 训练循环
        for epoch in range(train_iters):
            optimizer.zero_grad()
            output = self(features, adj)
            loss = self.loss(output[idx_train], labels[idx_train])
            loss.backward()
            optimizer.step()

            # 每10轮验证一次
            if epoch % 10 == 0:
                self.eval()
                with torch.no_grad():
                    output_val = self(features, adj)
                    val_loss = self.loss(output_val[idx_val], labels[idx_val])
                    preds = output_val[idx_val].argmax(dim=1)
                    val_acc = (preds == labels[idx_val]).float().mean().item()

                    # 更新最佳模型
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model_state = self.state_dict().copy()

                    print(
                        f"Epoch {epoch}: Train Loss {loss.item():.4f}, Val Loss {val_loss.item():.4f}, Val Acc {val_acc:.4f}")
                    # scheduler.step(val_acc)  # 根据验证准确率调整学习率
                self.train()

        # 加载最佳模型状态
        if best_model_state:
            self.load_state_dict(best_model_state)
            print(f"Loaded best model with Val Acc {best_val_acc:.4f}")


def GCN_initilization(features, labels, nhid, dropout, device, lr, weight_decay, with_bias, gcn_layer):
    target_gcn = GCN_extend(nfeat=features.shape[1],
                            nhid=nhid,
                            nclass=labels.max().item() + 1,
                            dropout=dropout,
                            device=device,
                            lr=lr,
                            weight_decay=weight_decay,
                            with_bias=with_bias,
                            gcn_layer=gcn_layer
                            )
    return target_gcn


def GCN_model(adj, features, labels, device, idx_train, idx_val, nhid, dropout, lr, weight_decay, with_bias,
              gcn_layer, target_gcn=None):
    """
    GCN model
    """
    output = None
    if target_gcn is None:
        target_gcn = GCN_initilization(features, labels, nhid, dropout, device, lr, weight_decay, with_bias, gcn_layer)
        target_gcn = target_gcn.to(device)

        dense_adj = torch.tensor(adj.toarray())
        norm_adj_1 = normalize_adj(dense_adj)
        # norm_adj = csr_matrix(norm_adj_1)

        # scaler = StandardScaler()
        # features_1 = features.toarray()
        # features_1 = scaler.fit_transform(features_1)
        features_1 = torch.tensor(features.toarray())

        labels_1 = torch.tensor(labels).long()

        target_gcn.train()
        # target_gcn.fit(features, norm_adj, labels, idx_train, idx_val, normalize=normalize, patience=500)
        target_gcn.fit_extended(features_1, norm_adj_1, labels_1, idx_train, idx_val)

        target_gcn.eval()
        # output = target_gcn.predict()
        output = target_gcn.forward(features_1, norm_adj_1)
    else:
        pass

    return target_gcn, output


def load_GCN_model(file_path, features, labels, nhid, dropout, device, lr, weight_decay, with_bias, gcn_layer):
    gnn_model = GCN_initilization(features, labels, nhid, dropout, device, lr, weight_decay, with_bias, gcn_layer)
    gnn_model.load_state_dict(torch.load(file_path))
    gnn_model.eval()
    return gnn_model


class PyGCompatibleGCN(nn.Module):
    """
    2-layer GCN used in GNN Explainer for cora tasks
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout, device, gcn_layer=2, with_bias=True):
        super().__init__()
        self.gcn_layer = gcn_layer
        if self.gcn_layer == 3:
            self.conv1 = GCNConv(in_channels, hidden_channels, bias=with_bias)
            self.conv2 = GCNConv(hidden_channels, hidden_channels, bias=with_bias)
            self.conv3 = GCNConv(hidden_channels, out_channels, bias=with_bias)
            self.lin = nn.Linear(hidden_channels + hidden_channels + out_channels, out_channels, bias=with_bias)
        else:
            self.conv1 = GCNConv(in_channels, hidden_channels, bias=with_bias)
            self.conv2 = GCNConv(hidden_channels, out_channels, bias=with_bias)
        self.dropout = dropout
        self.device = device

    def forward(self, x, edge_index):
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if self.gcn_layer==3:
            x1 = self.conv1(x, edge_index).relu()
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = self.conv2(x1, edge_index).relu()
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
            x3 = self.conv3(x2, edge_index)
            x = self.lin(torch.cat((x1, x2, x3), dim=1))
            return F.log_softmax(x, dim=1)
        else:
            x1 = self.conv1(x, edge_index).relu()
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = self.conv2(x1, edge_index)
            return F.log_softmax(x2, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

    def node_embeddings(self, sub_x, sub_edge_index):
        # edge_index, _ = add_self_loops(sub_edge_index, num_nodes=sub_x.size(0))
        x = self.conv1(sub_x, sub_edge_index).relu()
        node_embeddings = self.conv2(x, sub_edge_index)
        return node_embeddings

    def graph_embeddings(self, node_embeddings):
        emb = torch.mean(node_embeddings, dim=0)
        # emb = emb / torch.norm(emb)  # Key modification: L2 normalization
        return emb


def transfer_weights(dr_model, pyg_model, gcn_layer):
    print("PyG模型结构验证:")
    print(f"conv1.lin存在: {hasattr(pyg_model.conv1, 'lin')}")  # 应为True
    print(f"conv1.lin.weight形状: {pyg_model.conv1.lin.weight.shape}")
    print(f"DeepRobust gc1.weight形状: {dr_model.gc1.weight.shape}")

    # 第一层权重转置 (1433,16) -> (16,1433)
    pyg_model.conv1.lin.weight.data = dr_model.gc1.weight.data.t().clone()
    # pyg_model.conv1.lin.weight.data.copy_(dr_model.gc1.weight.data)
    pyg_model.conv1.bias.data.copy_(dr_model.gc1.bias.data)

    # 第二层权重转置 (16,16) -> (16,16)
    pyg_model.conv2.lin.weight.data = dr_model.gc2.weight.data.t().clone()
    # pyg_model.conv2.lin.weight.data.copy_(dr_model.gc2.weight.data)
    pyg_model.conv2.bias.data.copy_(dr_model.gc2.bias.data)

    if gcn_layer == 3:
        # 第三层权重转置 (16,7) -> (7,16)
        pyg_model.conv3.lin.weight.data = dr_model.gc3.weight.data.t().clone()
        # pyg_model.conv2.lin.weight.data.copy_(dr_model.gc2.weight.data)
        pyg_model.conv3.bias.data.copy_(dr_model.gc3.bias.data)

        # 线性拼接层权重转换
        pyg_model.lin.weight.data.copy_(dr_model.lin.weight.data)
        pyg_model.lin.bias.data.copy_(dr_model.lin.bias.data)
    else:
        pass

    return pyg_model


def GCNtoPYG(gnn_model, device, features, labels, gcn_layer):
    # initialize pyg gcn model
    pyg_gcn = PyGCompatibleGCN(
        in_channels=features.shape[1],
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=labels.max().item() + 1,
        dropout=DROPOUT,
        with_bias=WITH_BIAS,
        device=DEVICE,
        gcn_layer=gcn_layer
    )
    pyg_gcn = pyg_gcn.to(device)

    # Initialize model (using deeprobust adjacency matrix)
    dr_trained_model = gnn_model
    pyg_gcn = transfer_weights(dr_trained_model, pyg_gcn, gcn_layer)
    pyg_gcn.eval()
    return pyg_gcn


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


def dr_data_to_pyg_data_mask(adj, features, labels, idx_train, idx_val, idx_test):
    """
    transfer deeprobust data to pyg data
    :return:
    """
    features_dense = features.toarray() if issparse(features) else features

    # creat bool mask
    num_nodes = features.shape[0]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    pyg_data = Data(
        x=torch.tensor(features_dense, dtype=torch.float),
        edge_index=adj_to_edge_index(adj),
        # adj=torch.tensor(adj.toarray(), dtype=torch.float) if str(type(adj)) != "<class 'torch.Tensor'>" else adj,
        y=torch.tensor(labels),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    return pyg_data


if __name__ == '__main__':
    # Set up paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, base_path)

    # Load deeprobust data
    device = DEVICE
    dataset_name = DATA_NAME
    data_robust = Dataset(root=base_path + '/dataset', name=dataset_name)
    adj, features, labels = data_robust.adj, data_robust.features, data_robust.labels
    idx_train, idx_val, idx_test = data_robust.idx_train, data_robust.idx_val, data_robust.idx_test

    # 初始化PyG模型
    print(f"特征矩阵类型: {type(features)}")  # 应为scipy.sparse.csr.csr_matrix
    print(f"邻接矩阵类型: {type(adj)}")  # 应为scipy.sparse.csr.csr_matrix

    gnn_model, output = GCN_model(adj, features, labels, device, idx_train, idx_val)

    # Create PyG Data object
    pyg_data = dr_data_to_pyg_data(adj, features, labels)

    # initialize pyg model
    pyg_gcn = GCNtoPYG(gnn_model, device, features, labels)

    # 预测一致性检查
    # dr_pred = dr_trained_model.predict(features, adj)
    dr_logits = torch.tensor(output, device=device)  # 确保同设备
    dr_pred = dr_logits.argmax(dim=1)

    pyg_gcn.eval()
    pyg_logits = pyg_gcn.forward(pyg_data.x, pyg_data.edge_index)
    pyg_pred = pyg_logits.argmax(dim=1)

    accuracy = (dr_pred == pyg_pred).float().mean()
    print(f"验证集预测一致性: {accuracy.item() * 100:.2f}%")

    # 构建子图向量
    target_node = 5
    sub_adj, sub_edge_index, sub_feat, sub_labels, node_dict = get_neighbourhood(target_node, pyg_data.edge_index,
                                                                                 features, labels,GCN_LAYER)
    new_idx = node_dict[target_node]
    print("Output original model, full adj: {}".format(output[target_node]))
    norm_sub_adj = normalize_adj(sub_adj)
    print("Output original model, sub adj: {}".format(gnn_model.forward(sub_feat, norm_sub_adj)[new_idx]))

    # target_nodes = [5, 6, 7, 8]
    # sub_x = pyg_data.x = pyg_data.x[target_nodes]
    # sub_edge_index, _ = subgraph(
    #     subset=target_nodes,
    #     edge_index=pyg_data.edge_index,
    #     relabel_nodes=True  # 关键：重映射节点ID[9](@ref)
    # )
    # node_embeddings = pyg_gcn.node_embeddings(sub_x, sub_edge_index)
    # graph_embeddings = pyg_gcn.graph_embeddings(node_embeddings)

    # # Create explainer (using PyG-formatted data)
    # explainer = Explainer(
    #     model=pyg_gcn,
    #     algorithm=GNNExplainer(
    #         epochs=100,  # 减少训练轮次
    #         lr=0.1,  # 提高学习率
    #         log=False,  # 禁用日志
    #         coeffs={'edge_size': 0.005, 'node_feat_size': 0.1}  # 添加正则化防止梯度爆炸
    #     ),
    #     explanation_type='model',
    #     node_mask_type='attributes',
    #     edge_mask_type='object',
    #     model_config=dict(
    #         mode='multiclass_classification',
    #         task_level='node',
    #         return_type='log_probs'
    #     )
    # )
    #
    # start_time = time.time()
    # target_node = 1544
    # # generate explanation for target node from specified explainer
    # subset, edge_index_sub, mapping, _ = k_hop_subgraph(
    #     node_idx=target_node,
    #     num_hops=3,
    #     edge_index=pyg_data.edge_index,
    #     relabel_nodes=True,
    #     num_nodes=pyg_data.num_nodes
    # )
    #
    # # 创建子图特征
    # x_sub = pyg_data.x[subset]
    #
    # # 执行解释
    # explanation = explainer(
    #     x=x_sub,
    #     edge_index=edge_index_sub,
    #     index=mapping[0]
    # )
    #
    # # explanation = explainer(
    # #     x=pyg_data.x,
    # #     edge_index=pyg_data.edge_index,
    # #     index=1544
    # # )
    #
    # # get mask of edges an nodes
    # edge_mask = explanation.edge_mask
    # node_mask = explanation.node_mask
    #
    # elapsed = time.time() - start_time
    # print(f"explainer generates results in {elapsed:.4f}s!")
