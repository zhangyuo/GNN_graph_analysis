#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/9/17 10:53
# @Author   : **
# @Email    : **@**
# @File     : GraphConv.py
# @Software : PyCharm
# @Desc     : GraphConv based GNN model with edge_weight support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.utils import dense_to_sparse

from utilty.utils import normalize_adj


class GraphConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, lr=0.01, weight_decay=5e-4, device=None):
        super(GraphConvNet, self).__init__()

        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        # 第一层
        self.layers.append(GraphConv(in_channels, hidden_channels, aggr="add"))

        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(GraphConv(hidden_channels, hidden_channels, aggr="add"))

        # 最后一层
        self.layers.append(GraphConv(hidden_channels, out_channels, aggr="add"))

        # 设备选择
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # 最后一层
        x = self.layers[-1](x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)

    def train_step(self, data):
        """single step training"""
        self.train()
        self.optimizer.zero_grad()
        dense_adj = torch.tensor(data.adj.toarray(), device=self.device)
        norm_adj = normalize_adj(dense_adj)
        edge_index, edge_weight = dense_to_sparse(norm_adj)
        features = torch.tensor(data.features.toarray(), device=self.device)
        out = self(features, edge_index, edge_weight=edge_weight)
        loss = self.loss(out[data.idx_train], torch.tensor(data.labels, device=self.device).long()[data.idx_train])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test_step(self, data):
        """test，return (train_acc, val_acc, test_acc)"""
        self.eval()
        dense_adj = torch.tensor(data.adj.toarray(), device=self.device)
        norm_adj = normalize_adj(dense_adj)
        edge_index, edge_weight = dense_to_sparse(norm_adj)
        features = torch.tensor(data.features.toarray(), device=self.device)
        out = self(features, edge_index, edge_weight=edge_weight)
        pred = out.argmax(dim=1)

        accs = []
        labels = torch.tensor(data.labels, device=self.device).long()
        for mask in [data.idx_train, data.idx_val, data.idx_test]:
            correct = pred[mask].eq(labels[mask]).sum().item()
            accs.append(correct / len(mask))
        return accs

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

    def fit(self, data, train_iters=500):
        best_val_acc = 0.0
        best_model_state = None
        # 训练循环
        for epoch in range(train_iters):
            loss = self.train_step(data)
            train_acc, val_acc, test_acc = self.test_step(data)
            if epoch % 10 == 0:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.state_dict().copy()
                print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                      f"Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}")

        if best_model_state:
            self.load_state_dict(best_model_state)
            print(f"Loaded best model with Val Acc {best_val_acc:.4f}")


def GraphConv_initilization(data, hidden_channels, dropout, lr, weight_decay, num_layers, device):
    target_gnn = GraphConvNet(
        in_channels=data.features.shape[1],
        hidden_channels=hidden_channels,
        out_channels=data.labels.max().item() + 1,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        device=device
    )
    return target_gnn


def GraphConv_model(data, hidden_channels, dropout, lr, weight_decay, num_layers, epoch, device, target_gnn=None):
    """
    GraphConv model
    """
    if target_gnn is None:
        target_gnn = GraphConv_initilization(data, hidden_channels, dropout, lr, weight_decay, num_layers, device)
        target_gnn = target_gnn.to(device)
        target_gnn.train()
        target_gnn.fit(data, epoch)
        target_gnn.eval()
        features = torch.tensor(data.features.toarray(), device=device)
        dense_adj = torch.tensor(data.adj.toarray(), device=device)
        norm_adj = normalize_adj(dense_adj)
        edge_index, edge_weight = dense_to_sparse(norm_adj)
        output = target_gnn.forward(features, edge_index, edge_weight=edge_weight)
    else:
        output = None

    return target_gnn, output


def load_GraphConv_model(file_path, data, hidden_channels, dropout, device, lr, weight_decay, num_layers):
    gnn_model = GraphConv_initilization(data, hidden_channels, dropout, lr, weight_decay, num_layers, device)
    gnn_model.load_state_dict(torch.load(file_path))
    gnn_model.eval()
    return gnn_model
