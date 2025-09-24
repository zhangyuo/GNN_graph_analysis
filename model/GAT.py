#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/9/17 17:06
# @Author   : **
# @Email    : **@**
# @File     : GAT.py
# @Software : PyCharm
# @Desc     :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, DenseGATConv
from torch_geometric.utils import dense_to_sparse

from model.test_model import adj_to_edge_index
from utilty.utils import normalize_adj


class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, heads=2, dropout=0.5, lr=0.01, weight_decay=5e-4, device=None):
        super(GATNet, self).__init__()

        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        # 第一层
        self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=1))

        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, edge_dim=1))

        # 最后一层
        self.layers.append(
            GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout, edge_dim=1))

        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            edge_attr = edge_weight.view(-1, 1)  # [num_edges, 1]
        else:
            edge_attr = None

        for conv in self.layers[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)  # BA-SHAPES on GAT  use elu, other use relu
            x = F.dropout(x, p=self.dropout, training=self.training)
        # 最后一层
        x = self.layers[-1](x, edge_index, edge_attr=edge_attr)
        return F.log_softmax(x, dim=1)

    def train_step(self, data):
        """单步训练"""
        self.train()
        self.optimizer.zero_grad()

        dense_adj = torch.tensor(data.adj.toarray(), dtype=torch.float32, device=self.device)
        norm_adj = normalize_adj(dense_adj)
        edge_index, edge_weight = dense_to_sparse(norm_adj)
        edge_index, edge_weight = edge_index.to(self.device), edge_weight.to(self.device)

        features = torch.tensor(data.features.toarray(), dtype=torch.float32, device=self.device)
        labels = torch.tensor(data.labels, dtype=torch.long, device=self.device)

        out = self(features, edge_index, edge_weight=edge_weight)
        loss = self.loss(out[data.idx_train], labels[data.idx_train])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test_step(self, data):
        """测试，返回 (train_acc, val_acc, test_acc)"""
        self.eval()
        dense_adj = torch.tensor(data.adj.toarray(), dtype=torch.float32, device=self.device)
        norm_adj = normalize_adj(dense_adj)
        edge_index, edge_weight = dense_to_sparse(norm_adj)
        edge_index, edge_weight = edge_index.to(self.device), edge_weight.to(self.device)

        features = torch.tensor(data.features.toarray(), dtype=torch.float32, device=self.device)
        labels = torch.tensor(data.labels, dtype=torch.long, device=self.device)

        out = self(features, edge_index, edge_weight=edge_weight)
        pred = out.argmax(dim=1)

        accs = []
        for mask in [data.idx_train, data.idx_val, data.idx_test]:
            correct = pred[mask].eq(labels[mask]).sum().item()
            accs.append(correct / len(mask))
        return accs

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

    def fit(self, data, train_iters=500):
        best_val_acc = 0.0
        best_model_state = None
        print("Train label distribution:", torch.bincount(torch.tensor(data.labels)[data.idx_train]))
        print("Val label distribution:", torch.bincount(torch.tensor(data.labels)[data.idx_val]))
        print("Test label distribution:", torch.bincount(torch.tensor(data.labels)[data.idx_test]))
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


def GATNet_initialization(data, hidden_channels, dropout, lr, weight_decay, num_layers, heads_num, device):
    target_gnn = GATNet(
        in_channels=data.features.shape[1],
        hidden_channels=hidden_channels,
        out_channels=data.labels.max().item() + 1,
        num_layers=num_layers,
        heads=heads_num,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        device=device
    )
    return target_gnn


def GATNet_model(data, hidden_channels, dropout, lr, weight_decay, num_layers, heads_num, epoch, device,
                 target_gnn=None):
    if target_gnn is None:
        target_gnn = GATNet_initialization(data, hidden_channels, dropout, lr, weight_decay,
                                           num_layers, heads_num, device)
        target_gnn = target_gnn.to(device)
        target_gnn.train()
        target_gnn.fit(data, epoch)
        target_gnn.eval()

        features = torch.tensor(data.features.toarray(), dtype=torch.float32, device=device)
        dense_adj = torch.tensor(data.adj.toarray(), dtype=torch.float32, device=device)
        norm_adj = normalize_adj(dense_adj)
        edge_index, edge_weight = dense_to_sparse(norm_adj)
        edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)

        output = target_gnn.forward(features, edge_index, edge_weight=edge_weight)
        # output = target_gnn.forward(features, adj_to_edge_index(data.adj))
    else:
        output = None

    return target_gnn, output


def load_GATNet_model(file_path, data, hidden_channels, dropout, device, lr, weight_decay, num_layers, heads_num):
    gnn_model = GATNet_initialization(data, hidden_channels, dropout, lr, weight_decay, num_layers, heads_num, device)
    gnn_model.load_state_dict(torch.load(file_path))
    gnn_model.eval()
    return gnn_model
