#!/usr/bin/env python
# coding:utf-8
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import BAShapes
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

from utilty.utils import normalize_adj

# =======================
# 1. 加载 BA-Shapes 数据集
# =======================
dataset = BAShapes()
data = dataset[0]

# 使用 one-hot 作为节点特征
num_features = dataset.num_classes
data.x = F.one_hot(data.y).float()

# 随机划分训练/验证/测试
transform = RandomNodeSplit(split="train_rest", num_val=140, num_test=200)
data = transform(data)

print(f"Number of nodes: {data.num_nodes}, classes: {dataset.num_classes}")
print(f"Train/Val/Test nodes: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")


# =======================
# 2. 定义 GAT 模型
# =======================
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.2):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=1)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout, edge_dim=1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            edge_attr = edge_weight.view(-1, 1)  # [num_edges, 1]
        else:
            edge_attr = None
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x


def edge_index_to_adj(edge_index, num_nodes):
    """将 PyG 的 edge_index 转换为邻接矩阵"""
    import scipy.sparse as sp
    row, col = edge_index
    adj = sp.coo_matrix((np.ones(row.shape[0], dtype=np.float32), (row, col)),
                        shape=(num_nodes, num_nodes))
    return adj.tocsr()


# =======================
# 3. 设置超参数
# =======================
in_channels = num_features
hidden_channels = 64
out_channels = dataset.num_classes
heads = 8
lr = 0.005
weight_decay = 5e-4
epochs = 500

# =======================
# 4. 初始化模型和优化器
# =======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(in_channels, hidden_channels, out_channels, heads=heads).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()


# =======================
# 5. 训练函数
# =======================
def train():
    model.train()
    optimizer.zero_grad()
    adj = edge_index_to_adj(data.edge_index, data.num_nodes)
    norm_adj = normalize_adj(torch.tensor(adj.toarray()))
    edge_index, edge_weight = dense_to_sparse(norm_adj)
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)
    out = model(data.x, edge_index, edge_weight)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


# =======================
# 6. 测试函数
# =======================
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        acc = int((pred[mask] == data.y[mask]).sum()) / int(mask.sum())
        accs.append(acc)
    return accs


# =======================
# 7. 训练循环
# =======================
best_val_acc = 0
best_test_acc = 0
for epoch in range(1, epochs + 1):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    if epoch % 50 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:04d} | Loss: {loss:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

print(f"\nBest Val Acc: {best_val_acc:.4f} | Test Acc at Best Val: {best_test_acc:.4f}")
