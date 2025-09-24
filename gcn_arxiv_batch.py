#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/9/12 09:22
# @Author   : **
# @Email    : **@**
# @File     : gcn_arxiv_batch.py
# @Software : PyCharm
# @Desc     :
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from deeprobust.graph.defense import GCN, GraphConvolution
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected, to_dense_adj, k_hop_subgraph
from ogb.nodeproppred import PygNodePropPredDataset

from model.GCN import GCN_initilization, load_GCN_model
from utilty.utils import normalize_adj, accuracy, OGBNArxivDataset
from deeprobust.graph.data import Dataset
from torch_geometric.data import Data


def evaluate_gcn(model, data, device):
    """计算训练、验证、测试精度"""
    model.eval()
    with torch.no_grad():
        adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.size(0)).squeeze(0).to(device)
        norm_adj = normalize_adj(adj)
        out = model(data.x.to(device), norm_adj)
        pred = out.argmax(dim=1)

        accs = []
        for mask in [data.idx_train, data.idx_val, data.idx_test]:
            correct = pred[mask].eq(data.y[mask].to(device)).sum().item()
            accs.append(correct / len(mask))
    return accs  # train_acc, val_acc, test_acc

def evaluate_test_data(gnn_model, pyg_data, gcn_layer, device="cpu"):
    """
    使用 (gcn_layer+1)-hop 子图，对测试集目标节点进行预测并计算整体精度
    """
    gnn_model.eval()
    correct = 0
    total = 0
    preds = []
    labels = []

    with torch.no_grad():
        for node_id in pyg_data.idx_test.tolist():
            # 提取 (gcn_layer+1)-hop 子图
            node_index, edge_index, mapping, _ = k_hop_subgraph(
                node_idx=node_id,
                num_hops=gcn_layer + 1,
                edge_index=pyg_data.edge_index,
                relabel_nodes=True,
                num_nodes=pyg_data.num_nodes
            )

            # 子图特征 & 标签
            x_sub = pyg_data.x[node_index].to(device)
            y_sub = pyg_data.y[node_index].to(device)

            # 子图邻接矩阵 (dense)
            adj = to_dense_adj(edge_index, max_num_nodes=x_sub.size(0)).squeeze(0).to(device)
            norm_adj = normalize_adj(adj)

            # 前向传播
            out = gnn_model(x_sub, norm_adj)

            # mapping 是原始 node_id 在子图中的位置
            logit = out[mapping]
            pred = logit.argmax(dim=0, keepdim=True)

            preds.append(pred.item())
            labels.append(y_sub[mapping].item())

            if pred.item() == y_sub[mapping].item():
                correct += 1
            total += 1

def evaluate_val_mini_batch(model, data, batch_size=1024, num_neighbors=[10, 10], device="cpu"):
    """使用 mini-batch NeighborLoader 对验证集计算精度"""
    model.eval()
    val_idx = data.idx_val
    val_loader = NeighborLoader(
        data,
        input_nodes=val_idx,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            # 构建 mini-batch 的稀疏/局部 adjacency
            adj = to_dense_adj(batch.edge_index, max_num_nodes=batch.x.size(0)).squeeze(0).to(device)
            norm_adj = normalize_adj(adj)
            out = model(batch.x, norm_adj)
            logits = out[:batch.batch_size]
            target = batch.y[:batch.batch_size]
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += batch.batch_size
    return correct / total

def test_gcn(model, pyg_data, batch_size=512, num_neighbors=[5,5], device="cpu"):
    model.eval()
    test_loader = NeighborLoader(
        pyg_data,
        input_nodes=pyg_data.idx_test,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)

            adj = to_dense_adj(batch.edge_index, max_num_nodes=batch.x.size(0)).squeeze(0).to(device)
            norm_adj = normalize_adj(adj)

            out = model(batch.x, norm_adj)
            logits = out[:batch.batch_size]
            pred = logits.argmax(dim=1)

            target = batch.y[:batch.batch_size]
            correct += pred.eq(target).sum().item()
            total += batch.batch_size

    return correct / total

def GCN_model_batch(pyg_data, nhid, dropout, lr, weight_decay, with_bias,
                    batch_size=256, num_neighbors=[10, 10], epochs=5,
                    device="cpu", gcn_layer=2, target_gcn=None, save_path=None):
    """
    Mini-batch GCN training with normalize_adj
    """
    if target_gcn is None:
        target_gcn = GCN_initilization(
            features=pyg_data.x,
            labels=pyg_data.y,
            nhid=nhid,
            dropout=dropout,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            with_bias=with_bias,
            gcn_layer=gcn_layer
        )
        target_gcn = target_gcn.to(device)

        optimizer = torch.optim.Adam(target_gcn.parameters(), lr=lr, weight_decay=weight_decay)

        train_idx = pyg_data.idx_train
        train_loader = NeighborLoader(
            pyg_data,
            input_nodes=train_idx,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        print("Train label distribution:", torch.bincount(pyg_data.y[pyg_data.idx_train]))
        print("Val label distribution:", torch.bincount(pyg_data.y[pyg_data.idx_val]))
        print("Test label distribution:", torch.bincount(pyg_data.y[pyg_data.idx_test]))

        print("Start training ...")
        val_interval = 5  # 每隔多少个 epoch 计算一次验证集精度
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(1, epochs + 1):
            target_gcn.train()
            total_loss = 0
            n_samples = 0

            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for batch in batch_iter:
                batch = batch.to(device)
                optimizer.zero_grad()

                # 构建稠密邻接矩阵 (mini-batch) + 归一化
                adj = to_dense_adj(batch.edge_index, max_num_nodes=batch.x.size(0)).squeeze(0).to(device)
                # adj.fill_diagonal_(1.0)

                # 使用自定义的 normalize_adj
                norm_adj = normalize_adj(adj)

                out = target_gcn(batch.x, norm_adj)
                target = batch.y[:batch.batch_size]
                logits = out[:batch.batch_size]
                loss = target_gcn.loss(logits, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch.batch_size
                n_samples += batch.batch_size

                # 打印 batch 精度（可选）
                # pred = logits.argmax(dim=1)
                # correct = pred.eq(target).sum().item()
                # batch_iter.set_postfix(loss=loss.item(), batch_acc=f"{correct}/{batch.batch_size}")
                batch_iter.set_postfix(loss=loss.item())

            avg_loss = total_loss / max(1, n_samples)
            # train_acc, val_acc, test_acc = evaluate_gcn(target_gcn, pyg_data, device)

            # 保存最佳模型
            # if val_acc > best_val_acc:
            #     best_val_acc = val_acc
            #     best_model_state = target_gcn.state_dict().copy()

            # 打印 epoch 级信息
            # print(f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | "
            #       f"Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}")
            print(f"Epoch {epoch:03d} | Loss {avg_loss:.4f}")

            # 每隔 val_interval 计算一次验证集精度
            if epoch % val_interval == 0:
                val_acc = evaluate_val_mini_batch(target_gcn, pyg_data, batch_size=batch_size,
                                                  num_neighbors=num_neighbors, device=device)
                print(f"Validation Acc (mini-batch) at epoch {epoch}: {val_acc:.4f}")

                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = target_gcn.state_dict().copy()
                    if save_path:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save(best_model_state, save_path)
                        print(f"Best model saved at epoch {epoch} with Val Acc {best_val_acc:.4f}")

        # if best_model_state:
        #     target_gcn.load_state_dict(best_model_state)
        #     print(f"Loaded best model with Val Acc {best_val_acc:.4f}")

        # if save_path:
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     torch.save(target_gcn.state_dict(), save_path)
        #     print(f"Model saved to {save_path}")

    else:
        print("Using provided GCN model (skip training).")

    return target_gcn


if __name__ == "__main__":
    device = torch.device("cpu")
    save_path = "./model_save/GCN/ogbn-arxiv/2-layer/gcn_model.pt"

    dataset = PygNodePropPredDataset(name="ogbn-arxiv")
    pyg_data = dataset[0]
    pyg_data.edge_index = to_undirected(pyg_data.edge_index)
    pyg_data.y = pyg_data.y.view(-1).long()
    split_idx = dataset.get_idx_split()
    pyg_data.idx_train = split_idx["train"]
    pyg_data.idx_val = split_idx["valid"]
    pyg_data.idx_test = split_idx["test"]

    data = OGBNArxivDataset(dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    #triaing
    # gnn_model = GCN_model_batch(
    #     pyg_data,
    #     nhid=64,
    #     dropout=0.5,
    #     lr=0.01,
    #     weight_decay=5e-4,
    #     with_bias=True,
    #     batch_size=128,
    #     num_neighbors=[5, 5],
    #     epochs=200,
    #     device=device,
    #     gcn_layer=2,
    #     save_path=save_path
    # )

    # load model and test
    gnn_model = load_GCN_model(save_path, features, labels, nhid=64, dropout=0.5, device="cpu", lr=0.01,
                               weight_decay=5e-4, with_bias=True, gcn_layer=2)
    acc_test = test_gcn(gnn_model, pyg_data, batch_size=512, num_neighbors=[5,5], device="cpu")
    print(f"Test Accuracy = {acc_test:.4f}")

    # dense_adj = torch.tensor(adj.toarray())
    # norm_adj = normalize_adj(dense_adj)
    # pre_output = gnn_model.forward(torch.tensor(features.toarray()), norm_adj)
    #
    # y_pred_orig_gnn = pre_output.argmax(dim=1)
    # print("y_true counts: {}".format(np.unique(labels[idx_test], return_counts=True)))
    # print("y_pred_orig_gnn counts: {}".format(np.unique(y_pred_orig_gnn.numpy()[idx_test], return_counts=True)))
    # acc_test = accuracy(y_pred_orig_gnn[idx_test], labels[idx_test])
    # print("Test set results:", "accuracy = {:.4f}".format(acc_test))

    # data = Dataset(root="./dataset", name="cora")
    # adj, features, labels = data.adj, data.features, data.labels
    # idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    #
    # # 将 deeprobust Dataset 转为 PyG Data
    # edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    # edge_index = edge_index[[1, 0]]  # PyG 格式 [2, num_edges]
    # x = torch.tensor(features.toarray(), dtype=torch.float32)
    # y = torch.tensor(labels, dtype=torch.long)
    #
    # pyg_data = Data(x=x, edge_index=edge_index, y=y)
    # pyg_data.idx_train = idx_train
    # pyg_data.idx_val = idx_val
    # pyg_data.idx_test = idx_test

    # save_path = "./model_save/GCN/ogbn-arxiv/gcn_batch.pt"
    # gnn_model = GCN_model_batch(
    #     pyg_data,
    #     nhid=16,  # Cora 小图，隐藏层 16 就够
    #     dropout=0.5,  # 保持不变，可尝试 0.5 或 0.3
    #     lr=0.01,  # 学习率可保持
    #     weight_decay=5e-4,  # 权重衰减保持
    #     with_bias=True,
    #     batch_size=32,  # Cora 节点少，batch 32 即可
    #     num_neighbors=[10, 10],  # 两层邻居采样多一点，提高信息覆盖
    #     epochs=200,  # 小图训练轮数多一些，200~300 较常用
    #     device=device,
    #     gcn_layer=2,
    #     save_path=save_path
    # )

    # gnn_model = load_GCN_model(save_path, features, labels, nhid=16, dropout=0.5, device="cpu", lr=0.01, weight_decay=5e-4,
    #                            with_bias=True, gcn_layer=2)
    # dense_adj = torch.tensor(adj.toarray())
    # norm_adj = normalize_adj(dense_adj)
    # pre_output = gnn_model.forward(torch.tensor(features.toarray()), norm_adj)
    #
    # y_pred_orig_gnn = pre_output.argmax(dim=1)
    # print("y_true counts: {}".format(np.unique(labels[idx_test], return_counts=True)))
    # print("y_pred_orig_gnn counts: {}".format(np.unique(y_pred_orig_gnn.numpy()[idx_test], return_counts=True)))
    # acc_test = accuracy(y_pred_orig_gnn[idx_test], labels[idx_test])
    # print("Test set results:", "accuracy = {:.4f}".format(acc_test))
