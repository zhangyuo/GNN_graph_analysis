#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/9/12 09:22
# @Author   : **
# @Email    : **@**
# @File     : gcn_cpu_arxiv.py
# @Software : PyCharm
# @Desc     :
"""
# !/usr/bin/env python3
# cf_gcn_cpu_arxiv.py
# CPU-only (Intel Mac) example for ogbn-arxiv:
# - mini-batch NeighborLoader training
# - small GCN (hidden=64) to be CPU-friendly
# - batched inference
# - a lightweight CF-style edge-mask explainer on a k-hop subgraph
# - optional NetworkX visualization of the subgraph
#
# Requirements:
# pip install torch torchvision torchaudio
# pip install torch-geometric   # follow https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
# pip install ogb
# pip install networkx matplotlib tqdm

import os
import math
import time
import random
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected, k_hop_subgraph
from ogb.nodeproppred import PygNodePropPredDataset

import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------
# Config / hyperparams
# -----------------------
DEFAULTS = {
    "dataset_name": "ogbn-arxiv",
    "hidden": 64,  # 更小的隐藏层：CPU 友好
    "dropout": 0.5,
    "num_neighbors": [5, 5],  # 两层采样各采 5 个邻居
    "batch_size": 128,  # 可根据内存降低到 512/256/128
    "num_workers": None,  # None: 自动选择 (脚本内会回退到 0)
    "lr": 0.01,
    "weight_decay": 5e-4,
    "epochs": 1,  # CPU 上先少跑几轮试试 5/2/1
    "seed": 42,
    "cf_epochs": 200,
    "cf_lr": 0.1,
    "cf_lambda": 0.01,
    "cf_mask_init": 5.0,
    "k_hop": 2,
    "inference_batch_size": 1024
}


# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)  # 可选：提高可重复性但可能更慢


def get_num_workers(preferred: int = None) -> int:
    cpu_count = os.cpu_count() or 1
    if preferred is not None:
        n = max(0, min(preferred, cpu_count - 1))
    else:
        n = max(0, min(4, cpu_count - 1))
    return n


# -----------------------
# Model
# -----------------------
class GCNNet(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        # edge_weight is optional (we don't use it during normal training)
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


# -----------------------
# Training / inference
# -----------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch in tqdm(loader, desc="Train batches", leave=False):
        # batch is a torch_geometric.data.Data object for the subgraph
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        # only the first `batch.batch_size` nodes are the seed nodes (the ones we predict)
        target = batch.y[:batch.batch_size].view(-1)
        logits = out[:batch.batch_size]
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        bsz = int(batch.batch_size)
        total_loss += float(loss.item()) * bsz
        n_samples += bsz
    return total_loss / max(1, n_samples)


@torch.no_grad()
def inference_all(model, data, batch_size=1024, num_workers=0, device=torch.device('cpu')):
    model.eval()
    num_nodes = data.num_nodes
    num_classes = int(data.y.max().item() + 1) if data.y is not None else model.conv2.out_channels
    logits_all = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    loader = NeighborLoader(
        data,
        input_nodes=torch.arange(num_nodes),
        num_neighbors=[-1, -1],  # take full neighborhood (for correct message passing)
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    for batch in tqdm(loader, desc="Full-graph inference", leave=False):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        logits_all[batch.n_id] = out.cpu()
    return logits_all


def save_checkpoint(path, model, optimizer=None, epoch=None, val_acc=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc,
    }
    if optimizer is not None:
        ckpt['optimizer_state_dict'] = optimizer.state_dict()
    if extra is not None:
        ckpt['extra'] = extra
    torch.save(ckpt, path)
    print(f"Saved checkpoint to: {path} ({os.path.getsize(path) / 1e6:.2f} MB)")

def load_model(path, model, optimizer):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_val = ckpt.get('val_acc', 0.0)


# -----------------------
# Simple CF explainer on a k-hop subgraph (edge mask optimization)
# -----------------------
def cf_explain_node(model, data, node_idx: int, device, k_hop=2,
                    epochs=200, lr=0.1, lambda_reg=0.01, mask_init_val=5.0):
    model.eval()
    # 1) original prediction (fast approximate: do batched inference)
    full_logits = inference_all(model, data, batch_size=2048, num_workers=0, device=device)
    orig_label = int(full_logits[node_idx].argmax().item())
    print(f"[CF] node {node_idx} original label = {orig_label}")

    # 2) extract k-hop subgraph (relabel nodes)
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, num_hops=k_hop, edge_index=data.edge_index, relabel_nodes=True
    )
    center = int(mapping.item())
    sub_x = data.x[subset].to(device)
    sub_edge_index = sub_edge_index.to(device)

    # 3) learn a soft mask over edges (optimize logits(center) to flip)
    raw_mask = torch.nn.Parameter(torch.ones(sub_edge_index.size(1)) * mask_init_val, requires_grad=True)
    optim = torch.optim.Adam([raw_mask], lr=lr)

    for ep in range(epochs):
        optim.zero_grad()
        mask = torch.sigmoid(raw_mask)  # (0,1)
        out = model(sub_x, sub_edge_index, edge_weight=mask)
        logits_center = out[center].unsqueeze(0)  # shape [1, C]
        ce = F.cross_entropy(logits_center, torch.tensor([orig_label], device=device))
        # objective: encourage flipping (i.e., reduce probability of orig_label) -> maximize -ce
        loss = -ce + lambda_reg * mask.sum()
        loss.backward()
        optim.step()

        with torch.no_grad():
            pred_center = int(out[center].argmax().item())
        if ep % 20 == 0 or pred_center != orig_label:
            print(f"[CF] ep={ep} pred={pred_center} loss={loss.item():.4f}")
        if pred_center != orig_label:
            print(f"[CF] prediction flipped at epoch {ep} -> new_pred={pred_center}")
            break

    final_mask = torch.sigmoid(raw_mask).detach().cpu()
    removed_edges = (final_mask < 0.5).nonzero(as_tuple=False).view(-1).tolist()
    print(f"[CF] suggested to remove {len(removed_edges)} edges in subgraph (indices wrt sub_edge_index)")
    return final_mask, removed_edges, (subset, sub_edge_index, center)


# -----------------------
# Visualization (optional)
# -----------------------
def visualize_cf_subgraph(subset_nodes, sub_edge_index, center_idx, mask_vals, removed_threshold=0.5, figsize=(8, 6)):
    # convert to NetworkX
    g = nx.Graph()
    # add nodes (with original id labels)
    for i, nid in enumerate(subset_nodes.tolist()):
        g.add_node(i, orig_id=int(nid))
    # edges
    edge_index_np = sub_edge_index.cpu().numpy()
    for ei in range(edge_index_np.shape[1]):
        u = int(edge_index_np[0, ei])
        v = int(edge_index_np[1, ei])
        g.add_edge(int(u), int(v), idx=ei, mask=float(mask_vals[ei].item()))
    pos = nx.spring_layout(g, seed=0)
    plt.figure(figsize=figsize)
    edge_colors = ['red' if g[u][v]['mask'] < removed_threshold else 'black' for u, v in g.edges()]
    edge_widths = [1.5 if g[u][v]['mask'] < removed_threshold else 0.6 for u, v in g.edges()]
    nx.draw_networkx_nodes(g, pos, node_color='lightblue', node_size=50)
    nx.draw_networkx_edges(g, pos, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_nodes(g, pos, nodelist=[center_idx], node_color='gold', node_size=120)
    plt.title("k-hop subgraph (red edges suggested to remove by CF)")
    plt.axis('off')
    plt.show()


# -----------------------
# Main entry
# -----------------------
def main(args):
    cfg = DEFAULTS.copy()
    cfg.update(vars(args))

    set_seed(cfg['seed'])

    # device: force CPU
    device = torch.device('cpu')
    print("Device:", device)

    # set threading for CPU (helps BLAS / linear algebra speed)
    n_threads = max(1, (os.cpu_count() or 1) - 1)
    torch.set_num_threads(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    print(f"Using {n_threads} CPU threads for torch")

    # load dataset
    print("Loading dataset... (this will download/process if needed)")
    dataset = PygNodePropPredDataset(name=cfg['dataset_name'])
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    num_nodes = data.num_nodes
    num_classes = dataset.num_classes
    print(f"num_nodes={num_nodes}, num_classes={num_classes}")

    # convert directed -> undirected (paper experiments often do this)
    data.edge_index = to_undirected(data.edge_index)
    data.y = data.y.view(-1).long()

    # build model
    model = GCNNet(in_channels=data.num_node_features,
                   hidden=cfg['hidden'],
                   out_channels=num_classes,
                   dropout=cfg['dropout']).to(device)

    # DataLoader (NeighborLoader) - choose num_workers carefully
    preferred_workers = cfg['num_workers']
    if preferred_workers is None:
        preferred_workers = get_num_workers(None)
    num_workers = get_num_workers(preferred_workers)
    print(f"Attempting to use num_workers={num_workers} for NeighborLoader (will fallback on error)")

    train_idx = split_idx['train']

    try:
        train_loader = NeighborLoader(
            data,
            input_nodes=train_idx,
            num_neighbors=cfg['num_neighbors'],
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=num_workers
        )
    except Exception as e:
        print("NeighborLoader with multiprocessing failed; falling back to num_workers=0. Error:", e)
        num_workers = 0
        train_loader = NeighborLoader(
            data,
            input_nodes=train_idx,
            num_neighbors=cfg['num_neighbors'],
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=0
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    # quick check: print one batch shape
    print("Inspecting one batch (debug):")
    for b in train_loader:
        print(" batch.n_id len:", len(b.n_id), " batch.batch_size:", b.batch_size, " x.shape:", b.x.shape)
        break

    # training loop (very small epochs by default on CPU)
    best_val = 0.0
    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, device)
        t1 = time.time()
        print(f"[Train] epoch={epoch} loss={loss:.4f} time={t1 - t0:.1f}s")

        # every few epochs do a quick batched inference on validation and print acc
        if epoch % 5 == 0 or epoch == cfg['epochs']:
            logits = inference_all(model, data, batch_size=cfg['inference_batch_size'], num_workers=num_workers,
                                   device=device)
            pred = logits.argmax(dim=1)
            val_idx = split_idx['valid']
            val_acc = int((pred[val_idx] == data.y[val_idx]).sum()) / val_idx.size(0)
            print(f"  Val acc @ epoch {epoch}: {val_acc:.4f}")
            # save checkpoint
            if val_acc > best_val:
                best_val = val_acc
                path = f"./model_save/GCN/ogbn-arxiv/{'2-layer'}/checkpoints/gcn_hidden{cfg['hidden']}_best_val{val_acc:.4f}.pt"
                save_checkpoint(path, model, optimizer, epoch=epoch, val_acc=val_acc)

    # # Example: run CF explain on one validation node (small k-hop subgraph)
    # sample_node = int(split_idx['valid'][0].item())
    # print("Running CF explainer on node:", sample_node)
    # mask_vals, removed_edges, subgraph_info = cf_explain_node(
    #     model, data, node_idx=sample_node, device=device,
    #     k_hop=cfg['k_hop'], epochs=cfg['cf_epochs'],
    #     lr=cfg['cf_lr'], lambda_reg=cfg['cf_lambda'], mask_init_val=cfg['cf_mask_init']
    # )
    #
    # # Optional visualization (uncomment if you have matplotlib)
    # try:
    #     subset, sub_edge_index, center = subgraph_info
    #     visualize_cf_subgraph(subset, sub_edge_index, center, mask_vals, removed_threshold=0.5)
    # except Exception as e:
    #     print("Visualization failed or skipped:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=DEFAULTS['dataset_name'])
    parser.add_argument("--hidden", type=int, default=DEFAULTS['hidden'])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS['batch_size'])
    parser.add_argument("--num_workers", type=int,
                        default=DEFAULTS['num_workers'] if DEFAULTS['num_workers'] is not None else -1,
                        help="Set -1 to auto choose. If loader crashes on macOS, try 0.")
    parser.add_argument("--epochs", type=int, default=DEFAULTS['epochs'])
    parser.add_argument("--seed", type=int, default=DEFAULTS['seed'])
    args = parser.parse_args()

    # normalize num_workers: -1 => None for auto
    if getattr(args, "num_workers", -1) == -1:
        args.num_workers = None

    main(args)
