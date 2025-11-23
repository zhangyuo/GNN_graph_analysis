#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/23 14:20
# @Author   : **
# @Email    : **@**
# @File     : graph_embedding_vector.py
# @Software : PyCharm
# @Desc     :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, from_networkx
from torch_scatter import scatter_add


class GATSimilarity(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, add_self_loops=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)

    def forward(self, subgraph1, subgraph2):
        # Node embedding generation (returns attention weight tuple)
        emb1, attn_weights1 = self.conv1(
            subgraph1.x, subgraph1.edge_index, return_attention_weights=True
        )
        emb1 = F.relu(emb1)
        emb1 = self.conv2(emb1, subgraph1.edge_index)

        emb2, attn_weights2 = self.conv1(
            subgraph2.x, subgraph2.edge_index, return_attention_weights=True
        )
        emb2 = F.relu(emb2)
        emb2 = self.conv2(emb2, subgraph2.edge_index)

        # Graph-level representation (passing in the complete attention weight tuple)
        graph_emb1 = self._graph_pooling(emb1, attn_weights1)  # Only pass 2 parameters
        graph_emb2 = self._graph_pooling(emb2, attn_weights2)

        sim = F.cosine_similarity(graph_emb1, graph_emb2, dim=0)
        return round(sim.item(), 4), (attn_weights1, attn_weights2)

    def _graph_pooling(self, emb, attn_weights_tuple):
        """使用GAT返回的边索引（含自环）确保维度一致"""
