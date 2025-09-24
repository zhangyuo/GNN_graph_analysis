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
        # 节点嵌入生成（返回注意力权重元组）
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

        # 图级表示（传入完整的注意力权重元组）
        graph_emb1 = self._graph_pooling(emb1, attn_weights1)  # 仅传2个参数
        graph_emb2 = self._graph_pooling(emb2, attn_weights2)

        sim = F.cosine_similarity(graph_emb1, graph_emb2, dim=0)
        return round(sim.item(), 4), (attn_weights1, attn_weights2)

    def _graph_pooling(self, emb, attn_weights_tuple):
        """使用GAT返回的边索引（含自环）确保维度一致"""
        edge_idx, attn_weights = attn_weights_tuple  # 解包元组
        attn_val = torch.mean(attn_weights, dim=1)  # [num_edges_with_self_loops]

        # 关键修复：使用GAT返回的edge_idx（含自环）
        target_nodes = edge_idx[1]  # 目标节点索引 [num_edges_with_self_loops]

        # 聚合边权重到目标节点
        node_attn = scatter_add(
            src=attn_val,  # 输入数据 [num_edges_with_self_loops]
            index=target_nodes,  # 目标节点索引
            dim=0,
            dim_size=emb.size(0)  # 输出维度 = 节点数
        )

        # 加权池化
        weighted_emb = emb * node_attn.unsqueeze(1)
        attn_pool = torch.mean(weighted_emb, dim=0)

        # 均值池化
        mean_pool = torch.mean(emb, dim=0)

        return torch.cat([attn_pool, mean_pool], dim=-1)


def cosine_similarity(emb1, emb2):
    sim = F.cosine_similarity(emb1, emb2, dim=0)
    sim = round(sim.item(), 2)
    return sim


def get_subgraph_edges(subgraph):
    pyg_data = from_networkx(subgraph)
    return pyg_data.edge_index


def compute_graph_similarity(subgraph1, subgraph2, pyg_data, pyg_gnn_model):
    target_nodes1 = list(subgraph1.nodes)
    target_nodes2 = list(subgraph2.nodes)

    mapping1 = dict(zip(subgraph1.nodes(), range(subgraph1.number_of_nodes())))
    sub_x = pyg_data.x[target_nodes1]
    try:
        sub_edge_index, _ = subgraph(
            # subset=target_nodes1,
            # edge_index=pyg_data.edge_index,
            subset=[mapping1[n] for n in target_nodes1],
            edge_index=get_subgraph_edges(subgraph1),
            relabel_nodes=True  # Key: Redefinition of node IDs
        )
    except:
        sub_edge_index, _ = subgraph(
            subset=target_nodes1,
            edge_index=pyg_data.edge_index,
            relabel_nodes=True  # Key: Redefinition of node IDs
        )
    node_emb1 = pyg_gnn_model.node_embeddings(sub_x, sub_edge_index)
    subgraph1_ge = pyg_gnn_model.graph_embeddings(node_emb1)

    mapping2 = dict(zip(subgraph2.nodes(), range(subgraph2.number_of_nodes())))
    sub_x = pyg_data.x[target_nodes2]
    try:
        sub_edge_index, _ = subgraph(
            # subset=target_nodes2,
            # edge_index=pyg_data.edge_index,
            subset=[mapping2[n] for n in target_nodes2],
            edge_index=get_subgraph_edges(subgraph2),
            relabel_nodes=True  # Key: Redefinition of node IDs
        )
    except:
        sub_edge_index, _ = subgraph(
            subset=target_nodes2,
            edge_index=pyg_data.edge_index,
            relabel_nodes=True  # Key: Redefinition of node IDs
        )
    node_emb2 = pyg_gnn_model.node_embeddings(sub_x, sub_edge_index)
    subgraph2_ge = pyg_gnn_model.graph_embeddings(node_emb2)

    sim = cosine_similarity(subgraph1_ge, subgraph2_ge)
    return sim


# 测试用例
if __name__ == "__main__":
    # 测试用例（含自环边验证）
    subgraph1 = Data(
        x=torch.randn(5, 16),
        edge_index=torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long)
    )
    subgraph2 = Data(
        x=torch.randn(3, 16),
        edge_index=torch.tensor([[0, 1, 2], [1, 0, 1]], dtype=torch.long)
    )

    model = GATSimilarity(in_dim=16, hidden_dim=32)
    similarity, (attn_weights1, attn_weights2) = model.forward(subgraph1, subgraph2)
    print(f"Similarity: {similarity.item():.4f}")  # 正常输出无报错
    print((attn_weights1, attn_weights2))

    # # 计算两子图边权差异: 对抗攻击特征：解释子图中高权重的边在对抗子图中权重显著降低（攻击者弱化关键连接）
    # threshold = 0.7
    # edge_idx_1, attn_weights_1 = attn_weights1
    # edge_idx_2, attn_weights_2 = attn_weights2
    # diff = torch.abs(attn_weights_1 - attn_weights_2)
    # high_diff_mask = (diff > threshold)  # 阈值根据分布设定
    # perturbed_edges = edge_idx_1[:, high_diff_mask]  # 扰动边坐标
    # print(perturbed_edges)
