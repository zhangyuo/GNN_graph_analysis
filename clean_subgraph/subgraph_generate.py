#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/25 14:23
# @Author   : **
# @Email    : **@**
# @File     : subgraph_generate.py
# @Software : PyCharm
# @Desc     :
"""
import os
import sys

from deeprobust.graph.data import Dataset

from utilty.clean_subgraph_visualization import visualize_restricted_clean_subgraph

if __name__ == '__main__':
    res = os.path.abspath(__file__)  # acquire absolute path of current file
    base_path = os.path.dirname(os.path.dirname(res))  # acquire the parent path of current file's parent path
    sys.path.insert(0, base_path)

    # test case
    data = Dataset(root=base_path + '/dataset', name='cora')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    target_node = 1554

    visualize_restricted_clean_subgraph(
        adj,
        labels,
        features,
        target_node,
        k_hop=2,
        max_nodes=25,
        title="Visualization for Clean Subgraph",
        pic_path=base_path+'/clean_subgraph/results/'
    )

    print("ok")

    # # 1 动态交互式可视化（PyVis）
    # from pyvis.network import Network
    #
    # # 生成交互式HTML
    # net = Network(notebook=True, cdn_resources='remote', height="800px")
    # net.from_nx(G_pert)
    # net.show("attacked_graph.html")

    # # 2 特征降维可视化
    # from sklearn.manifold import TSNE
    #
    # # 降维特征到2D
    # tsne = TSNE(n_components=2, random_state=42)
    # feat_2d = tsne.fit_transform(perturbed_features.toarray())
    #
    # # 绘制特征空间分布
    # plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=labels, cmap=plt.cm.tab10, s=20)
    # plt.colorbar(label="Node Class")
    # plt.title("对抗样本的节点特征分布 (t-SNE降维)")

    # # 3 攻击强度量化
    # # 计算扰动比例
    # n_edges_orig = original_adj.sum() // 2
    # n_edges_pert = perturbed_adj.sum() // 2
    # perturb_ratio = abs(n_edges_pert - n_edges_orig) / n_edges_orig
    #
    # # 在标题中显示
    # plt.title(f"攻击修改比例: {perturb_ratio:.2%}", fontsize=14)

    # # 4 标签重叠处理
    # # 通过 nx.draw_networkx_labels 的 font_size 和 alpha 参数调整标签密度，或仅显示度中心性高的节点标签
    # degrees = dict(G_pert.degree())
    # high_degree_nodes = [n for n in G_pert.nodes() if degrees[n] > 5]
    # labels_subset = {n: label_dict[n] for n in high_degree_nodes}
