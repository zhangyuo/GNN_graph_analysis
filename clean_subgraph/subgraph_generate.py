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

    # #1 Dynamic interactive visualization (PyVis)
    # from pyvis.network import Network
    #
    # # Generate interactive HTML
    # net = Network(notebook=True, cdn_resources='remote', height="800px")
    # net.from_nx(G_pert)
    # net.show("attacked_graph.html")

    # #2 Visualization of feature dimensionality reduction
    # from sklearn.manifold import TSNE
    #
    # # Reduce dimensionality features to 2D
    # tsne = TSNE(n_components=2, random_state=42)
    # feat_2d = tsne.fit_transform(perturbed_features.toarray())
    #
    # # Draw feature space distribution
    # plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=labels, cmap=plt.cm.tab10, s=20)
    # plt.colorbar(label="Node Class")
    # plt.title("Node feature distribution of adversarial samples (t-SNE dimensionality reduction)")

    # #3 Quantification of attack intensity
    # # Calculate the disturbance ratio
    # n_edges_orig = original_adj.sum() // 2
    # n_edges_pert = perturbed_adj.sum() // 2
    # perturb_ratio = abs(n_edges_pert - n_edges_orig) / n_edges_orig
    #
    # # Show in title
    # plt.title(f"Attack modification ratio: {perturb_ratio:.2%}", fontsize=14)

    # #4 Label overlap processing
    # # Adjust the label density through the font_size and alpha parameters of nx.draw_networkx_labels, or only display node labels with high degree centrality
    # degrees = dict(G_pert.degree())
    # high_degree_nodes = [n for n in G_pert.nodes() if degrees[n] > 5]
    # labels_subset = {n: label_dict[n] for n in high_degree_nodes}
