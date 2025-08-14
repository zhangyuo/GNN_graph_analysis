import os
import errno
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph
from deeprobust.graph.utils import classification_margin


def select_test_nodes(attack_type, explanation_type, idx_test, ori_output, labels):
    """
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    :param attack_type:
    :param explanation_type:
    :param idx_test:
    :return:
    """
    node_list = []
    node_list1 = []
    if attack_type is None:
        pass
    else:
        margin_dict_correct = {}
        margin_dict_incorrect = {}
        for idx in idx_test:
            margin = classification_margin(ori_output[idx], labels[idx])
            if margin < 0:  # only keep the nodes correctly classified
                margin_dict_incorrect[idx] = margin
            else:
                margin_dict_correct[idx] = margin

        sorted_margins = sorted(margin_dict_correct.items(), key=lambda x: x[1], reverse=True)
        high = []
        low = []
        other = []
        for class_num in set(labels):
            class_num_sorted_margins = [x for x, y in sorted_margins if labels[x] == class_num]
            high += [x for x in class_num_sorted_margins[: 10]]
            low += [x for x in class_num_sorted_margins[-10:]]
            other_0 = [x for x in class_num_sorted_margins[10: -10]]
            other += np.random.choice(other_0, 20, replace=False).tolist()

        node_list += high + low + other
        node_list = [int(x) for x in node_list]
        node_list = list(set(node_list))

        sorted_margins = sorted(margin_dict_incorrect.items(), key=lambda x: x[1], reverse=True)
        high = []
        low = []
        other = []
        for class_num in set(labels):
            class_num_sorted_margins = [x for x, y in sorted_margins if labels[x] == class_num]
            high += [x for x in class_num_sorted_margins[: 10]]
            low += [x for x in class_num_sorted_margins[-10:]]
            other_0 = [x for x in class_num_sorted_margins[10: -10]]
            other += np.random.choice(other_0, 20, replace=True).tolist()

        node_list1 += high + low + other
        node_list1 = [int(x) for x in node_list1]
        node_list1 = list(set(node_list1))

    return node_list, node_list1


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def safe_open(path, w):
    ''' Open "path" for writing, creating any parent directories as needed.'''
    mkdir_p(os.path.dirname(path))
    return open(path, w)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_degree_matrix(adj):
    return torch.diag(sum(adj))


def normalize_adj(adj):
    # Normalize adjacancy matrix according to reparam trick in GCN paper
    A_tilde = adj + torch.eye(adj.shape[0])
    D_tilde = get_degree_matrix(A_tilde)
    # Raise to power -1/2, set all infs to 0s
    D_tilde_exp = D_tilde ** (-1 / 2)
    D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

    # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
    norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
    return norm_adj


# def get_neighbourhood(node_idx, edge_index, n_hops, features, labels):
# 	edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index[0])     # Get all nodes involved
# 	edge_subset_relabel = subgraph(edge_subset[0], edge_index[0], relabel_nodes=True)       # Get relabelled subset of edges
# 	sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
# 	sub_feat = features[edge_subset[0], :]
# 	sub_labels = labels[edge_subset[0]]
# 	new_index = np.array([i for i in range(len(edge_subset[0]))])
# 	node_dict = dict(zip(edge_subset[0].numpy(), new_index))        # Maps orig labels to new
# 	# print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
# 	return sub_adj, sub_feat, sub_labels, node_dict

def get_neighbourhood(target_node, edge_index, features, labels, gcn_layer):
    # generate explanation for target node from specified explainer
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=target_node,
        num_hops=gcn_layer + 1,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=edge_index.max() + 1
    )

    sub_adj = to_dense_adj(sub_edge_index).squeeze()
    sub_feat = features[subset, :]
    sub_feat = torch.tensor(sub_feat.toarray(), dtype=torch.float)
    sub_labels = labels[subset]
    node_dict = {int(orig_id): idx for idx, orig_id in enumerate(subset.tolist())}
    # print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
    return sub_adj, sub_edge_index, sub_feat, sub_labels, node_dict


def create_symm_matrix_from_vec(vector, n_rows):
    matrix = torch.zeros(n_rows, n_rows)
    idx = torch.tril_indices(n_rows, n_rows)
    matrix[idx[0], idx[1]] = vector
    symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
    return symm_matrix


def create_vec_from_symm_matrix(matrix, P_vec_size):
    idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
    vector = matrix[idx[0], idx[1]]
    return vector


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def get_S_values(pickled_results, header):
    df_prep = []
    for example in pickled_results:
        if example != []:
            df_prep.append(example[0])
    return pd.DataFrame(df_prep, columns=header)


def redo_dataset_pgexplainer_format(dataset, train_idx, test_idx):
    dataset.data.train_mask = index_to_mask(train_idx, size=dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(test_idx[len(test_idx)], size=dataset.data.num_nodes)


def accuracy(pred, labels):
    """计算准确率"""
    pred_labels = torch.tensor(pred, dtype=int)
    labels = torch.tensor(labels, dtype=int)
    correct = (pred_labels == labels).float().sum()  # 统计正确预测数
    return correct / len(labels)  # 返回准确率
