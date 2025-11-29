#!/usr/bin/env python
# coding:utf-8
import os
import sys
import time
from torch_geometric.utils import k_hop_subgraph
import numpy as np
import scipy.sparse as sp
import torch
import argparse
from dgl.data import BAShapeDataset
from tqdm import tqdm
import torch.nn.functional as F
from counterfactual_explanation_subgraph.NSEG.model import GCN, GIN
from utilty.common_utils import Selector
from counterfactual_explanation_subgraph.NSEG.explainer_NSEG import NSEG
from utilty.utils import compute_deg_diff, compute_motif_viol

res = os.path.abspath(__file__)
base_path = os.path.dirname(res)
sys.path.insert(0, base_path)


def arg_parse():
    parser = argparse.ArgumentParser(description="NSEG arguments")
    parser.add_argument("--dataset_name", dest="dataset_name", type=str, help="BA-Shapes, Tree-Cycles, or Tree-Grid")
    parser.add_argument("--device", dest="device", type=str, help="cpu or cuda")
    parser.add_argument("--seed", dest="seed", type=int, help="seed")
    parser.add_argument("--model_arc", dest="model_arc", type=str, help="model architecture")
    parser.set_defaults(
        dataset_name="BA-Shapes",
        device="cpu",
        seed=0,
        model_arc='GCN'
    )
    return parser.parse_args()


def main():
    prog_args = arg_parse()
    dataset_name = prog_args.dataset_name
    device = prog_args.device
    seed = prog_args.seed
    model_arc = prog_args.model_arc
    print(dataset_name, device, seed, model_arc)

    torch.manual_seed(seed)

    # Load dataset
    if dataset_name == 'BA-Shapes':
        dataset = BAShapeDataset()
        graph = dataset[0].to(device)
        features = graph.ndata['feat'].float().to(device)
        labels = graph.ndata['label'].long().to(device)
        num_nodes = graph.number_of_nodes()
        nodes = [3, 6, 8, 16, 17, 18, 19, 36, 42, 49, 57, 59, 61, 71, 82, 102, 111, 120, 138, 140, 162, 175, 178, 185,
                 206, 208, 214, 224, 230, 235, 236, 238, 240, 246, 253, 258, 260, 268, 272, 274, 277, 281, 284, 288,
                 293, 294, 300, 305, 306, 307, 308, 312, 315, 319, 321, 324, 327, 333, 337, 338, 339, 342, 344, 351,
                 359, 361, 363, 365, 372, 376, 378, 379, 384, 387, 388, 394, 395, 396, 409, 414, 415, 417, 424, 425,
                 429, 435, 438, 439, 441, 445, 446, 448, 449, 455, 461, 462, 465, 466, 469, 474, 478, 480, 481, 485,
                 486, 488, 493, 494, 495, 504, 505, 506, 508, 509, 513, 517, 519, 523, 524, 526, 528, 529, 534, 540,
                 547, 549, 552, 559, 562, 570, 573, 576, 583, 584, 586, 589, 594, 595, 599, 602, 604, 609, 614, 616,
                 618, 619, 623, 624, 629, 634, 650, 654, 663, 668, 669, 670, 671, 673, 675, 676, 678, 679, 684, 685,
                 688, 689, 691, 692, 693, 694]  # nodes for explanation
        nodes = nodes
        k_edge = 6
    else:
        raise NotImplementedError("Other datasets not yet implemented.")

    print("# nodes: {}, # edges: {}".format(num_nodes, graph.number_of_edges() // 2))

    # Define GCN model (with input linear layer)
    dim_input = features.shape[1]  # one-hot dim
    dim_hidden = 100  # match training hidden_dim
    num_classes = labels.max().item() + 1
    num_layers = 2  # match training layers

    if model_arc == 'GCN':
        model = GCN(dim_input, dim_hidden, num_classes, num_layers=num_layers).to(device)
    elif model_arc == 'GIN':
        model = GIN(dim_input, dim_hidden, num_classes, num_layers=num_layers).to(device)
    else:
        raise NotImplementedError

    model.load_state_dict(torch.load(os.path.join(base_path, 'log/BA_Shapes_logs/model.model')))
    model.eval()

    # Load NSEG configurations
    config_path = 'configs/{}.json'.format(dataset_name)
    config = Selector(config_path).args
    alpha_e = config.alpha_e
    beta_e = config.beta_e
    alpha_f = config.alpha_f
    beta_f = config.beta_f
    objective = config.objective
    type_ex = config.type_explanation
    num_epochs = config.num_epochs
    lr = config.lr

    # Initialize NSEG
    explainer = NSEG(model=model,
                     num_hops=num_layers,
                     alpha_e=alpha_e,
                     beta_e=beta_e,
                     alpha_f=alpha_f,
                     beta_f=beta_f,
                     num_epochs=num_epochs,
                     objective=objective,
                     type_ex=type_ex,
                     device=device,
                     lr=lr)

    explanations = []
    thres = 0.5
    misclas_num = 0
    fidelity = 0.0
    S_plau = 0.0
    edited_num = 0.0
    added_edges_num = 0.0
    deleted_edges_num = 0.0
    time_list = []
    for node in tqdm(nodes):
        start = time.time()
        mask_edge_sigmoid, edge_ids = explainer.explain_node(node, graph, features)
        explanations.append([mask_edge_sigmoid.detach().cpu().numpy(), edge_ids.detach().cpu().numpy()])
        time_cost = time.time() - start
        time_list.append(time_cost)

        node_index, edge_index, mapping, _ = k_hop_subgraph(
            node_idx=node,
            num_hops=num_layers + 1,  # Cover the receptive field of GCN
            edge_index=torch.stack([graph.edges()[0], graph.edges()[1]], dim=0),
            relabel_nodes=True,
            num_nodes=graph.num_nodes()
        )
        # original adjacency matrix
        u, v = graph.edges()
        adj_orig = sp.coo_matrix(
            (np.ones(len(u)), (u.numpy(), v.numpy())),
            shape=(graph.num_nodes(), graph.num_nodes())
        )
        adj_orig_dense = torch.tensor(adj_orig.toarray(), dtype=torch.float32)

        # Construct the perturbed adjacency matrix
        # For example, select retained edges based on mask_edge_sigmoid threshold
        mask_selected = (mask_edge_sigmoid > thres).float()

        # Initialize a new adjacency matrix (dense)
        adj_perturb = adj_orig_dense.clone()
        # edges_ids corresponds to the index order of graph.edges() and needs to correspond to
        src, dst = graph.edges()
        for i, eid in enumerate(edge_ids):
            adj_perturb[src[eid], dst[eid]] *= mask_selected[i]
            adj_perturb[dst[eid], src[eid]] *= mask_selected[i]  # symmetry

        # original forecast
        model.eval()
        with torch.no_grad():
            pred_orig = model(graph, features)
            pred_node_orig = pred_orig[node].argmax().item()

        # Prediction after perturbation
        # If the model supports edge weight input, you can directly pass the mask; otherwise, you need to build a new graph.
        graph_perturb = graph.clone()
        graph_perturb.edata['weight'] = mask_edge_sigmoid  # If GraphConv supports edge_weight
        with torch.no_grad():
            pred_perturb = model(graph_perturb, features, eweight=mask_edge_sigmoid)
            pred_node_perturb = pred_perturb[node].argmax().item()

        # print("Adjacency matrix before perturbation:\n", adj_orig_dense)
        # print("Adjacency matrix after perturbation:\n", adj_perturb)
        # print("Node prediction before perturbation:", pred_node_orig)
        # print("Node prediction after perturbation:", pred_node_perturb)

        # fidelity
        prob_pred_orig = F.softmax(pred_orig[node])
        label_pred_orig = pred_node_orig
        prob_new_actual = F.softmax(pred_perturb[node])
        fidelity += prob_pred_orig[label_pred_orig] - prob_new_actual[label_pred_orig]
        if len(edge_ids) <= 5 and pred_node_orig != pred_node_perturb:
            print("found cf")
            misclas_num += 1
            edited_num += len(edge_ids)
            deleted_edges_num += len(edge_ids)
            # plausibility
            α2 = 1.5
            α3 = 1.0
            tau_c = 0
            k = 1
            L_plau = α2 * compute_deg_diff(adj_orig_dense[node_index][:, node_index],
                                           adj_perturb[node_index][:, node_index]) + α3 * compute_motif_viol(
                adj_orig_dense[node_index][:, node_index], adj_perturb[node_index][:, node_index], tau_c)
            S_plau += 2 * (1 - 1 / (1 + torch.exp(-1 * k * L_plau)))

    # evaluate
    print("Num of target nodes: ", len(nodes))
    print("Num of misclassification: ", misclas_num)
    print("Num of cf examples found: {}/{}".format(misclas_num, len(nodes)))
    print("Metric 1 - Misclassification Rate: {:.2f}".format(misclas_num / len(nodes)))
    print("Metric 2 - Fidelity: {:.4f}".format(fidelity / len(nodes)))
    if misclas_num == 0:
        misclas_num = 1
    print("Metric 3 - Average Explanation Size: {:.2f}, E+: {:.2f}, E-: {:.2f}".format(edited_num / misclas_num,
                                                                                       added_edges_num / misclas_num,
                                                                                       deleted_edges_num / misclas_num))
    print("Metric 4 - Average Plausibility: {:.2f}".format(S_plau / misclas_num))
    print("Metric 5 - Average Time Cost: {:.2f}s/per".format(np.mean(np.array(time_list))))

    print("Done generating explanations for {} nodes.".format(len(nodes)))


if __name__ == "__main__":
    main()
