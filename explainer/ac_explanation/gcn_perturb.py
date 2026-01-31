#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/8/27 20:17
# @Author   : **
# @Email    : **@**
# @File     : gcn_perturb.py
# @Software : PyCharm
# @Desc     :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from deeprobust.graph.defense import GraphConvolution
import numpy as np

from utilty.utils import normalize_adj, get_degree_matrix, compute_deg_diff, compute_motif_viol, compute_feat_sim
from config.config import TEST_MODEL
from torch_geometric.nn import GINConv, SAGEConv, APPNP, GraphConv, TransformerConv, GATConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.utils import dense_to_sparse


class SignedMaskPerturbation(nn.Module):
    def __init__(self,
                 extended_sub_adj: torch.Tensor,
                 node_idx: int,
                 node_num_l_hop: list,
                 top_k: int = 5,
                 C: float = 1.0,
                 tau_plus: float = 0.5,
                 tau_minus: float = -0.5,
                 test_model: str = "GCN",
                 dataset_name: str = "cora"):
        """
        The symbolic masking perturbation module of AC-Explainer
        Parameters:
        extended_sub_adj: Extended subgraph adjacency matrix [n, n]
        node_idx: Index of the target node in the extended subgraph
        top_k: Maximum number of edges to retain (default 5)
        tau_plus: Threshold for adding edges (default 0.5)
        tau_minus: Threshold for removing edges (default -0.5)
        """
        super().__init__()
        self.extended_sub_adj = extended_sub_adj  # Clone and expand the subgraph adjacency matrix to avoid affecting the original data
        self.node_idx = node_idx  # The index of the target node in the subgraph
        self.tau_plus = tau_plus  # Add edge threshold
        self.tau_minus = tau_minus  # Threshold for edge deletion
        self.node_num_l_hop = node_num_l_hop
        self.C = C
        self.top_k = top_k  # The maximum number of edge modifications to retain
        self.n_nodes = extended_sub_adj.size(0)  # Expand the number of nodes in a subgraph
        self.plan_added_node_idx = []
        self.plan_deleted_node_idx = []
        self.test_model = test_model
        self.dataset_name = dataset_name

        # Initialize signed mask parameters
        self.M = self._initialize_mask()

    def _initialize_mask(self) -> nn.Parameter:
        """Initialize the mask based on the target node and the expanded subgraph"""
        eps = 10 ** -4
        mask_init_values = []
        mask_index = 0

        [node_index, attack_nodes, node_dict] = self.node_num_l_hop
        attack_nodes_idx = [node_dict[ad] for ad in attack_nodes]
        lhop_node_index = [node_dict[ni] for ni in node_index]

        # Traverse all existing edges in extended_sub_adj
        # sub_adj = self.extended_sub_adj[lhop_node_index, :][:, lhop_node_index]
        # init_value = -0.8  # GCN:-0.5 GraphConv:-1.0 GraphTransformer: -0.8
        try:
            init_value = {
                "GCN": {"cora": -0.5, "BA-SHAPES": -0.5, "TREE-CYCLES": -0.5, "Loan-Decision": -0.5, "ogbn-arxiv": -0.5},
                "GraphTransformer": {"cora": -0.8, "BA-SHAPES": -0.6, "TREE-CYCLES": -0.8, "Loan-Decision": -0.8, "ogbn-arxiv": -0.8},
                "GraphConv": {"cora": -1.0, "BA-SHAPES": -1.0, "TREE-CYCLES": -1.0, "Loan-Decision": -1.0, "ogbn-arxiv": -1.0},
                "GAT": {"cora": -0.8, "BA-SHAPES": -0.8, "TREE-CYCLES": -0.8, "Loan-Decision": -0.8, "ogbn-arxiv": -0.8}
            }[self.test_model][self.dataset_name]
        except:
            init_value = self.tau_minus
        init_value += 0.01*torch.randn(1).item()

        ones_indices = torch.nonzero(self.extended_sub_adj == 1)
        non_diagonal_ones = ones_indices[ones_indices[:, 0] != ones_indices[:, 1]].tolist()

        if self.dataset_name == "chameleon":  # simplification in complex edges relation for target node
            target_idx = self.node_idx
            neighbors = torch.nonzero(self.extended_sub_adj[target_idx] == 1).flatten().tolist()
            for nbr in neighbors:
                if nbr != target_idx and nbr in lhop_node_index:
                    a, b = sorted([target_idx, nbr])
                    mask_init_values.append(init_value)
                    self.plan_deleted_node_idx.append([mask_index, [a, b], True])
                    mask_index += 1
        else:
            for i in range(len(non_diagonal_ones)):
                if non_diagonal_ones[i][0] in lhop_node_index and non_diagonal_ones[i][1] in lhop_node_index and \
                        non_diagonal_ones[i][0] < non_diagonal_ones[i][1]:
                    # Existing edges are initialized to small negative numbers (favoring deletion)
                    mask_init_values.append(init_value)
                    self.plan_deleted_node_idx.append([mask_index, non_diagonal_ones[i], True])  # True denotes that orignal adj have edge
                    mask_index += 1

        # Traverse all attack_nodes, and tend to add them in scenarios where there are no existing edges, but the addition of edges needs to be suppressed.
        # init_value = 0.8  # GCN：0.4 GraphConv:0.55 GraphTransformer: 0.8
        try:
            init_value = {
                "GCN": {"cora": 0.4, "BA-SHAPES": 0.8, "TREE-CYCLES": 0.4, "Loan-Decision": 0.4, "ogbn-arxiv": 0.4},
                "GraphTransformer": {"cora": 0.8, "BA-SHAPES": 0.6, "TREE-CYCLES": 0.8, "Loan-Decision": 0.8, "ogbn-arxiv": 0.8},
                "GraphConv": {"cora": 0.55, "BA-SHAPES": 0.55, "TREE-CYCLES": 0.55, "Loan-Decision": 0.55, "ogbn-arxiv": 0.55},
                "GAT": {"cora": 0.8, "BA-SHAPES": 0.8, "TREE-CYCLES": 0.8, "Loan-Decision": 0.8, "ogbn-arxiv": 0.8}
            }[self.test_model][self.dataset_name]
        except:
            init_value = self.tau_plus
        init_value -= 0.01 * torch.randn(1).item()

        for i in attack_nodes_idx:
            if i != self.node_idx:
                mask_init_values.append(init_value)
                if [self.node_idx, i] in non_diagonal_ones or [i, self.node_idx] in non_diagonal_ones:
                    self.plan_added_node_idx.append([mask_index, [self.node_idx, i], True])
                else:
                    self.plan_added_node_idx.append([mask_index, [self.node_idx, i], False])
                mask_index += 1

        return nn.Parameter(torch.tensor(mask_init_values, dtype=torch.float32))

    def _apply_discretization(self, M_e: torch.Tensor) -> torch.Tensor:
        """
        Apply TopK sparsification (retain only the k largest perturbations in the gradient)
        Apply ternary discretization: -1 (delete), 0 (remain unchanged), +1 (add)
        """

        with torch.no_grad():
            costs = torch.zeros_like(M_e)
            for data in self.plan_added_node_idx + self.plan_deleted_node_idx:
                me_idx = data[0]
                is_original_edge = data[2]  # True=have edge / False=no edge
                if is_original_edge:
                    costs[me_idx] = 1
                else:
                    costs[me_idx] = self.C
            scores = torch.abs(M_e)
            _, sorted_idx = torch.sort(scores, descending=True)

            budget = 0.0
            selected = []

            for idx in sorted_idx:
                cost = costs[idx].item()
                if budget + cost > self.top_k:
                    continue
                selected.append(idx)
                budget += cost
                if budget == self.top_k:
                    break

            selected = torch.tensor(selected, dtype=torch.long)

            sparse_mask = torch.zeros_like(M_e)
            sparse_mask[selected] = 1

            top_k_M_e = M_e * sparse_mask

            full_mask = torch.zeros_like(self.extended_sub_adj)
            for data in self.plan_added_node_idx + self.plan_deleted_node_idx:
                full_mask[data[1][0], data[1][1]] = top_k_M_e[data[0]]
                full_mask[data[1][1], data[1][0]] = top_k_M_e[data[0]]
            # edge_idx = 0
            # for i in range(self.n_nodes):
            #     if i != self.node_idx and i in (self.plan_added_node_idx + self.plan_deleted_node_idx):
            #         full_mask[self.node_idx, i] = top_k_M_e[edge_idx]
            #         full_mask[i, self.node_idx] = top_k_M_e[edge_idx]
            #         edge_idx += 1

            # Calculate discrete values (use torch.where for ternarization)
            delta_A = torch.where(
                full_mask > self.tau_plus,
                1,
                torch.where(
                    full_mask < self.tau_minus,
                    -1,
                    0
                )
            )

        return delta_A

    def train_forward(self) -> torch.Tensor:
        """
        Training mode: Return consecutive approximate masks (while preserving gradients)
        Use a direct gradient estimator to maintain differentiability
        """
        full_mask = torch.zeros_like(self.extended_sub_adj)
        for data in self.plan_added_node_idx + self.plan_deleted_node_idx:
            full_mask[data[1][0], data[1][1]] = self.M[data[0]]
            full_mask[data[1][1], data[1][0]] = self.M[data[0]]

        return full_mask

    def predict_forward(self) -> torch.Tensor:
        """Prediction mode: Return completely discrete masks (without gradients)"""
        M_e = torch.tanh(self.M)
        delta_A = self._apply_discretization(M_e)
        return delta_A

    def build_perturbed_adj(self, adj, delta_A):
        perturbed_adj = torch.where(
            delta_A == 1,  # Condition: if delta_A indicates "add edge"
            torch.ones_like(adj),  # Then the corresponding position is set to 1
            torch.where(
                delta_A == -1,  # Otherwise, if delta_A indicates "delete edge"
                torch.zeros_like(adj),  # Then the corresponding position is set to 0
                adj  # Otherwise (delta_A == 0), keep the value of the original adjacency matrix unchanged.
            )
        )
        return perturbed_adj

    def ste_perturbed_adj(self, adj, full_mask):
        # 1. Map full_mask to the [-1, 1] interval through tanh, which is a differentiable operation
        continuous_mask = torch.tanh(full_mask)  # maintain gradient flow
        # 2. In forward propagation, discrete decisions are made based on the value of continuous_mask
        with torch.no_grad():
            # Create a matrix with the same shape as continuous_mask to store discrete decisions
            discrete_decision = torch.where(continuous_mask > 0.5,  # Condition 1: greater than 0.5
                                            torch.ones_like(continuous_mask),  # Condition 1 is met: set to 1 (add edge)
                                            torch.where(continuous_mask < -0.5,  # Condition 2: less than -0.5
                                                        -torch.ones_like(continuous_mask),  # Meet condition 2: Set -1 (delete edge)
                                                        torch.zeros_like(continuous_mask)))  # Otherwise: set to 0 (unchanged)
            # Generating perturbed adjacency matrices based on discrete decisions
            perturbed_adj_discrete = torch.where(discrete_decision > 0.5,
                                                 torch.ones_like(adj),
                                                 torch.where(discrete_decision < -0.5,
                                                             torch.zeros_like(adj),
                                                             adj))

        # 3. Key step: Use a straight-through estimator to connect forward discrete decisions and backward continuous gradients
        perturbed_adj = perturbed_adj_discrete + (continuous_mask - continuous_mask.detach())

        return perturbed_adj


class GNNPerturb(nn.Module):
    def __init__(self,
                 nfeat: int,
                 nhid: int,
                 nclass: int,
                 extended_sub_adj: torch.Tensor,
                 sub_feat: torch.Tensor,
                 node_idx: int,
                 node_num_l_hop: list,
                 dropout: float = 0.5,
                 lambda_pred: float = 1.0,
                 lambda_dist: float = 0.5,
                 lambda_plau: float = 0.2,
                 top_k: int = 5,
                 C: float = 1.0,
                 tau_plus: float = 0.5,
                 tau_minus: float = -0.5,
                 α1: float = 0.1,
                 α2: float = 0.1,
                 α3: float = 0.1,
                 α4: float = 0.5,
                 tau_c: float = 0.1,
                 gcn_layer: int = 2,
                 with_bias: bool = True,
                 test_model: str = "GCN",
                 heads: int = 2,
                 dataset_name: str = "cora"):
        super().__init__()

        self.gcn_layer = gcn_layer
        self.lambda_pred = lambda_pred  # Predicted loss weight
        self.lambda_dist = lambda_dist  # Sparse loss weight
        self.lambda_plau = lambda_plau  # Reality Loss Weight
        self.dropout = dropout
        self.extended_sub_adj = extended_sub_adj
        self.sub_feat = sub_feat
        self.num_nodes = self.extended_sub_adj.shape[0]
        self.node_idx = node_idx
        self.node_num_l_hop = node_num_l_hop
        self.α1 = α1
        self.α2 = α2
        self.α3 = α3
        self.α4 = α4
        self.tau_c = tau_c
        self.model_name = test_model
        self.dataset_name = dataset_name
        self.heads = heads
        self.C = C

        # Disturbance layer
        print(f"Input extended_sub_adj.requires_grad: {extended_sub_adj.requires_grad}")
        self.perturb_layer = SignedMaskPerturbation(extended_sub_adj, node_idx, node_num_l_hop, top_k, C, tau_plus,
                                                    tau_minus, test_model, dataset_name)

        # GCN layer definition
        if self.model_name == "GCN":
            if self.gcn_layer == 3:
                self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
                self.gc2 = GraphConvolution(nhid, nhid, with_bias=with_bias)
                self.gc3 = GraphConvolution(nhid, nclass, with_bias=with_bias)
                self.lin = nn.Linear(nhid + nhid + nclass, nclass)
            else:
                self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
                self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        elif self.model_name == "GraphTransformer":
            self.layers = nn.ModuleList()
            self.layers.append(TransformerConv(nfeat, nhid, heads=heads, dropout=dropout, edge_dim=1))
            for _ in range(self.gcn_layer - 2):
                self.layers.append(
                    TransformerConv(nhid * heads, nhid, heads=heads, dropout=dropout, edge_dim=1))
            self.layers.append(TransformerConv(nhid * heads, nclass, heads=1, dropout=dropout, edge_dim=1))
        elif self.model_name == "GraphConv":
            self.layers = nn.ModuleList()
            self.layers.append(GraphConv(nfeat, nhid, aggr="add"))
            for _ in range(self.gcn_layer - 2):
                self.layers.append(GraphConv(nhid, nhid, aggr="add"))
            self.layers.append(GraphConv(nhid, nclass, aggr="add"))
        elif self.model_name == "GAT":
            self.layers = nn.ModuleList()
            self.layers.append(GATConv(nfeat, nhid, heads=heads, dropout=dropout, edge_dim=1))
            for _ in range(self.gcn_layer - 2):
                self.layers.append(GATConv(nhid * heads, nhid, heads=heads, dropout=dropout, edge_dim=1))
            if self.dataset_name == "Cora":
                self.layers.append(GATConv(nhid * heads, nclass, heads=1, concat=False, dropout=dropout, edge_dim=1))
            else:
                self.layers.append(GATConv(nhid * heads, nclass, heads=1, dropout=dropout, edge_dim=1))

    def forward(self, x: torch.Tensor, sub_adj: torch.Tensor) -> torch.Tensor:
        """Training mode: Use continuous perturbation matrix"""
        self.sub_adj = sub_adj
        self.full_mask = self.perturb_layer.train_forward()

        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True

        # Use tanh to bound full mask in [-1,1]
        scale = 1.0
        perturbed_adj = self.perturb_layer.ste_perturbed_adj(self.sub_adj, scale * self.full_mask)
        A_tilde = perturbed_adj + torch.eye(self.num_nodes)

        D_tilde = get_degree_matrix(A_tilde).detach()  # Don't need gradient of this degree matrix
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)  # Normalized adjacency matrix

        return self._gcn_forward(x, norm_adj)

    def forward_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Prediction mode: Utilizing discrete perturbation matrix"""
        self.delta_A = self.perturb_layer.predict_forward()

        A_tilde = self.perturb_layer.build_perturbed_adj(self.extended_sub_adj, self.delta_A) + torch.eye(
            self.num_nodes)  # discretized adjacency matrix

        D_tilde = get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        return self._gcn_forward(x, norm_adj)

    def _gcn_forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        if self.model_name == "GCN":
            if self.gcn_layer == 3:
                x1 = F.relu(self.gc1(x, norm_adj))
                x1 = F.dropout(x1, self.dropout, training=self.training)
                x2 = F.relu(self.gc2(x1, norm_adj))
                x2 = F.dropout(x2, self.dropout, training=self.training)
                x3 = self.gc3(x2, norm_adj)
                x = self.lin(torch.cat((x1, x2, x3), dim=1))
                return F.log_softmax(x, dim=1)
            else:
                x1 = F.relu(self.gc1(x, norm_adj))
                x1 = F.dropout(x1, self.dropout, training=self.training)
                x2 = self.gc2(x1, norm_adj)
                return F.log_softmax(x2, dim=1)
        elif self.model_name in ["GraphTransformer", "GAT"]:
            edge_index, edge_weight = dense_to_sparse(norm_adj)
            edge_index = edge_index.to(x.device)
            edge_attr = edge_weight.view(-1, 1)  # [num_edges, 1]
            edge_attr.requires_grad_(True)
            for conv in self.layers[:-1]:
                x = conv(x, edge_index, edge_attr=edge_attr)
                if self.model_name == "GAT" and self.dataset_name == "BA-SHAPES":
                    x = F.elu(x)
                else:
                    x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # last layer
            x = self.layers[-1](x, edge_index, edge_attr=edge_attr)
            return F.log_softmax(x, dim=1)
        elif self.model_name in ["GraphConv"]:
            edge_index, edge_weight = dense_to_sparse(norm_adj)
            edge_index = edge_index.to(x.device)
            edge_attr = edge_weight.view(-1, 1)  # [num_edges, 1]
            edge_attr.requires_grad_(True)
            for conv in self.layers[:-1]:
                x = conv(x, edge_index, edge_weight=edge_attr)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # last layer
            x = self.layers[-1](x, edge_index, edge_weight=edge_attr)
            return F.log_softmax(x, dim=1)

    def get_mask_parameters(self) -> nn.Parameter:
        """Obtain trainable mask parameters"""
        return self.perturb_layer.M

    def compute_losses(self,
                       output: torch.Tensor,
                       y_pred_orig: torch.Tensor,
                       y_pred_new_actual: torch.Tensor) -> tuple:
        """Calculate the multi-objective loss function"""

        # Prediction loss (encourages flipping predictions)
        pred_loss = -F.nll_loss(
            output[self.node_idx].unsqueeze(0),
            y_pred_orig.unsqueeze(0)
        ) * (y_pred_new_actual == y_pred_orig.unsqueeze(0)).float()

        cf_adj = self.perturb_layer.ste_perturbed_adj(self.extended_sub_adj, self.full_mask)
        if self.lambda_dist == 0:
            dist_loss = torch.tensor(0.0)
        else:
            dist_loss = torch.sum(torch.abs(cf_adj - self.extended_sub_adj)) / 2
            diff = cf_adj - self.extended_sub_adj
            add_mask = (diff == 1)
            num_additions = add_mask.sum() / 2
            del_mask = (diff == -1)
            num_deletions = del_mask.sum() / 2
            dist_loss = self.C * num_additions + 1.0 * num_deletions

            # loss of reality
        if self.lambda_plau == 0:
            plau_loss = torch.tensor(0.0)
        else:
            plau_loss = self.compute_plausibility_loss()

        # weighted total loss
        total_loss = self.lambda_pred * pred_loss + self.lambda_dist * dist_loss + self.lambda_plau * plau_loss

        return total_loss, pred_loss, dist_loss, plau_loss, cf_adj, self.delta_A, self.perturb_layer, self.full_mask

    def compute_plausibility_loss(self) -> torch.Tensor:
        """Calculate the loss of reality"""
        loss_components_1 = torch.tensor(0.0)
        loss_components_2 = torch.tensor(0.0)
        loss_components_3 = torch.tensor(0.0)
        loss_components_4 = torch.tensor(0.0)

        # 1. Feature similarity penalty (only for added edges)
        # add_mask = (self.delta_A > 0.5)
        # if add_mask.sum() > 0:
        #     # Calculate feature similarity
        #     target_feat = self.sub_feat[self.node_idx]
        #     for i in range(self.extended_sub_adj.size(0)):
        #         if add_mask[self.node_idx, i]:
        #             feat_sim = compute_feat_sim(target_feat, self.sub_feat[i])
        #             loss_components_1 = loss_components_1 + (1 - feat_sim) * self.α1
        #     loss_components_1 = loss_components_1 / add_mask.sum()

        # 2. Degree distribution penalty
        orig_sub_adj = self.extended_sub_adj
        # edited_sub_adj = self.perturb_layer.build_perturbed_adj(self.extended_sub_adj, self.delta_A)
        # deg_diff = compute_deg_diff(orig_sub_adj, edited_sub_adj)

        cf_adj_soft = self.perturb_layer.ste_perturbed_adj(self.extended_sub_adj, self.full_mask)
        deg_diff = compute_deg_diff(orig_sub_adj, cf_adj_soft)

        loss_components_2 = deg_diff * self.α2

        # 3. penalty of clustering coefficients drastic changes
        # motif_violation = compute_motif_viol(orig_sub_adj, edited_sub_adj, self.tau_c)

        motif_violation = compute_motif_viol(orig_sub_adj, cf_adj_soft, self.tau_c)

        loss_components_3 = motif_violation * self.α3

        # 4. domain-specific constraint
        # publish_year = None
        # violation_count = 0
        # if add_mask.sum() > 0 and publish_year:
        #     target_year = publish_year[self.node_idx]
        #     for i in range(self.extended_sub_adj.size(0)):
        #         if add_mask[self.node_idx, i]:
        #             year_i = publish_year[i]
        #             # If the year of node i is earlier than the target node, but there is an edge from i to j, the rule is violated
        #             if (year_i < target_year) and self.extended_sub_adj[i, self.node_idx]:
        #                 violation_count += 1
        #             # Also check for the opposite case
        #             elif (target_year < year_i) and self.extended_sub_adj[self.node_idx, i]:
        #                 violation_count += 1
        #     sem_cost = violation_count / add_mask.sum()
        #     loss_components_4 = loss_components_4 + torch.tensor(sem_cost, dtype=float) * self.α4

        loss_components = loss_components_1 + loss_components_2 + loss_components_3 + loss_components_4

        return loss_components
