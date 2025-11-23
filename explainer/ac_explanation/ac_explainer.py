#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/8/27 20:16
# @Author   : **
# @Email    : **@**
# @File     : ac_explainer.py
# @Software : PyCharm
# @Desc     :
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from config.config import PRUNING
from utilty.utils import normalize_adj
from explainer.ac_explanation.gcn_perturb import GNNPerturb


class ACExplainer:
    def __init__(self,
                 model: nn.Module,
                 target_node: int,
                 node_idx: int,
                 node_num_l_hop: list,
                 extended_sub_adj: torch.Tensor,
                 sub_feat: torch.Tensor,
                 sub_labels: torch.Tensor,
                 y_pred_orig: torch.Tensor,
                 nclass: int,
                 nhid: int = 16,
                 dropout: float = 0.5,
                 lambda_pred: float = 1.0,
                 lambda_dist: float = 0.5,
                 lambda_plau: float = 0.2,
                 epoch: int = 200,
                 optimizer: str = 'SGD',
                 n_momentum: float = 0.9,
                 lr: float = 0.01,
                 top_k: int = 5,
                 C: float = 1.0,
                 tau_plus: float = 0.5,
                 tau_minus: float = -0.5,
                 α1: float = 0.1,
                 α2: float = 0.1,
                 α3: float = 0.1,
                 α4: float = 0.5,
                 tau_c: float = 0.1,
                 device: str = "cuda",
                 gcn_layer: int = 2,
                 with_bias: bool = True,
                 test_model: str = "GCN",
                 heads: int = 2,
                 dataset_name: str = "cora"):
        # Move models and data to a specified device
        self.model = model.to(device)
        self.model.eval()
        self.extended_sub_adj = extended_sub_adj.to(device)
        self.sub_feat = sub_feat.to(device)

        self.n_hid = nhid
        self.dropout = dropout
        self.sub_labels = sub_labels
        self.y_pred_orig = y_pred_orig

        # loss weight
        self.lambda_pred = lambda_pred  # Predicted loss weight
        self.lambda_dist = lambda_dist  # Sparse loss weight
        self.lambda_plau = lambda_plau  # Reality Loss Weight

        self.num_classes = nclass
        self.target_node = target_node
        self.node_idx = node_idx
        self.node_num_l_hop = node_num_l_hop
        self.epoch = epoch
        self.optimizer_type = optimizer
        self.n_momentum = n_momentum
        self.lr = lr
        self.top_k = top_k
        self.C = C
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.α1 = α1
        self.α2 = α2
        self.α3 = α3
        self.α4 = α4
        self.tau_c = tau_c
        self.device = device
        self.gcn_layer = gcn_layer
        self.with_bias = with_bias
        self.test_model = test_model

        # Create a perturbation model
        self.cf_model = GNNPerturb(
            nfeat=self.sub_feat.size(1),
            nhid=self.n_hid,
            nclass=self.num_classes,
            extended_sub_adj=self.extended_sub_adj,
            sub_feat=self.sub_feat,
            node_idx=self.node_idx,
            node_num_l_hop=self.node_num_l_hop,
            dropout=dropout,
            lambda_pred=self.lambda_pred,
            lambda_dist=self.lambda_dist,
            lambda_plau=self.lambda_plau,
            top_k=self.top_k,
            C=self.C,
            tau_plus=self.tau_plus,
            tau_minus=self.tau_minus,
            α1=self.α1,
            α2=self.α2,
            α3=self.α3,
            α4=self.α4,
            tau_c=self.tau_c,
            gcn_layer=gcn_layer,
            with_bias=with_bias,
            test_model=test_model,
            heads=heads,
            dataset_name=dataset_name
        ).to(device)

        # Inherit original model parameters
        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

        # if test_model == "GCN":
        # Freeze weights from original model in cf_model Freeze the original parameters and only train the perturbation matrix
        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias") or name.endswith("att_src") or name.endswith("att_dst") or name.endswith("att_edge"):
            # if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            print("orig model requires_grad: ", name, param.requires_grad)
        for name, param in self.cf_model.named_parameters():
            print("cf model requires_grad: ", name, param.requires_grad)

        # Optimizer settings
        if self.optimizer_type == "SGD" and self.n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=self.lr)
        elif self.optimizer_type == "SGD" and self.n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=self.lr, nesterov=True,
                                          momentum=self.n_momentum)
        elif self.optimizer_type == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=self.lr)
        elif self.optimizer_type == "Adam":
            self.cf_optimizer = optim.Adam([self.cf_model.get_mask_parameters()], lr=self.lr)

        # lr adaptive
        self.scheduler = optim.lr_scheduler.StepLR(self.cf_optimizer, step_size=100, gamma=0.5)

    def explain(self) -> dict:
        """Train the explainer"""
        best_loss = float('inf')
        best_delta_A = None
        best_pred = None
        best_cf_adj = None
        best_plau_loss = None
        no_improve = 0
        best_loss_1 = float('inf')
        best_delta_A_1 = None
        best_pred_1 = None
        best_cf_adj_1 = None
        best_plau_loss_1 = None

        self.cf_model.eval()  # The counterfactual model g training phase uses evaluation mode, freezing dropout and batchnorm

        for epoch in tqdm(range(self.epoch)):
            # print(f"\n######## epoch: {epoch + 1} #############")
            self.cf_optimizer.zero_grad()

            # forward propagation
            output = self.cf_model.forward(self.sub_feat, self.extended_sub_adj)  # Differentiable prediction
            output_actual = self.cf_model.forward_prediction(self.sub_feat)  # discrete forecast

            y_pred_new = torch.argmax(output[self.node_idx])
            y_pred_new_actual = torch.argmax(output_actual[self.node_idx])

            # Calculate losses
            total_loss, pred_loss, dist_loss, plau_loss, cf_adj, delta_A, perturb_layer, full_mask = self.cf_model.compute_losses(
                output,
                self.y_pred_orig,
                y_pred_new_actual)

            # Backpropagation
            total_loss.backward()

            # Check mask parameter gradient
            print("M.grad:", self.cf_model.get_mask_parameters().grad)
            if self.cf_model.get_mask_parameters().grad is not None:
                print(f"Mask grad norm: {self.cf_model.get_mask_parameters().grad.norm().item()}")

            clip_grad_norm_(self.cf_model.parameters(), 2.0)  # Clipping gradient magnitude
            self.cf_optimizer.step()
            # if epoch % 20 == 0:
            print('Target node: {}'.format(self.target_node),
                  'New idx: {}'.format(self.node_idx),
                  'Epoch: {:04d}'.format(epoch + 1),
                  'loss: {:.4f}'.format(total_loss.item()),
                  'pred loss: {:.4f}'.format(pred_loss.item()),
                  'dist loss: {:.4f}'.format(dist_loss.item()),
                  'plau loss: {:.4f}'.format(plau_loss.item()))
            print('Output: {}\n'.format(output[self.node_idx].data),
                  'Output nondiff: {}\n'.format(output_actual[self.node_idx].data),
                  'orig pred: {}, new pred: {}, new pred nondiff: {}'.format(self.y_pred_orig, y_pred_new,
                                                                             y_pred_new_actual))
            print(" ")
            # Early stop inspection
            if y_pred_new_actual != self.y_pred_orig and total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_delta_A = delta_A.detach().clone()
                best_pred = y_pred_new_actual
                best_cf_adj = cf_adj
                best_plau_loss = plau_loss
                no_improve = 0
            elif y_pred_new_actual != self.y_pred_orig:
                no_improve += 1
            elif y_pred_new_actual == self.y_pred_orig and total_loss.item() < best_loss_1:
                best_loss_1 = total_loss.item()
                best_delta_A_1 = delta_A.detach().clone()
                best_pred_1 = y_pred_new_actual
                best_cf_adj_1 = cf_adj
                best_plau_loss_1 = plau_loss

            if no_improve > 20:  # Stop early
                break
            self.scheduler.step()

        print("Start minimality pruning")
        if best_delta_A is not None:
            # post-pruning
            if PRUNING:
                pruned_delta_A = self.minimality_pruning(best_delta_A, perturb_layer, full_mask)
                final_result = {
                    "success": True,
                    "delta_A": pruned_delta_A,  # Perturbation after using pruning
                    "cf_adj": perturb_layer.build_perturbed_adj(
                        self.extended_sub_adj,
                        pruned_delta_A
                    ),
                    "original_pred": self.y_pred_orig,
                    "new_pred": self._validate_pruning(pruned_delta_A, perturb_layer),  # Validate predictions
                    "plau_loss": best_plau_loss
                }
            else:
                final_result = {
                    "success": True,
                    "delta_A": best_delta_A,
                    "cf_adj": best_cf_adj,
                    "original_pred": self.y_pred_orig,
                    "new_pred": best_pred,
                    "plau_loss": best_plau_loss
                }
        else:
            final_result = {
                "success": False,
                "delta_A": best_delta_A_1,
                "cf_adj": best_cf_adj_1,
                "original_pred": self.y_pred_orig,
                "new_pred": best_pred_1,  # Validate predictions
                "plau_loss": best_plau_loss_1
            }

        print("Finish minimality pruning")

        return final_result

    def minimality_pruning(self, delta_A: torch.Tensor, perturb_layer, full_mask) -> torch.Tensor:
        current_delta = delta_A.clone()
        edge_indices = torch.nonzero(current_delta != 0)

        if edge_indices.size(0) == 0:
            return current_delta

        # 1. Get the full_mask value of all edges to be pruned
        edge_mask_values = full_mask[edge_indices[:, 0], edge_indices[:, 1]]

        # 2. Build a custom sort key: add edges (positive values) first, then subtract edges (negative values)
        #    Tip: Give the plus edge a large offset (e.g. +1000) to ensure it comes before the minus edge
        #    While maintaining descending order of absolute value within the group
        sorting_key = torch.where(
            edge_mask_values > 0,
            1000 + torch.abs(edge_mask_values),  # Add edge group: key value range [1000, 1000+max_abs]
            torch.abs(edge_mask_values)  # Subtraction edge group: key value range [0, max_abs]
        )

        # 3. Sort in descending order by custom key -> add edges (high key value) in front, subtract edges (low key value) in the back; the ones with the largest absolute value in the group are in front
        # Sort by disturbance intensity (the larger the absolute value, the more important)
        sorted_indices = torch.argsort(sorting_key, descending=True)  # Sort descending

        # abs_values = torch.abs(full_mask[edge_indices[:, 0], edge_indices[:, 1]])
        # sorted_indices = torch.argsort(abs_values, descending=True) # Sort in descending order

        # 4. Try to remove starting from the least important edge (low absolute value)
        for idx in sorted_indices:
            i, j = edge_indices[idx]
            temp_delta = current_delta.clone()
            temp_delta[i, j] = 0
            temp_delta[j, i] = 0  # Symmetric treatment

            # Verify that the prediction is still flipped
            if self._validate_flip(temp_delta, perturb_layer):
                current_delta = temp_delta  # Keep removal

        return current_delta

    def _validate_flip(self, delta_A, perturb_layer):
        perturbed_adj = perturb_layer.build_perturbed_adj(
            self.extended_sub_adj,
            delta_A
        )
        norm_adj = normalize_adj(perturbed_adj)
        with torch.no_grad():
            if self.test_model == "GCN":
                output = self.model(self.sub_feat, norm_adj)
            elif self.test_model in ["GraphTransformer", "GraphConv", "GAT"]:
                edge_index, edge_weight = dense_to_sparse(norm_adj)
                output = self.model(self.sub_feat, edge_index, edge_weight=edge_weight)
            else:
                output = None
            return output[self.node_idx].argmax() != self.y_pred_orig

    def compute_edge_importance(self, delta_A: torch.Tensor) -> torch.Tensor:
        """Calculate the importance score of the edges (based on gradient sensitivity)"""
        self.model.zero_grad()

        # Calculate prediction loss
        output = self.model(self.sub_feat)
        # pred_loss = -F.nll_loss(
        #     output[self.node_idx].unsqueeze(0),
        #     torch.tensor(self.y_pred_orig, device=self.device)
        # ) * (output[self.node_idx].unsqueeze(0) == self.y_pred_orig).float()
        pred_loss = -F.nll_loss(
            output[self.node_idx].unsqueeze(0),
            torch.tensor([self.y_pred_orig], device=self.device)
        )

        # Backpropagation to obtain gradients
        pred_loss.backward()
        mask_grad = self.model.get_mask_parameters().grad.abs()

        # Reconstruct the full-size gradient matrix
        grad_matrix = torch.zeros_like(delta_A)
        edge_idx = 0
        for i in range(self.original_adj.size(0)):
            if i != self.node_idx:
                grad_matrix[self.node_idx, i] = mask_grad[edge_idx]
                grad_matrix[i, self.node_idx] = mask_grad[edge_idx]
                edge_idx += 1

        return grad_matrix

    def _validate_pruning(self, delta_A, perturb_layer):
        print("Start validating flip")
        with torch.no_grad():
            perturbed_adj = perturb_layer.build_perturbed_adj(
                self.extended_sub_adj,
                delta_A
            )
            norm_adj = normalize_adj(perturbed_adj)

            if self.test_model == "GCN":
                output = self.model(self.sub_feat, norm_adj)
            elif self.test_model in ["GraphTransformer", "GraphConv", "GAT"]:
                edge_index, edge_weight = dense_to_sparse(norm_adj)
                output = self.model(self.sub_feat, edge_index, edge_weight=edge_weight)
            else:
                output = None

            print("Finish validating flip")
            return output[self.node_idx].argmax()
