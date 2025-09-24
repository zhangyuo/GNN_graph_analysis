#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/9/4 17:17
# @Author   : **
# @Email    : **@**
# @File     : cf_evaluator.py
# @Software : PyCharm
# @Desc     :
"""
import os
import pickle
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score

# Configuration parameters (adjust as needed)
k = 10  # Scaling factor for plausibility sigmoid
GROUND_TRUTH_MOTIFS = None  # Replace with actual ground truth if available
FEATURE_SIM_THRESHOLD = 0.1  # Threshold for feature similarity
PUBLISH_YEAR = None  # Replace with actual publication year data if available


def load_results(results_path):
    """Load saved counterfactual explanation results"""
    with open(os.path.join(results_path, "test_cf_examples.pkl"), "rb") as f:
        cf_examples = pickle.load(f)
    with open(os.path.join(results_path, "cfexp_subgraph.pkl"), "rb") as f:
        cf_subgraphs = pickle.load(f)
    return cf_examples, cf_subgraphs


def compute_accuracy(cf_example, target_node):
    """
    Compute accuracy based on ground-truth motifs.
    Only applicable if ground-truth annotations exist.
    """
    if GROUND_TRUTH_MOTIFS is None:
        return 0.0  # Return 0 if no ground truth available

    motif_edges = set(GROUND_TRUTH_MOTIFS.get(target_node, []))
    cf_edges = set(cf_example["added_edges"] + cf_example["removed_edges"])

    # Calculate intersection of edges
    correct_edges = motif_edges.intersection(cf_edges)
    return len(correct_edges) / len(cf_edges) if cf_edges else 0.0


def compute_fidelity(cf_example):
    """Compute fidelity: 1 if prediction changed, 0 otherwise"""
    return 1 if cf_example["original_pred"] != cf_example["new_pred"] else 0


def compute_explanation_size(cf_example):
    """Compute number of edge modifications"""
    return len(cf_example["added_edges"]) + len(cf_example["removed_edges"])


def compute_sparsity(cf_example, original_adj):
    """Compute sparsity: proportion of edited edges"""
    delta_edges = compute_explanation_size(cf_example)
    return delta_edges / original_adj.nnz


def compute_plausibility(cf_example, features):
    """
    Compute plausibility score from sigmoid transformation of realism penalty.
    Higher scores indicate more plausible explanations.
    """
    # Feature similarity penalty
    feat_sim_penalty = 0
    target_feat = features[cf_example["target_node"]]
    for edge in cf_example["added_edges"]:
        neighbor = edge[1] if edge[0] == cf_example["target_node"] else edge[0]
        sim = F.cosine_similarity(
            torch.tensor(target_feat).unsqueeze(0),
            torch.tensor(features[neighbor]).unsqueeze(0)
        ).item()
        if sim < FEATURE_SIM_THRESHOLD:
            feat_sim_penalty += (1 - sim)

    # Degree distribution penalty
    orig_degrees = original_adj.sum(axis=1).A1
    new_degrees = orig_degrees.copy()
    for edge in cf_example["added_edges"]:
        new_degrees[edge[0]] += 1
        new_degrees[edge[1]] += 1
    for edge in cf_example["removed_edges"]:
        new_degrees[edge[0]] = max(0, new_degrees[edge[0]] - 1)
        new_degrees[edge[1]] = max(0, new_degrees[edge[1]] - 1)
    deg_diff = np.abs(new_degrees - orig_degrees) / (1 + orig_degrees)
    deg_penalty = np.sum(deg_diff)

    # Temporal constraint penalty
    temp_penalty = 0
    if PUBLISH_YEAR is not None:
        target_year = PUBLISH_YEAR[cf_example["target_node"]]
        for edge in cf_example["added_edges"]:
            neighbor = edge[1] if edge[0] == cf_example["target_node"] else edge[0]
            neighbor_year = PUBLISH_YEAR[neighbor]
            if (neighbor_year < target_year) and edge in cf_example["removed_edges"]:
                temp_penalty += 1
            elif (target_year < neighbor_year) and edge in cf_example["added_edges"]:
                temp_penalty += 1

    # Clustering coefficient penalty
    orig_cc = clustering_coefficient(original_adj)
    # Create modified adjacency matrix
    modified_adj = original_adj.copy().tolil()
    for edge in cf_example["added_edges"]:
        modified_adj[edge[0], edge[1]] = 1
        modified_adj[edge[1], edge[0]] = 1
    for edge in cf_example["removed_edges"]:
        modified_adj[edge[0], edge[1]] = 0
        modified_adj[edge[1], edge[0]] = 0
    modified_cc = clustering_coefficient(modified_adj.tocsr())

    # Calculate CC difference
    cc_diff = np.abs(modified_cc - orig_cc)
    cc_penalty = np.sum(np.clip(cc_diff - cf_example.get("tau_c", 0.1), 0, None))

    # Total plausibility loss (L_plau)
    L_plau = (
            cf_example.get("α1", 0.1) * feat_sim_penalty +
            cf_example.get("α2", 0.1) * deg_penalty +
            cf_example.get("α3", 0.1) * cc_penalty +
            cf_example.get("α4", 0.5) * temp_penalty
    )

    # Convert to plausibility score
    S_plau = 1 / (1 + np.exp(-k * L_plau))
    return S_plau


def clustering_coefficient(adj):
    """Compute clustering coefficients for all nodes"""
    G = nx.from_scipy_sparse_array(adj)
    return np.array(list(nx.clustering(G).values()))


def evaluate_explanations(cf_examples, features, original_adj, time_per_node):
    """Compute all evaluation metrics for a set of explanations"""
    metrics = {
        "accuracy": [],
        "fidelity": [],
        "explanation_size": [],
        "sparsity": [],
        "plausibility": [],
        "time_cost": time_per_node
    }

    for example in cf_examples:
        target_node = example["target_node"]

        # Compute all metrics
        metrics["accuracy"].append(compute_accuracy(example, target_node))
        metrics["fidelity"].append(compute_fidelity(example))
        metrics["explanation_size"].append(compute_explanation_size(example))
        metrics["sparsity"].append(compute_sparsity(example, original_adj))
        metrics["plausibility"].append(compute_plausibility(example, features))

    # Compute averages
    avg_metrics = {k: np.mean(v) if k != "time_cost" else v for k, v in metrics.items()}
    return avg_metrics


def fns_score(cf_examples, labels):
    """Compute F_NS score (node classification F1-score)"""
    y_true, y_pred = [], []
    for example in cf_examples:
        y_true.append(example["original_pred"])
        y_pred.append(example["new_pred"])
    return f1_score(y_true, y_pred, average="macro")


def charact_score(cf_examples):
    """Compute charact score (characterization accuracy)"""
    # This is dataset-specific and requires domain knowledge
    # Implement your characterization logic here
    return 0.0


def main():
    # # Configuration
    # results_path = "../results/2025-09-04/counterfactual_subgraph/attack_GOttack_counterfactual_ACExplainer_cora_budget[5,10]"
    # dataset_path = "../dataset"
    #
    # # Load data and results
    # data = Dataset(root=dataset_path, name="cora")
    # cf_examples, cf_subgraphs = load_results(results_path)

    # Compute evaluation metrics
    metrics = evaluate_explanations(
        cf_examples=cf_examples,
        features=data.features.toarray(),
        original_adj=data.adj,
        time_per_node=total_time / len(cf_examples)  # From main execution time
    )

    # Add alternative metrics
    metrics["fns_score"] = fns_score(cf_examples, data.labels)
    metrics["charact_score"] = charact_score(cf_examples)

    # Print results
    print("\n" + "=" * 50)
    print("ACExplainer Evaluation Results")
    print("=" * 50)
    print(f"{'Metric':<20} | {'Value':>10}")
    print("-" * 50)
    print(f"{'Accuracy':<20} | {metrics['accuracy']:>10.4f}")
    print(f"{'Fidelity':<20} | {metrics['fidelity']:>10.4f} (lower is better)")
    print(f"{'Explanation Size':<20} | {metrics['explanation_size']:>10.4f}")
    print(f"{'Sparsity':<20} | {metrics['sparsity']:>10.4f}")
    print(f"{'Plausibility':<20} | {metrics['plausibility']:>10.4f} (higher is better)")
    print(f"{'Time Cost (sec)':<20} | {metrics['time_cost']:>10.4f}")
    print(f"{'F_NS Score':<20} | {metrics['fns_score']:>10.4f}")
    print(f"{'Charact Score':<20} | {metrics['charact_score']:>10.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
