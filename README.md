
# ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks

<p align="center">
  <img src="figs/architectureDiagram.png.igr" alt="Method" width="800"/>
</p>

---

## Overview  

ATEX-CF is a novel framework that unifies adversarial attack strategies with counterfactual explanation generation for Graph Neural Networks. By integrating both edge additions and deletions within a constrained budget, it produces minimal, plausible, and highly effective explanations that outperform deletion-only and attack-only baselines across multiple benchmarks.

This repository provides the implementation of ATEX-CF, including scripts for model training, explanation generation, and evaluation, to facilitate reproducibility and further research.  

---

## Abstract  

Counterfactual explanations offer an intuitive way to interpret graph neural networks (GNNs) by identifying minimal changes that alter a model’s prediction, thereby answering “what must differ for a different outcome?”. 

In this work, we propose a novel framework, ATEX-CF that unifies adversarial attack techniques with counterfactual explanation generation—a connection made feasible by their shared goal of flipping a node’s prediction, yet differing in perturbation strategy: adversarial attacks often rely on edge additions, while counterfactual methods typically use deletions.

Unlike traditional approaches that treat explanation and attack separately, our method efficiently integrates both edge additions and deletions, grounded in theory, leveraging adversarial insights to explore impactful counterfactuals.

In addition, by jointly optimizing fidelity, sparsity, and plausibility under a constrained perturbation budget, our method produces instance-level explanations that are both informative and realistic.
Experiments on synthetic and real-world node classification benchmarks demonstrate that ATEX-CF generates faithful, concise, and plausible explanations, highlighting the effectiveness of integrating adversarial insights into counterfactual reasoning for GNNs.

---

## Dataset  

XX

---


## Installation  

### Prerequisites  
- Python 3.10+  
- Libraries listed in `requirements.txt`  

### Steps  

1. Install PyTorch  
```bash
XX
```

2. install PyG

```
XX
```

3. install PyTorch Geonetric Temporal (optional)

```
XX
```
### Prerequisites

- Python 3.10+
- Libraries listed in `installed_packages.txt`



