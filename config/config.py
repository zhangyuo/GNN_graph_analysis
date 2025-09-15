#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/8/2 10:35
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : config.py
# @Software : PyCharm
# @Desc     :
"""
# baseline model
TEST_MODEL = "GCN"  # ["GCN", "GraphSAGE", "GIN"]

# dataset
DATA_NAME = "Loan-Decision"  # ["cora", "BA-SHAPES", "TREE-CYCLES", "Loan-Decision", "ogbn-arxiv"]

# running device
DEVICE = 'cpu'  # ["cpu", "gpu"]

# random seed
SEED_NUM = 102  # first experiment is 102

################ Explainer parameters ################
EXPLANATION_TYPE = 'counterfactual'  # ['instance-level', 'class-level', 'counterfactual']
EXPLAINER_METHOD = 'CFExplainer'  # ['GNNExplainer', 'PGExplainer', 'CFExplainer', 'ACExplainer']
####################################################################

################ GCN model parameters for datasets ################
if DATA_NAME == "cora":
    HIDDEN_CHANNELS = 16
    DROPOUT = 0.5
    WITH_BIAS = True
    WEIGHT_DECAY = 0.01  # weight decay coefficient (l2 normalization) for GCN
    LEARNING_RATE = 0.01
    GCN_LAYER = 2
    GCN_EPOCHS = 500
elif DATA_NAME == "BA-SHAPES":
    HIDDEN_CHANNELS = 100
    DROPOUT = 0
    WITH_BIAS = True
    WEIGHT_DECAY = 0.001
    LEARNING_RATE = 0.001
    GCN_LAYER = 2
    GCN_EPOCHS = 20000
elif DATA_NAME == "TREE-CYCLES":
    HIDDEN_CHANNELS = 100
    DROPOUT = 0
    WITH_BIAS = True
    WEIGHT_DECAY = 0.001
    LEARNING_RATE = 0.001
    GCN_LAYER = 3
    GCN_EPOCHS = 5000
elif DATA_NAME == "Loan-Decision":
    HIDDEN_CHANNELS = 100
    DROPOUT = 0
    WITH_BIAS = True
    WEIGHT_DECAY = 0.001
    LEARNING_RATE = 0.001
    GCN_LAYER = 3
    GCN_EPOCHS = 5000
elif DATA_NAME == "ogbn-arxiv":
    HIDDEN_CHANNELS = 100
    DROPOUT = 0.5
    WITH_BIAS = True
    WEIGHT_DECAY = 0.001
    LEARNING_RATE = 0.01
    GCN_LAYER = 2
    GCN_EPOCHS = 5000
####################################################################

################ Attack model parameters for datasets in graph analysis ##############
ATTACK_TYPE = 'Evasion'  # ['Evasion', 'Poison'] our project only considers the evasion attack
ATTACK_METHOD = 'GOttack'  # GOttack
ATTACK_BUDGET_LIST = [5]  # [5,4,3,2,1] only can be used in attack graph analysis for similarity analysis
####################################################################

################ CFExplainer parameters for datasets ##############
BETA = 0.5  # Tradeoff for dist loss
NUM_EPOCHS = 500  # Num epochs for explainer
OPTIMIZER = "SGD"  # SGD or Adadelta
N_Momentum = 0.9  # Nesterov momentum
####################################################################

################ ACExplainer parameters for datasets ##############
LEARNING_RATE_AC = 10 ** -2  # learning rate for acexplainer training
MAX_ATTACK_NODES_NUM = 20  # max number of selected attacked nodes
NUM_EPOCHS_AC = 200  # Num epochs for explainer
OPTIMIZER_AC = "SGD"  # SGD or Adadelta or Adam
N_Momentum_AC = 0.9  # Nesterov momentum
LAMBDA_PRED = 1.0  # 预测损失权重
LAMBDA_DIST = 0.5  # 稀疏项惩罚权重
LAMBDA_PLAU = 0.5  # 现实惩罚项权重
MAX_EDITS = 5  # 最大扰动预算
TAU_PLUS = 0.5  # 加边阈值
TAU_MINUS = -0.5  # 减边阈值
α1 = 0  # 特征相似惩罚项权重0
α2 = 1.5  # 度变化惩罚项权重1.5
α3 = 1.0  # 聚类系数变化惩罚项权重1.0
α4 = 0  # 领域知识破坏惩罚项权重0
TAU_C = 0  # 聚类系数变化容忍度阈值0
####################################################################
k = 1  # 1
