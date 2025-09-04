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
TEST_MODEL = "GCN"  # GCN

# dataset
DATA_NAME = "cora"

# running device
DEVICE = 'cpu'  # cpu or gpu

# random seed
SEED_NUM = 102

################ GCN model parameters for Cora data ################
HIDDEN_CHANNELS = 16
DROPOUT = 0.5
WITH_BIAS = True
WEIGHT_DECAY = 0.01  # weight decay coefficient (l2 normalization) for GCN
LEARNING_RATE = 0.01
GCN_LAYER = 2
####################################################################

################ Explainer parameters ################
EXPLANATION_TYPE = 'counterfactual'  # ['instance-level', 'class-level', 'counterfactual']
EXPLAINER_METHOD = 'CFExplainer'  # ['GNNExplainer', 'PGExplainer', 'CFExplainer', 'ACExplainer']
####################################################################

################ Attack model parameters for Cora data ##############
ATTACK_TYPE = 'Evasion'  # ['Evasion', 'Poison'] our project only considers the evasion attack
ATTACK_BUDGET_LIST = [5]  # [5,4,3,2,1]
ATTACK_METHOD = 'GOttack'  # GOttack
####################################################################

################ CFExplainer parameters for Cora data ##############
BETA = 0.5  # Tradeoff for dist loss
NUM_EPOCHS = 500  # Num epochs for explainer
OPTIMIZER = "SGD"  # SGD or Adadelta
N_Momentum = 0.9  # Nesterov momentum
####################################################################

################ ACExplainer parameters for Cora data ##############
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
α1 = 10  # 特征相似惩罚项权重-针对加边
α2 = 1  # 度变化惩罚项权重
α3 = 10  # 聚类系数变化惩罚项权重，保持同等量级
α4 = 20  # 领域知识破坏惩罚项权重-针对加边
TAU_C = 0.1  # 聚类系数变化容忍度阈值
####################################################################