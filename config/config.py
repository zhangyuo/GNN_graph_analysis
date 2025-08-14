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

################ Explainer parameters ################
EXPLANATION_TYPE = 'counterfactual'  # ['instance-level', 'class-level', 'counterfactual']
EXPLAINER_METHOD = 'CFExplainer'  # ['GNNExplainer', 'PGExplainer', 'CFExplainer']
####################################################################

################ GCN model parameters for Cora data ################
HIDDEN_CHANNELS = 16
DROPOUT = 0.5
WITH_BIAS = True
WEIGHT_DECAY = 0.01  # weight decay coefficient (l2 normalization) for GCN
LEARNING_RATE = 0.01
GCN_LAYER = 2
####################################################################

################ Attack model parameters for Cora data ##############
ATTACK_TYPE = 'Evasion'  # ['Evasion', 'Poison']
ATTACK_BUDGET_LIST = [5]  # [5,4,3,2,1]
ATTACK_METHOD = 'GOttack'  # GOttack
####################################################################

################ CFExplainer parameters for Cora data ##############
BETA = 0.5  # Tradeoff for dist loss
NUM_EPOCHS = 500  # Num epochs for explainer
OPTIMIZER = "SGD"  # SGD or Adadelta
N_Momentum = 0.9  # Nesterov momentum
DROPOUT_CFEXP = 0.0
####################################################################
