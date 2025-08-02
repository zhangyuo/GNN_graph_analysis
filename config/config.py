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
# GCN model parameters for Cora data
HIDDEN_CHANNELS = 16
DROP_OUT = 0.5
WITH_BIAS = True

# CFExplainer parameters
BETA = 0.5  # Tradeoff for dist loss
LEARNING_RATE = 0.01
NUM_EPOCHS = 500  # Num epochs for explainer
OPTIMIZER = "SGD"  # SGD or Adadelta
N_Momentum = 0.9  # Nesterov momentum
