#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/20 12:10
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : gcn_model.py
# @Software : PyCharm
# @Desc     :
"""
from deeprobust.graph.defense import GCN


def GCN_model(adj, features, labels, device, idx_train, idx_val, target_gcn=None):
    '''
    GCN model
    '''

    if target_gcn is None:
        target_gcn = GCN(nfeat=features.shape[1],
                         nhid=16,
                         nclass=labels.max().item() + 1,
                         with_bias=False,
                         dropout=0.5, device=device)
        target_gcn = target_gcn.to(device)
        target_gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    target_gcn.eval()
    output = target_gcn.predict()
    return target_gcn, output
