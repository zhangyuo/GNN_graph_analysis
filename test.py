#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/6/24 14:55
# @Author   : Yu Zhang
# @Email    : yuzhang@cs.aau.dk
# @File     : test.py
# @Software : PyCharm
# @Desc     :
"""
import os
import sys
from deeprobust.graph.data import Dataset

if __name__ == '__main__':
    res = os.path.abspath(__file__)
    base_path = os.path.dirname(res)
    sys.path.insert(0, base_path)

    data = Dataset(root=base_path + '/dataset', name='cora')
    adj, features, labels = data.adj, data.features, data.labels