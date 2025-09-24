#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/7/15 19:37
# @Author   : **
# @Email    : **@**
# @File     : evasion_PGDattack.py
# @Software : PyCharm
# @Desc     :
"""
import os
import sys
import random
from deeprobust.graph.data import Dataset

if __name__ == '__main__':
    res = os.path.abspath(__file__)  # acquire absolute path of current file
    base_path = os.path.dirname(
        os.path.dirname(os.path.dirname(res)))  # acquire the parent path of current file's parent path
    sys.path.insert(0, base_path)

    method = ['PGD']
    # budget_list = [5, 4, 3, 2, 1]
    budget_list = [5]
    random.seed(102)
    dataset_name = 'cora'
    device = "cpu"

    print("INFO: Applying adversarial techniques {} on {} dataset with perturbation budget {} ".format(method,
                                                                                                       dataset_name,
                                                                                                       budget_list))

    ######################### Loading dataset  #########################
    data = Dataset(root=base_path + '/dataset', name=dataset_name)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
