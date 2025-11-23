#!/usr/bin/env python
# coding:utf-8
"""
# @Time     : 2025/11/23 21:09
# @Author   : ***
# @Email    : ***@***.***.***
# @File     : translator.py
# @Software : PyCharm
# @Desc     :
"""
import os
import re
from googletrans import Translator  # pip install googletrans==4.0.0rc1

translator = Translator()

def is_chinese(text):
    return re.search('[\u4e00-\u9fff]', text)

def translate_comment(line):
    # match = re.match(r'(\s*#\s*)(.*)', line)
    match = re.search(r'(.*?#\s*)(.*)', line)
    if match:
        prefix, comment = match.groups()
        if is_chinese(comment):
            translated = translator.translate(comment, src='zh-CN', dest='en').text
            return prefix + translated + '\n'
    return line

root_dir = '/Users/zhangyu/Documents/PycharmProject/GNN_graph_analysis'

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(subdir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(translate_comment(line))
