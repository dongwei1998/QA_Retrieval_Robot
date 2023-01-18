#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
# ================================================================
from time import time

import jieba
import pickle

import numpy as np
import tensorflow.compat.v1 as tf
from utils import data_help
from tensorflow.python.platform import gfile


def read_pb_return_tensors(graph, pb_file, return_elements):
    with gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements


return_elements = [
    "input/input_data:0",
    "bidirectional_lstm/network/output/embedding/vector:0",
    "bidirectional_lstm/loss/similarity:0"
]
pb_file = "./config/bidirectional_lstm.pb"
mapping_file = "./config/mapping_file.pkl"
word_2_idx, idx_2_word, vocab_size = pickle.load(open(mapping_file, 'rb'))
text = "﻿怎么更改花呗手机号码"
text = jieba.lcut(text)
text = data_help.convert_text_2_idx([text], word_2_idx)  # [1,N]
graph = tf.Graph()
return_tensors = read_pb_return_tensors(graph, pb_file, return_elements)
start_time = time()
with tf.Session(graph=graph) as sess:
    vector = sess.run(return_tensors[1],
                      feed_dict={return_tensors[0]: text})
    print("=" * 100)
    print(np.shape(vector))
    print(vector)
