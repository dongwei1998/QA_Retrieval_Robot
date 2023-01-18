#! /usr/bin/env python
# coding=utf-8


import os
import pickle
import tensorflow.compat.v1 as tf
from net import bidirectional_lstm

# 数据集参数
tf.flags.DEFINE_boolean("is_train", True,
                        "给定True或者False表示模型训练还是预测!!")
tf.flags.DEFINE_string('train_data_path',"./similarity_data/train.csv",
                       '训练数据集，默认: ./similarity_data/train.csv')
tf.flags.DEFINE_string('text_data_path',"./similarity_data/test.csv",
                       '验证数据集，默认: ./similarity_data/test.csv')

# 超参数
tf.flags.DEFINE_string('network_name',"bidirectional_lstm",
                       '使用的网络名称，默认: bidirectional_lstm')
tf.flags.DEFINE_bool("stop_word", False,
                      "是否对数据做停用词处理")
tf.flags.DEFINE_float('learning_rate',1e-3,
                      "学习率，默认为1e-3")
tf.flags.DEFINE_float('clip_grad',5,
                      "梯度裁剪的阈值")
tf.flags.DEFINE_integer('batch_size',256,
                        "每次训练的批次大小，默认为128")
tf.flags.DEFINE_integer('num_epochs',50,
                        "所有数据轮循的总次数，默认为20")
tf.flags.DEFINE_float("dropout_keep_prob", 0.85,
                      "神经元丢弃的概率，一般默认0.75")


# 模型参数
tf.flags.DEFINE_integer('embedding_size',128,
                        "embedding转换后的单词向量维度大小，默认为128")
tf.flags.DEFINE_integer('num_units',128,
                        "LSTM中神经元的个数，默认为128/")
tf.flags.DEFINE_integer('layers',3,
                        "LSTM的层次，默认为2")

tf.flags.DEFINE_string('fc_units',"[256, 512, 256]",
                       "对于LSTM输出值做FC全连接操作，全连接的神经元个数，可以是int或者list或者None")
tf.flags.DEFINE_integer('vector_size',128,
                        "最终转换得到的文本向量大小，默认为128")
tf.flags.DEFINE_boolean('load_mapping',True,
                        '是否加载映射词典，默认为True加载词典')
tf.flags.DEFINE_string('mapping_file',"./config/mapping_file.pkl",
                       '保存映射词典，默认./config/mapping_file.pkl')

# 模型持久化参数
tf.flags.DEFINE_integer('max_num_checkpoints',2,
                        "最多保存几个持久化模型，默认为2")
tf.flags.DEFINE_string('checkpoint_dir',"./running/model",
                       "模型持久化的路径")
tf.flags.DEFINE_integer('checkpoint_every',100,
                        "每多少步进模型的持久化")

# 模型可视化参数
tf.flags.DEFINE_string('summary_dir',"./running/graph",
                       "模型可视化保存路径")
tf.flags.DEFINE_string('train_summary_dir',"train",
                       "训练模型可视化保存路径")
tf.flags.DEFINE_string('dev_summary_dir',"dev",
                       "验证模型可视化保存路径")

# 参数解析一下
FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("参数解析:（先注释掉了！！！）")
# # 打印一下刚刚设置的参数
# for attr, value in sorted(FLAGS.flag_values_dict().items()):
#     print("{}={}".format(attr.upper(), value))
# print("")


word_2_idx, idx_2_word, vocab_size = pickle.load(open(FLAGS.mapping_file, 'rb'))


pb_file = "./config/bidirectional_lstm.pb"
ckpt_file = "./running/model/bidirectional_lstm"



output_node_names = ["input/input_data",
                     "bidirectional_lstm/network/output/embedding/vector",
                     "bidirectional_lstm/loss/similarity"]

def model_class(name):
    name_2_class = {
        "bidirectional_lstm": (bidirectional_lstm, "bidirectional_lstm"),
    }
    if name not in name_2_class:
        raise Exception("参数异常，网络名称可以选择的范围为:{}".format(name_2_class.keys()))
    return name_2_class[name]

with tf.name_scope('input'):
    input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_data')

with tf.variable_scope(FLAGS.network_name):
    # 2. 网络的前向过程构建
    with tf.variable_scope("network"):
        model, name = model_class(FLAGS.network_name)
        net = model.Network(input_tensor=input,
                            vocab_size=vocab_size,  # 总词汇的大小
                            embedding_size=FLAGS.embedding_size,  # embedding转换后的单词向量维度大小
                            num_units=FLAGS.num_units,  # LSTM中神经元的个数
                            layers=FLAGS.layers,  # lstm层数
                            fc_units=FLAGS.fc_units,  # 对于LSTM输出值做FC全连接操作，全连接的神经元个数
                            vector_size=FLAGS.vector_size,  # 最终转换得到的文本向量大小
                            dropout_keep_prob=FLAGS.dropout_keep_prob  # 神经元丢弃的概率
                            )

    # 3. 构建损失函数
    with tf.variable_scope("loss"):
        # a. 将文本转换得到的向量进行split拆分，还原文本信息
        # [N, vector_size], [N, vector_size]
        embedding1, embedding2 = tf.split(net.vector_embeddings, 2, 0)
        # b. 计算对应样本之间的相似度(只需要计算embedding1[i]和embedding2[i]的之间的相似度，其它不同行之间的相似度可以不考虑)
        # 最终得到N个相似度的值
        a = tf.reduce_sum(tf.multiply(embedding1, embedding2), axis=-1)  # [N,]
        b = tf.sqrt(tf.reduce_sum(tf.square(embedding1), axis=-1) + 1e-10)  # [N,]
        c = tf.sqrt(tf.reduce_sum(tf.square(embedding2), axis=-1) + 1e-10)  # [N,]
        similarity = tf.identity(a / tf.multiply(b, c), 'similarity')  # [N,], 取值范围: (-1,1)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state(ckpt_file)
if ckpt and ckpt.model_checkpoint_path:
    print("进行模型恢复操作")
    print("Restore model weight from '{}'".format(ckpt_file))
    # restore：进行模型恢复操作
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print("模型没有初始化好！！！")



converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                   input_graph_def=sess.graph.as_graph_def(),
                                                                   output_node_names=output_node_names)


with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())
