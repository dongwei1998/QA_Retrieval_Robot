from datetime import datetime
import os
import pickle
import tensorflow.compat.v1 as tf
from net import bidirectional_lstm
from utils import data_help
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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

# 其他参数
tf.flags.DEFINE_boolean('convert_bp',True,
                        '是否基于持久化好的模型转换pb文件')

# 参数解析一下
FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("参数解析:（先注释掉了！！！）")
# # 打印一下刚刚设置的参数
# for attr, value in sorted(FLAGS.flag_values_dict().items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

# 创建一些路径
def check_directory(path, create=True):
    flag = os.path.exists(path)
    if not flag:
        if create:
            os.makedirs(path)
            flag = True
    return flag

# 创建模型可视化以及持久化文件夹
def create_dir():
    train_summary_dir = os.path.join(FLAGS.summary_dir, FLAGS.network_name, FLAGS.train_summary_dir)
    dev_summary_dir = os.path.join(FLAGS.summary_dir, FLAGS.network_name, FLAGS.dev_summary_dir)
    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.network_name)
    model_file_name = os.path.join(FLAGS.network_name + ".ckpt")
    checkpoint_prefix = os.path.join(checkpoint_dir, model_file_name)

    # 以递归方式建立父目录及其子目录，如果目录已存在且是可覆盖则会创建成功，否则报错，无返回。
    if not os.path.exists(train_summary_dir):
        os.makedirs(train_summary_dir)# 原因：tf.gfile.MakeDirs要求路径的分隔符必须是/
    if not os.path.exists(dev_summary_dir):
        os.makedirs(dev_summary_dir)  # 原因：tf.gfile.MakeDirs要求路径的分隔符必须是/
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)  # 原因：tf.gfile.MakeDirs要求路径的分隔符必须是/

    # print("".format(train_summary_dir, dev_summary_dir, checkpoint_dir, model_file_name, checkpoint_prefix))

    return train_summary_dir, dev_summary_dir, checkpoint_dir, model_file_name, checkpoint_prefix

# 模型的选择
def model_class(name):
    name_2_class = {
        "bidirectional_lstm": (bidirectional_lstm, "bidirectional_lstm"),
    }
    if name not in name_2_class:
        raise Exception("参数异常，网络名称可以选择的范围为:{}".format(name_2_class.keys()))
    return name_2_class[name]



def train():
    # 必要文件夹创建
    train_summary_dir, dev_summary_dir, checkpoint_dir, model_file_name, checkpoint_prefix = create_dir()


    # 1.数据加载(X1和X2中是原始的分词后的信息)
    X1, X2, Y = data_help.read_data(data_files=FLAGS.train_data_path,stop_word=FLAGS.stop_word)

    #
    # q = data_help.question("./que_ans_data/question.csv")

    # 2. 构建单词和id之间的映射关系
    if FLAGS.load_mapping and os.path.exists(FLAGS.mapping_file):
        tf.logging.info("从磁盘进行单词映射关系等数据的加载恢复!!!!")
        word_2_idx, idx_2_word, vocab_size = pickle.load(open(FLAGS.mapping_file, 'rb'))
    else:
        tf.logging.info("基于训练数据重新构建映射关系，并将映射关系保存到磁盘路径中!!!")
        # 加载
        word_2_idx, idx_2_word, vocab_size = data_help.create_mapping(datasets=X1 + X2)
        # 保存
        with open(FLAGS.mapping_file, 'wb') as writer:
            pickle.dump((word_2_idx, idx_2_word, vocab_size), writer)

    # 3. 文本数据转换为id来表示(这个过程中不进行填充，但是对于不在词表中的数据来讲，使用UNKNOWN进行替换)
    X1 = data_help.convert_text_2_idx(X1, word_2_idx)
    X2 = data_help.convert_text_2_idx(X2, word_2_idx)

    # 4. 批次数据的获取
    batches = data_help.batch_iter(
        data=list(zip(X1, X2, Y)),
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs
    )

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # 一、执行图的构建
            with tf.variable_scope(FLAGS.network_name):
                # 1. 占位符的构建
                with tf.variable_scope("placeholder"):
                    input_x1 = tf.placeholder(tf.int32, shape=[None, None])  # [N,T]
                    input_x2 = tf.placeholder(tf.int32, shape=[None, None])  # [N,T]
                    labels = tf.placeholder(tf.int32, shape=[None])  # [N,]
                    # 将第一组和第二组合并
                    input = tf.concat([input_x1, input_x2], axis=0)  # [2N,T]
                # 2. 网络的前向过程构建
                with tf.variable_scope("network"):
                    model, name = model_class(FLAGS.network_name)
                    net = model.Network(input_tensor=input,
                                        vocab_size=vocab_size,  # 总词汇的大小
                                        embedding_size=FLAGS.embedding_size, # embedding转换后的单词向量维度大小
                                        num_units=FLAGS.num_units,          # LSTM中神经元的个数
                                        layers=FLAGS.layers,                # lstm层数
                                        fc_units=FLAGS.fc_units,            # 对于LSTM输出值做FC全连接操作，全连接的神经元个数
                                        vector_size=FLAGS.vector_size,      # 最终转换得到的文本向量大小
                                        dropout_keep_prob=FLAGS.dropout_keep_prob   # 神经元丢弃的概率
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

                    # # 欧式距离转换的相似度
                    # dist = tf.reduce_sum(tf.square(embedding1 - embedding2), axis=-1)  # [N,]
                    # similarity = tf.identity(1.0 / (dist + 1.0), 'similarity')  # [N,] 取值范围[0,1]

                    # c. 如果target实际为1，那么希望similarity越大越好，如果为0，希望越小越好
                    logits = tf.concat([
                        tf.expand_dims(0 - similarity, -1),  # [N, 1] --> 表示的是不相似的可能性
                        tf.expand_dims(similarity, -1)  # [N,1] --> 表示的是相似的可能性
                    ], axis=-1)  # [N,2]
                    loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
                    tf.losses.add_loss(loss)
                    loss = tf.losses.get_total_loss()
                    tf.summary.scalar('loss', loss)

                # 4. 构建准确率
                with tf.variable_scope("accuracy"):
                    predictions = tf.argmax(logits, axis=-1)
                    correct_predictions = tf.equal(predictions, tf.cast(labels, predictions.dtype))
                    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
                    tf.summary.scalar('accuracy', accuracy)

                # 5. 构建优化器
                with tf.variable_scope("train_op"):
                    global_step = tf.Variable(0, name="global_step", trainable=False)
                    optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
                    grads_and_vars = optim.compute_gradients(loss)
                    grads_and_vars_clip = [[tf.clip_by_value(g, -FLAGS.clip_grad, FLAGS.clip_grad), v] for g, v in
                                           grads_and_vars]
                    train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)

                # 6、可视化对象构建
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, tf.get_default_graph())
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, tf.get_default_graph())
                summary_op = tf.summary.merge_all()

                # 7. 持久化相关操作的构建
                # 记录全局参数
                saver = tf.train.Saver(max_to_keep=FLAGS.max_num_checkpoints)
                # 二、模型的迭代训练
                # 1.模型初始化恢复
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Restore model weight from '{}'".format(checkpoint_dir))
                # restore：进行模型恢复操作
                saver.restore(sess, ckpt.model_checkpoint_path)
                # recover_last_checkpoints：模型保存的时候，我们会保存多个模型文件，默认情况下，模型恢复的时候，磁盘文件不会进行任何操作，为了保证磁盘中最多只有max_to_keep个模型文件，故需要使用下列API
                saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
            # 2. 数据的遍历、然后进行训练
            def train_step(x1_batch, x2_batch, y_batch):
                feed_dict = {
                    input_x1: x1_batch,
                    input_x2: x2_batch,
                    labels: y_batch
                }
                _, step, summaries, _loss, _accuracy= sess.run(
                    [train_op, global_step, summary_op, loss, accuracy],
                    feed_dict)

                time_str = datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step,
                                                                                _loss, _accuracy))
                train_summary_writer.add_summary(summaries, step)
            # d. 迭代所有批次
            for batch in batches:
                # 1. 将x和y分割开
                x1_batch, x2_batch, y_batch = batch
                # 2. 训练操作
                train_step(x1_batch, x2_batch, y_batch)
                # 3. 获取当前的更新的次数
                current_step = tf.train.global_step(sess, global_step)
                # 4. 进行模型持久化输出
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            # e. 最终持久化
            path = saver.save(sess, checkpoint_prefix)
            print("Saved model checkpoint to {}\n".format(path))




def eval():
    # 必要文件夹创建
    train_summary_dir, dev_summary_dir, checkpoint_dir, model_file_name, checkpoint_prefix = create_dir()

    # 1.数据加载(X1和X2中是原始的分词后的信息)
    X1, X2, Y = data_help.read_data(data_files=FLAGS.train_data_path, stop_word=FLAGS.stop_word)

    # 2. 构建单词和id之间的映射关系

    tf.logging.info("从磁盘进行单词映射关系等数据的加载恢复!!!!")
    word_2_idx, idx_2_word, vocab_size = pickle.load(open(FLAGS.mapping_file, 'rb'))

    # 3. 文本数据转换为id来表示(这个过程中不进行填充，但是对于不在词表中的数据来讲，使用UNKNOWN进行替换)
    X1 = data_help.convert_text_2_idx(X1, word_2_idx)
    X2 = data_help.convert_text_2_idx(X2, word_2_idx)

    # 4. 批次数据的获取
    batches = data_help.batch_iter(
        data=list(zip(X1, X2, Y)),
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs
    )

    with tf.variable_scope(FLAGS.network_name):
        # 1. 占位符的构建
        with tf.variable_scope("placeholder"):
            input_x1 = tf.placeholder(tf.int32, shape=[None, None])  # [N,T]
            input_x2 = tf.placeholder(tf.int32, shape=[None, None])  # [N,T]
            labels = tf.placeholder(tf.int32, shape=[None])  # [N,]
            # 将第一组和第二组合并
            input = tf.concat([input_x1, input_x2], axis=0)  # [2N,T]
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

            # # 欧式距离转换的相似度
            # dist = tf.reduce_sum(tf.square(embedding1 - embedding2), axis=-1)  # [N,]
            # similarity = tf.identity(1.0 / (dist + 1.0), 'similarity')  # [N,] 取值范围[0,1]

            # c. 如果target实际为1，那么希望similarity越大越好，如果为0，希望越小越好
            logits = tf.concat([
                tf.expand_dims(0 - similarity, -1),  # [N, 1] --> 表示的是不相似的可能性
                tf.expand_dims(similarity, -1)  # [N,1] --> 表示的是相似的可能性
            ], axis=-1)  # [N,2]
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            tf.losses.add_loss(loss)
            loss = tf.losses.get_total_loss()
            tf.summary.scalar('loss', loss)
        # 4. 构建准确率
        with tf.variable_scope("accuracy"):
            predictions = tf.argmax(logits, axis=-1)
            correct_predictions = tf.equal(predictions, tf.cast(labels, predictions.dtype))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
            tf.summary.scalar('accuracy', accuracy)
        # 5. 构建优化器
        with tf.variable_scope("train_op"):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            grads_and_vars = optim.compute_gradients(loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -FLAGS.clip_grad, FLAGS.clip_grad), v] for g, v in
                                   grads_and_vars]


    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("进行模型恢复操作")
        print("Restore model weight from '{}'".format(checkpoint_dir))
        # restore：进行模型恢复操作
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("模型没有初始化好！！！")

    def dev_step(x1_batch, x2_batch, y_batch):
        feed_dict = {
            input_x1: x1_batch,
            input_x2: x2_batch,
            labels: y_batch
        }
        _logits, _loss, _accuracy = sess.run(
            [logits, loss, accuracy],
            feed_dict)
        return _loss, _accuracy

    _avg_loss = 0
    _avg_acc = 0
    count = 0
    for batch in batches:
        x1_batch, x2_batch, y_batch = batch
        _loss, _acc = dev_step(x1_batch, x2_batch, y_batch)
        _avg_loss += _loss
        _avg_acc += _acc
        count += 1
    print("AVG LOSS:{:g}, AVG ACC:{:g}".format(_avg_loss / count, _avg_acc / count))




def bp_convert():
    word_2_idx, idx_2_word, vocab_size = pickle.load(open(FLAGS.mapping_file, 'rb'))
    train_summary_dir, dev_summary_dir, checkpoint_dir, model_file_name, checkpoint_prefix = create_dir()
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

def main(_):
    if FLAGS.is_train:
        tf.logging.info("进行模型训练....")
        train()
    elif FLAGS.convert_bp:
        tf.logging.info("模型bp文件转换")
        bp_convert()
    else:
        tf.logging.info("进行模型的测试...")
        eval()


if __name__ == '__main__':
    tf.app.run()


