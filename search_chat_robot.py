import pymysql
import pickle
import numpy as np
from flask import Flask, request, jsonify
import tensorflow.compat.v1 as tf
from sklearn.neighbors import KDTree
from utils import data_help
import os
import sqlite3
from tqdm import tqdm
import re


tf.flags.DEFINE_boolean("init",
                            False,
                            "是否做初始化操作，当设置为True的时候，进行初始化操作，"
                            "就是将所有的问题转换为向量然后保存到数据库中")
tf.flags.DEFINE_string("init_file",
                           "./que_ans_data/question.csv",
                           "给定初始化的时候，对应的初始文件路径!!!")
FLAGS = tf.flags.FLAGS


# 相似度的计算
def cosine_similarity(x, y):
    """
    计算两个向量x和y的相似度，要求x和y的维度大小是一致的
    :param x:
    :param y:
    :return:
    """
    # 1. 类型强制转换
    x = np.reshape(x, -1)
    y = np.reshape(y, -1)

    assert len(x) == len(y), "x和y大小不一致，不能计算相似度"
    # 2. 开始计算相似度
    a = np.sum(x * y)
    b = np.sqrt(np.sum(np.square(x))) * np.sqrt(np.sum(np.square(y)))
    return 1.0 * a / b



# 验证
def valid(a, max_len=0):
    if len(a) > 0 and contain_chinese(a):
        if max_len <= 0:
            return True
        elif len(a) <= max_len:  # 防止验证太长
            return True
    return False

def contain_chinese(s):
    # 是否包含中文
    if re.findall('[\u4e00-\u9fa5]+', s):
        return True
    return False

def insert(a, b, c,cur):
    cur.execute("""
    INSERT INTO question (questions, vectors, ans_ids) VALUES
    ('{}', '{}','{}')
    """.format(a.replace("'", ""), ','.join(map(str, b)),c))

def insert_if(question, answer,answoer_id, cur):
    if valid(question):
        insert(question, answer,answoer_id, cur)
        return 1
    return 0

class SimilarityPredictor(object):

    def __init__(self):
        mapping_file = "./config/mapping_file.pkl"
        print("恢复映射关系.....")
        self.word_2_idx, self.idx_2_word, self.vocab_size = pickle.load(open(mapping_file, 'rb'))

        print("深度学习模型参数恢复.....")
        self.return_elements = [
            "input/input_data:0",
            "bidirectional_lstm/network/output/embedding/vector:0",
            "bidirectional_lstm/loss/similarity:0"
        ]
        self.pb_file = "./config/bidirectional_lstm.pb"
        self.graph = tf.Graph()
        self.graph.as_default()
        self.return_tensors = self.read_pb_return_tensors()
        self.sess = tf.Session(graph=self.graph)

        print("开始进行数据库链接的相关获取操作....")
        self.conn = pymysql.connect(host="localhost", user="root", password="root",
                                    database="qa_bot", port=3306, charset="utf8")
        self.sqlite = sqlite3.connect('./config/qa_bot.db')

        self.select_sql = "SELECT vectors,answers FROM question join answer ON question.answer_id=answer.que_id"
        self.sqlect_sqlite3 ="SELECT vectors,answers FROM question as a join answer as b ON a.ans_id=b.que_id"
        # self.select_sql = "SELECT vectors,answer FROM tb_question a join tb_answer b ON a.answer_id=b.id"
        self.insert_sql = "INSERT INTO question(question,vectors,answer_id) VALUES(%s, %s, %s)"
        self.reload_kdtree_algo = True
        self.algo = None
        self.answers = None
        print("恢复成功!!!!")

    # 读取pb文件
    def read_pb_return_tensors(self):
        with tf.gfile.FastGFile(self.pb_file, 'rb') as f:
            frozen_graph_def = tf.GraphDef()
            frozen_graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            return_elements = tf.import_graph_def(frozen_graph_def,
                                                  return_elements=self.return_elements)
        return return_elements

        # 将文本转换为向量

    # 将文本转换为向量
    def convert_vector(self, text):
        """
        将文本转换为向量
        :param text:
        :return:
        """
        flag = False
        if isinstance(text, str):
            flag = True
            text = [text]

        vector = self.sess.run(self.return_tensors[1],
                               feed_dict={
                                   self.return_tensors[0]:data_help.pred_process_text(text, self.word_2_idx)
                               })
        if flag:
            return vector[0]
        else:
            return vector

    # 返回相似度
    def calc_similarity(self, text_a, text_b):
        if not isinstance(text_a, str):
            raise Exception("参数必须为字符串!!!")
        if not isinstance(text_b, str):
            raise Exception("参数必须为字符串!!!")
        text = [text_a, text_b]
        simi = self.sess.run(self.return_tensors[2],
                             feed_dict={
                                 self.return_tensors[0]: data_help.pred_process_text(text, self.word_2_idx)
                             })
        return simi

    # 原始的所有问题的信息转换为向量并保存数据库mysql（当init为True是使用）
    def insert_question_vectors(self, questions, vectors,answoer_ids):
        """
        原始的所有问题的信息转换为向量并保存
        :param questions:
        :param vectors:
        :return:
        """
        with self.conn.cursor() as cursor:
            for question, vector,answoer_id in zip(questions, vectors, answoer_ids):
                # print(question, vector,(question, ','.join(map(str, vector))),answoer_id)
                cursor.execute(
                    self.insert_sql, (question, ','.join(map(str, vector)),answoer_id)
                )

            self.conn.commit()

    #   原始的所有问题的信息转换为向量并保存数据库sqlite3（当init为True是使用）
    def sqlite3_conver(self, questions, vectors,answoer_ids):
        db = './config/qa_bot.db'
        # if os.path.exists(db):
        #     os.remove(db)
        # 如果db这个文件不存在则会创建这个文件  （SQlits数据库）
        if not os.path.exists(os.path.dirname(db)):
            os.makedirs(os.path.dirname(db))
        # 连接数据库
        conn = sqlite3.connect(db)
        # 创建游标
        cur = conn.cursor()
        # 如果这个表不存在则创建这个表
        cur.execute("""
                CREATE TABLE IF NOT EXISTS question
                (questions text, vectors text, ans_ids int);
                """)
        # 提交事务
        conn.commit()
        for question, vector, answoer_id in zip(questions, vectors, answoer_ids):
            # 将问答对的数据存到aqlite3中
            insert_if(question, vector,answoer_id, cur)

            # 批量提交
        conn.commit()

    # 使用KTtree加速
    def fetch_kdtree_algo(self):
        if self.reload_kdtree_algo:
            with self.conn.cursor() as cursor:
                print("数据库连接成功过！")
                cursor.execute(self.select_sql)
                X = []
                answers = []
                for tmp in cursor.fetchall():
                    X.append(list(map(float, tmp[0].split(","))))
                    answers.append(tmp[1])
            self.algo = KDTree(X, leaf_size=40)
            self.answers = answers
            self.reload_kdtree_algo = False
        return self.algo, self.answers

    # 使用sqliteKTtree加速
    def sqlite_fetch_kdtree_algo(self):
            if self.reload_kdtree_algo:
                with sqlite3.connect('./config/qa_bot.db') as conn:
                # 1. 获取数据库连接
                # conn = sqlite3.connect('./config/qa_bot.db')
                    # 2. 获取游标
                    cursor = conn.cursor()

                    sql = "SELECT vectors,answers FROM question join answer ON question.ans_ids=answer.que_ids"
                    value = cursor.execute(sql).fetchall()

                    # cursor.execute(self.sqlect_sqlite3)
                    X = []
                    answers = []
                    for tmp in value:
                        X.append(list(map(float, tmp[0].split(","))))
                        answers.append(tmp[1])
                self.algo = KDTree(X,leaf_size=10)
                self.answers = answers
                self.reload_kdtree_algo = False
            return self.algo, self.answers

    # 基于给定的question文本以及阈值threshold选择返回对应的answer
    def search_answer_by_question_with_kdtree(self, question, threshold):
        """
        基于给定的question文本以及阈值threshold选择返回对应的answer
        NOTE: 只能针对欧式距离训练的模型
        :param question:
        :param threshold:
        :return:
        """
        answer = None
        # 一、当前问题对应的128维的高阶向量
        vector = self.convert_vector(question)  # [N,]

        # 二、获取模型以及标签值
        algo, answers = self.fetch_kdtree_algo()
        print(algo, answers)

        # 三、模型获取最解决的id以及距离
        dist, ind = algo.query([vector], k=1)
        dist = dist[0][0]
        ind = ind[0][0]

        # 四、距离转换为相似度
        max_sim = 1.0 / (dist + 1.0)

        # 四、相似度的过滤
        if max_sim > threshold:
            # 最相似度的超过阈值，那么表示对应的answer就是我们需要的，否则返回None
            answer = answers[ind]
            tf.logging.info("问题:【{}】，最匹配的回复: 【{}】-【{}】，相关性为: 【{}】".format(
                question, ind, answer, max_sim))
        else:
            tf.logging.info("没有找到问题的最佳匹配，问题:【{}】，最大匹配相关性为: 【{}】-【{}】，阈值为: 【{}】".format(
                question, ind, max_sim, threshold))
        return max_sim, answer

    # 基于给定的question文本以及阈值threshold选择返回对应的answer
    def search_answer_by_question(self, question, threshold):
        """
        基于给定的question文本以及阈值threshold选择返回对应的answer
        :param question:
        :param threshold:
        :return:
        """

        answer = None
        # 一、当前问题对应的128维的高阶向量
        vector = self.convert_vector(question)

        # 二、获取数据库中所有问题对应的高阶向量
        with self.conn.cursor() as cursor:
            cursor.execute(self.select_sql)
            vectors = []
            answers = []
            for tmp in cursor.fetchall():
                vectors.append(list(map(float, tmp[0].split(","))))
                answers.append(tmp[1])

        # 三、比较vector和vectors中的所有向量，根据相似度的计算，选择相似度最高的
        max_sim = -1
        max_idx = -1
        for idx, other_vector in enumerate(vectors):
            # 1. 计算相似度
            sim = cosine_similarity(vector, other_vector)
            # 2. 相似度比较
            if sim > max_sim:
                max_sim = sim
                max_idx = idx

        # 四、相似度的过滤
        if max_sim > threshold:
            # 最相似度的超过阈值，那么表示对应的answer就是我们需要的，否则返回None
            answer = answers[max_idx]
            tf.logging.info("问题:【{}】，最匹配的回复: 【{}】-【{}】，相关性为: 【{}】".format(
                question, max_idx, answer, max_sim))
        else:
            tf.logging.info("没有找到问题的最佳匹配，问题:【{}】，最大匹配相关性为: 【{}】-【{}】，阈值为: 【{}】".format(
                question, max_idx, max_sim, threshold))
        return max_sim, answer

    # 基于给定的question文本以及阈值threshold选择返回对应的answer
    def sqlite3(self, question, threshold):
        """
        基于给定的question文本以及阈值threshold选择返回对应的answer
        :param question:
        :param threshold:
        :return:
        """

        answer = None
        # 一、当前问题对应的128维的高阶向量
        vector = self.convert_vector(question)
        # 1. 获取数据库连接
        conn = sqlite3.connect('./config/qa_bot.db')
        # 2. 获取游标
        cursor = conn.cursor()

        sql = "SELECT vectors,answers FROM question join answer ON question.ans_ids=answer.que_ids"
        value = cursor.execute(sql).fetchall()

        # cursor.execute(self.sqlect_sqlite3)
        vectors = []
        answers = []
        for tmp in value:
            vectors.append(list(map(float, tmp[0].split(","))))
            answers.append(tmp[1])

        # 三、比较vector和vectors中的所有向量，根据相似度的计算，选择相似度最高的
        max_sim = -1
        max_idx = -1
        for idx, other_vector in enumerate(vectors):
            # 1. 计算相似度
            sim = cosine_similarity(vector, other_vector)
            # 2. 相似度比较
            if sim > max_sim:
                max_sim = sim
                max_idx = idx

        # 四、相似度的过滤
        if max_sim > threshold:
            # 最相似度的超过阈值，那么表示对应的answer就是我们需要的，否则返回None
            answer = answers[max_idx]
            tf.logging.info("问题:【{}】，最匹配的回复: 【{}】-【{}】，相关性为: 【{}】".format(
                question, max_idx, answer, max_sim))
        else:
            tf.logging.info("没有找到问题的最佳匹配，问题:【{}】，最大匹配相关性为: 【{}】-【{}】，阈值为: 【{}】".format(
                question, max_idx, max_sim, threshold))
        return max_sim, answer

    # 基于给定的question文本以及阈值threshold选择返回对应的answer
    def sqlite3_KDTree(self, question, threshold):
            """
            基于给定的question文本以及阈值threshold选择返回对应的answer
            :param question:
            :param threshold:
            :return:
            """
            answer = None
            # 一、当前问题对应的128维的高阶向量
            vector = self.convert_vector(question)  # [N,]

            # 二、获取模型以及标签值
            algo, answers = self.sqlite_fetch_kdtree_algo()

            # 三、模型获取最解决的id以及距离
            dist, ind = algo.query([vector], k=1)
            dist = dist[0][0]
            ind = ind[0][0]

            # 四、距离转换为相似度
            max_sim = 1.0 / (dist + 1.0)

            # 四、相似度的过滤
            if max_sim > threshold:
                # 最相似度的超过阈值，那么表示对应的answer就是我们需要的，否则返回None
                answer = answers[ind]
                tf.logging.info("问题:【{}】，最匹配的回复: 【{}】-【{}】，相关性为: 【{}】".format(
                    question, ind, answer, max_sim))
            else:
                tf.logging.info("没有找到问题的最佳匹配，问题:【{}】，最大匹配相关性为: 【{}】-【{}】，阈值为: 【{}】".format(
                    question, ind, max_sim, threshold))
            return max_sim, answer



if __name__ == '__main__':
    # 一、构建对象
    tf.logging.set_verbosity(tf.logging.DEBUG)
    predictor = SimilarityPredictor()

    # 二、根据参数决定是否进行初始化操作
    if FLAGS.init:
        if tf.gfile.Exists(FLAGS.init_file):

            # 1. 加载所有数据
            X = []
            ID = []
            with open(FLAGS.init_file, 'r', encoding='utf-8-sig') as reader:
                tmp_x = []
                tmp_id = []
                num = 0
                for line in reader:
                    q_id,text = line.strip().split(",")
                    tmp_x.append(text)
                    tmp_id.append(q_id)
                    num+=1
                    if len(tmp_x) >= 1000 and len(tmp_id)>=1000:
                        X.append(tmp_x)
                        ID.append(tmp_id)
                        tmp_x = []
                        tmp_id = []
                if len(tmp_x) > 0 and len(tmp_id)>0:
                    X.append(tmp_x)
                    ID.append(tmp_id)

            # 2. 遍历所有的文本，然后获取对应的向量
            for datas,IDS in tqdm(zip(X,ID),total=len(X)):
                # a. 得到当前批次的文本对应的向量
                vectors = predictor.convert_vector(datas)
                # print(vectors)
                # b. 将文本和向量转换后输出到数据库中
                predictor.sqlite3_conver(datas, vectors,IDS)
                predictor.insert_question_vectors(datas, vectors,IDS)
            print("{}条数据插入成功！！".format(num))

    # APP应用构建
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False


    @app.route('/')
    @app.route('/index')
    def _index():
        return "你好，欢迎使用Flask Web API，检索式机器人!!!"


    @app.route('/similarity', methods=['POST'])
    def similarity():
        tf.logging.info("基于给定的文本，计算两个文本的相似度.....")
        try:
            # 参数获取
            text_a = request.form.get("text1")
            text_b = request.form.get("text2")

            # 参数检查
            if text_a is None:
                return jsonify({
                    'code': 501,
                    'msg': '请给定参数text1！！！'
                })
            if text_b is None:
                return jsonify({
                    'code': 501,
                    'msg': '请给定参数text2！！！'
                })

            # 直接调用预测的API
            simi = float(predictor.calc_similarity(text_a, text_b))
            return jsonify({
                'code': 200,
                'msg': '成功',
                'data': [
                    {
                        'text1': text_a,
                        'text2': text_b,
                        'similarity': simi
                    }
                ]
            })
        except Exception as e:
            tf.logging.error("异常信息为:{}".format(e))
            return jsonify({
                'code': 502,
                'msg': '预测数据失败, 异常信息为:{}'.format(e)
            })

    @app.route('/sqlite3_bot',methods=['POST'])
    def sqlite3_bot():
        tf.logging.info("基于给定的问题，使用检索的方式从sqlite3数据库中获取匹配的回复.....")
        try:
            # 参数获取
            question = request.form.get("question")
            threshold = float(request.form.get("threshold", "0.9"))
            # 参数检查
            if question is None:
                return jsonify({
                    'code': 501,
                    'msg': '请给定参数question！！！'
                })
            # 直接调用预测的API
            max_similarity, answer = predictor.sqlite3(question, threshold)
            if answer is None:
                return jsonify({
                    'code': 505,
                    'msg': '没有匹配的问题,最大相似度为:{}'.format(max_similarity)
                })
            else:
                return jsonify({
                    'code': 200,
                    'msg': '成功',
                    'data': [
                        {
                            'answer': answer,
                            'question': question,
                            'max_similarity': float(max_similarity)
                        }
                    ]
                })
        except Exception as e:
            tf.logging.error("异常信息为:{}".format(e))
            return jsonify({
                'code': 502,
                'msg': '预测数据失败, 异常信息为:{}'.format(e)
            })


    @app.route('/sqlite3_bot_KDTree', methods=['POST'])
    def sqlite3_bot_KDTree():
        tf.logging.info("基于给定的问题，使用检索的方式从sqlite3数据库中获取匹配的回复.....")
        try:
            # 参数获取
            question = request.form.get("question")
            threshold = float(request.form.get("threshold", "0.9"))
            # 参数检查
            if question is None:
                return jsonify({
                    'code': 501,
                    'msg': '请给定参数question！！！'
                })
            # 直接调用预测的API
            max_similarity, answer = predictor.sqlite3_KDTree(question, threshold)
            if answer is None:
                return jsonify({
                    'code': 505,
                    'msg': '没有匹配的问题,最大相似度为:{}'.format(max_similarity)
                })
            else:
                return jsonify({
                    'code': 200,
                    'msg': '成功',
                    'data': [
                        {
                            'answer': answer,
                            'question': question,
                            'max_similarity': float(max_similarity)
                        }
                    ]
                })
        except Exception as e:
            tf.logging.error("异常信息为:{}".format(e))
            return jsonify({
                'code': 502,
                'msg': '预测数据失败, 异常信息为:{}'.format(e)
            })


    @app.route('/chat_bot', methods=['POST'])
    def chat_robot():
        tf.logging.info("基于给定的问题，使用检索的方式从mysql数据库中获取匹配的回复.....")
        try:
            # 参数获取
            question = request.form.get("question")
            threshold = float(request.form.get("threshold", "0.9"))
            # 参数检查
            if question is None:
                return jsonify({
                    'code': 501,
                    'msg': '请给定参数question！！！'
                })
            # 直接调用预测的API
            max_similarity, answer = predictor.search_answer_by_question(question, threshold)
            print(max_similarity, answer)
            if answer is None:
                return jsonify({
                    'code': 505,
                    'msg': '没有匹配的问题,最大相似度为:{}'.format(max_similarity)
                })
            else:
                print(4)
                return jsonify({
                    'code': 200,
                    'msg': '成功',
                    'data': [
                        {
                            'question': question,
                            'answer': answer,
                            'max_similarity': float(max_similarity)
                        }
                    ]
                })
        except Exception as e:
            tf.logging.error("异常信息为:{}".format(e))
            return jsonify({
                'code': 502,
                'msg': '预测数据失败, 异常信息为:{}'.format(e)
            })


    @app.route('/chat_bot_KDTree', methods=['POST'])
    def chat_bot_KDTree():
        tf.logging.info("基于给定的问题，使用检索的方式从数据库中获取匹配的回复（KDTree加速）.....")
        try:
            # 参数获取
            question = request.form.get("question")
            threshold = float(request.form.get("threshold", "0.9"))

            # 参数检查
            if question is None:
                return jsonify({
                    'code': 501,
                    'msg': '请给定参数question！！！'
                })
            # 直接调用预测的API
            max_similarity, answer = predictor.search_answer_by_question_with_kdtree(question, threshold)
            if answer is None:
                return jsonify({
                    'code': 505,
                    'msg': '没有匹配的问题,最大相似度为:{}'.format(max_similarity)
                })
            else:
                return jsonify({
                    'code': 200,
                    'msg': '成功',
                    'data': [
                        {
                            'question': question,
                            'answer': answer,
                            'max_similarity': float(max_similarity)
                        }
                    ]
                })

        except Exception as e:
            tf.logging.error("异常信息为:{}".format(e))
            return jsonify({
                'code': 502,
                'msg': '预测数据失败, 异常信息为:{}'.format(e)
            })


    # 启动
    app.run(host='0.0.0.0', port=5000)
