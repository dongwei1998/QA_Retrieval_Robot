# -- encoding:utf-8 --

import sqlite3
from tqdm import tqdm
import json
import numpy as np


if __name__ == '__main__':
    # 1. 获取数据库连接
    conn = sqlite3.connect('../config/qa_bot.db')
    # 2. 获取游标
    cursor = conn.cursor()
    try:
        # 3. 业务执行
        # 3.1 查询数据
        # sql = "SELECT MAX(ROWID) FROM answer;"
        # number = cursor.execute(sql).fetchone()[0]
        # print("总数据量为:{}".format(number))
        # cursor.execute(sql)
        # 3.2 插入语句
        # sql = """
        # INSERT INTO qa_data (ask, answer) VALUES
        # ('你的姓名是', '我是小明')
        # """
        # cursor.execute(sql)
        '''
        cur.execute("""
        CREATE TABLE IF NOT EXISTS answer
        (que_id text, answer text);
        """)
        
        cur.execute("""
                CREATE TABLE IF NOT EXISTS question
                (questions text, vectors text, answoer_ids int);
                """)
        '''
        # # 3.3 查询语句
        # "SELECT vectors,answer FROM question join answer ON question.answer_id=answer.que_id"
        # sql = "SELECT questions,answer FROM question as a join answer as b ON a.answoer_ids=b.que_id;"
        # ask,answer = cursor.execute(sql).fetchone()
        # print(ask,"\n",answer)

        # data = cursor.execute(sql)
        # words = set()
        # for ask,answer in tqdm(data,total=number):
        #     for word in list(ask):
        #         words.add(word)
        #     for word in list(answer):
        #         words.add(word)
        # words = list(words)
        # words = sorted(words)   # 排序
        # print("一共找到{}个词".format(len(words)))

        # 3.3 查询语句
        # batch_size = 10
        # start_id = np.random.randint(1, number - batch_size + 1)
        # sql = "SELECT questions,vectors FROM question WHERE ROWID >= {} AND ROWID < {}".format(
        #     start_id, start_id + batch_size)
        sql = "SELECT vectors,answers FROM question join answer ON question.ans_ids=answer.que_ids"
        value = cursor.execute(sql).fetchall()
        print(value)



    finally:
        cursor.close()

