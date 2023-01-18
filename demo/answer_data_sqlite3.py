#!/usr/bin/env python3

import os
import re
import sys
import sqlite3

from tqdm import tqdm

# 返回处理的数据，大的list，
def file_lines(file_path):
    with open(file_path, 'rb') as fp:# 以二进制的方式读取原始文件
        b = fp.read()
    content = b.decode('utf8', 'ignore')# 然后将二进制转换为utf-8的格式
    lines = []
    for line in tqdm(content.split('\n')):# 读取数据，以换行符分割

        line = line.replace('\n', '').strip()# 读取一行数据，去除换行符
        lines.append(line)

    return lines


def contain_chinese(s):
    # 是否包含中文
    if re.findall('[\u4e00-\u9fa5]+', s):
        return True
    return False


# 验证
def valid(a, max_len=0):
    if len(a) > 0 and contain_chinese(a):
        if max_len <= 0:
            return True
        elif len(a) <= max_len:  # 防止验证太长
            return True
    return False


def insert(answer,que_ids,cur):
    cur.execute("""
    INSERT INTO answer (answers,que_ids) VALUES
    ('{}','{}')
    """.format(answer.replace("'", ""),que_ids))


def insert_if(answer,que_ids ,cur):
    if valid(answer):
        insert(answer,que_ids, cur)
        return 1
    return 0


def main(file_path):
    # 读取所有数据，将所有数据添加到内存中
    lines = file_lines(file_path)

    print('一共读取 %d 行数据' % len(lines))

    db = '../config/qa_bot.db'
    if not os.path.exists(os.path.dirname(db)):
        os.makedirs(os.path.dirname(db))

    conn = sqlite3.connect(db)
    # 创建游标
    cur = conn.cursor()
    # 如果这个表不存在则创建这个表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS answer
        (answers text,que_ids int);
        """)
    # 提交事务
    conn.commit()


    inserted = 0
    for index,line in tqdm(enumerate(lines), total=len(lines)):
        # 将问答对的数据存到aqlite3中
        if line=='':
            break
        _,que_id, answer = line.split(",")

        inserted += insert_if(answer,que_id,cur)

    conn.commit()

    print("总样本数目为:{}".format(inserted))


if __name__ == '__main__':
    # 原始的文本数据
    file_path = '../que_ans_data/answer.csv'
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    # 判断文件是否存在
    if not os.path.exists(file_path):
        print('文件 {} 不存在'.format(file_path))
    else:
        main(file_path)
