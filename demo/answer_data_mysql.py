import pymysql
import pandas as pd

conn = pymysql.connect(host="localhost", user="root", password="root",
                                    database="qa_bot", port=3306, charset="utf8")

insert_sql = "INSERT INTO answer(que_id, answer) VALUES(%s, %s)"

data_file = "../que_ans_data/answer.csv"
df = pd.read_csv(data_file, sep=',', header=None, names=["ans_id","question_id","content"])
with conn.cursor() as cursor:
    for _,question_id, answers in df.values[1:]:
        # print(question, vector,(question, ','.join(map(str, vector))))
        cursor.execute(
            insert_sql, (question_id, answers)
        )
    conn.commit()


insert_sql = "INSERT INTO question (answoer_ids, vectors,questions) VALUES(%s ,%s, %s)"
data_file = "../que_ans_data/question.csv"
df = pd.read_csv(data_file, sep=',', header=None, names=["question_id","content"])
with conn.cursor() as cursor:
    for _,question_id, answers in df.values[1:]:
        # print(question, vector,(question, ','.join(map(str, vector))))
        cursor.execute(
            insert_sql, (question_id, answers)
        )
    conn.commit()
