import pandas as pd

ques_file = "../que_ans_data/question.csv"
answ_file = "../que_ans_data/answer.csv"

ques = pd.read_csv(ques_file, sep=',', header=None, names=["question_id","content"])
answ = pd.read_csv(answ_file, sep=',', header=None, names=["ans_id","question_id","content"])
o = 0
with open('../que_ans_data/q_a_data.csv', 'w', encoding='utf-8') as writer:
    for question_id,q_text in ques.values[1:]:
        for ans_id,q_id,a_text in answ.values[1:]:
            if ans_id == 29842:
                print(question_id,q_id)
            if question_id==q_id:
                writer.writelines(",".join([q_text,a_text,question_id])+"\n")
                o+=1
                print(o)

