# -- encoding:utf-8 --

import base64
import requests
import json
import time




# url = "http://127.0.0.1:5000/similarity"
# data = {
#     "text1": "如何绑定花呗的支付方式",
#     "text2": "花呗的支付方式怎么绑定"
# }

'''
sqlite3_bot
sqlite3_bot_KDTree
chat_bot
chat_bot_KDTree

'''
url = 'http://127.0.0.1:5000/sqlite3_bot'
while True:
    question = input("请输入问题：")
    time_start = time.time()
    data = {
        "question": "{}".format(question),
        "threshold": 0.999
    }
    requests.head(url,data=data)
    result = requests.post(url=url, data=data)
    if result.status_code == 200:
        print(result.text)
    time_end=time.time()
    print('totally cost',time_end-time_start)

