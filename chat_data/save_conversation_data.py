# -*- coding: utf-8 -*-
"""
会話データをDBに保存するファイル
"""
import unittest
import re
import MeCab
import sqlite3
from collections import defaultdict
import types
import io

class SaveConversation(object):
    u"""
    DBに保存するクラス
    """

    DB_PATH = "conversation.db"
    DB_SCHEMA_PATH = "conversation_schema.sql"
    
    def __init__(self):
        u"""
        初期化メソッド
        @param text チェーンを生成するための文章
        """
        # 形態素解析用タガー
        self.tagger = MeCab.Tagger('-Ochasen')

    def save(data_list, init=False):
        u"""
        会話をDBに保存
        @param 
        """
        # DBオープン
        con = sqlite3.connect(SaveConversation.DB_PATH)

        # 初期化から始める場合
        if init:
            # DBの初期化
            with open(SaveConversation.DB_SCHEMA_PATH, "r") as f:
                schema = f.read()
                con.executescript(schema)

        # データ挿入
        p_statement = "insert into conversation_pair (user1, user1_talk, user2, user2_talk) values (?, ?, ?, ?)"
        con.executemany(p_statement, data_list)

        # コミットしてクローズ
        con.commit()
        con.close()

if __name__ == '__main__':
    dataset=[]

    

    for line in open('temp.csv', 'r', encoding="utf-8"):
        try:
            line = re.sub(r'([0-9])+', '', line)
            line = line.replace('-br-', '')
            line = line.replace(',', '')
            line = line.replace('	', ',')
            
            result = line
            # print("result",result)

            result_splited = result.split(",,", 1)
            # print("1:",result_splited[0])
            # print("2:",result_splited[1])

            m = re.search('(,@[\w]+)', result_splited[0])
            # print("3",m.group(1))
            user1 = str(m.group(1)).replace(',','')
            user1_talk = str(result_splited[0]).replace(str(m.group(1)),'')

            m = re.search('(@[\w]+)', result_splited[1])
            # print("4",m.group(1)) 
            user2 = str(m.group(1)).replace(',','')
            user2_talk = str(result_splited[1]).replace(str(m.group(1)),'')
            print(str(user1)+ ":" +str(user1_talk))
            print(str(user2)+ ":" +str(user2_talk))
            dataset.append((user1, user1_talk, user2, user2_talk))
            print("----------")
        except:
            pass

    # SaveConversation.save(dataset, True)
    SaveConversation.save(dataset)




