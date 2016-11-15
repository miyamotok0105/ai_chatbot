# coding: utf_8
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import csv
#

#いろいろ大事な情報クラス。ツイッターAPIに接続するための情報
class twitterApi():
    def apiData():
        consumer_key = 'hogehoge'
        consumer_secret = 'hogehoge'
        access_key = 'hogehoge'
        access_secret = 'hogehoge'
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)                                           
        auth.set_access_token(access_key, access_secret)                                                                                           
        api = tweepy.API(auth_handler=auth)
        return api

#csvに書き込むクラス
class csvWrite():
    def strWrite(strText):
        listStr = [strText]
        fileOpen = open("/home/mluser/Desktop/Sample/testWrite.csv","w")
        writer=csv.writer(fileOpen,lineterminator='\n')
        writer.writerow(listStr)
        fileOpen.close()
        print("書き込み成功")

if __name__ == '__main__':
    try:
        #APIの呼び出し
        getApi = twitterApi
        api=getApi.apiData()
        
        #CSVクラスの呼び出し
        writeCSV = csvWrite
        
        #検索したいキーワードの設定
        keyword =['新宿','ランチ']
        #キーワードの格納
        query = ' AND '.join(keyword)
        #キーワードによる検索結果の取得
        search_result = api.search(q=query)
        #キーワード取得の確認
        print(search_result[1].text)
        #ツイートで取得したつぶやきをCSVに書き込む
        # writeCSV.strWrite(search_result[1].text)

    except tweepy.TweepError as e:
        print(e.reason)



