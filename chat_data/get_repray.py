#-*- encoding: utf-8 -*-
from twitter import *
import os,sys,time
#http://blog.maqua.tech/entry/2014/10/15/172420

#NG words
#ここでタプルに追加した単語（記号）を含むツイートは収集から除外される。
#今回は、URLやHashtag、【定期】のように使われるカッコなどを除外とした。
check_chara = ('http', '#', '\\', '【','】')

#Retry MAX
retry_max = 10
#Retry time
retry_time = 10

#ここにはTwitter Devで登録したアプリケーションのCONSUMER_KEY,SECRETを記入する。
CONSUMER_NAME = 'YOUR APPLICATION NAME'
CONSUMER_KEY =  'hogehoge'
CONSUMER_SECRET = 'hogehoge'

#過去に認証したときのTokenを探す。
TWITTER_CREDS = os.path.expanduser('.credentials')
if not os.path.exists(TWITTER_CREDS):
  #無ければ、OAuthで認証
  oauth_dance(CONSUMER_NAME, CONSUMER_KEY, CONSUMER_SECRET, TWITTER_CREDS)
oauth_token, oauth_secret = read_token_file(TWITTER_CREDS)

# token
# Tokenを用いて、Streaming APIで新着ツイートを見つけるstreamと、普通のTwitter APIで'in_reply_to_status_id'を参照するtwitterの2つを用意する。
stream = TwitterStream(auth=OAuth(oauth_token, oauth_secret, CONSUMER_KEY, CONSUMER_SECRET))
twitter = Twitter(auth=OAuth(oauth_token, oauth_secret, CONSUMER_KEY, CONSUMER_SECRET))

#in_reply_to_status_idから宛先となるツイートを参照する。
def show(_status):
  #print('IN:\n'+str(_status), file=sys.stderr)
  #リトライを設定
  for r in range(retry_max):
    #言語が日本語、in_reply_to_status_idが明示されており、禁止単語を含まないツイートの場合
    if 'lang' in _status and _status['lang'] == 'ja' and 'in_reply_to_status_id' in _status and not _status['in_reply_to_status_id'] is None and 'text' in _status and check(_status['text']):
      try:
        #ツイートを参照
        status=twitter.statuses.show(id=_status['in_reply_to_status_id'])
      except:
        #print('! Error\tRetry to read in_reply_tweet - '+str(r), file=sys.stderr)
        time.sleep(retry_time)
        continue
      #print('OUT:\n'+str(status), file=sys.stderr)
      #宛先のツイートが日本語、in_reply_to_status_idが明示されており、禁止単語を含まないツイートの場合
      if 'lang' in _status and _status['lang'] == 'ja' and 'text' in status and check(status['text']):
        #標準出力で応答の対を出力
        print(str(status['id'])+'\t'+trim(status['text'])+'\t'+str(_status['id'])+'\t'+trim(_status['text']))
        #さらに、宛先のツイートがin_reply_to_status_idを持つ場合は、さらに参照を行う。
        if 'in_reply_to_status_id' in status and not status['in_reply_to_status_id'] is None:
          show(status)
        break
      else:
        #print('! Info\tin_reply_tweet is not acceptable', file=sys.stderr)
        break
    else:
      break

#改行を適当に置き換える
def trim(text):
  return text.replace('\r','-br-').replace('\n','-br-')

#禁止文字のチェックを行う
def check(text):
  for char in check_chara:
    if char in text:
      return False
  return True

#Streaming APIからツイートを読み出す
def main():
  while 1:
    #ストリームに接続して適当にツイートを読み出す
    statuses = stream.statuses.sample()
    for status in statuses:
      #参照に送る
      show(status)

if __name__ == '__main__':
 main()


