# -*- coding: utf_8 -*-
import json
import urllib
from urllib.parse import quote
import urllib.request as ur
import doctest
import requests
import xmltodict
import sqlite3
from collections import OrderedDict

# from xmlutils.xml2json import xml2json
# from xml2json import json2xml
# import xml2json

def sample(query):

	q = {'appid':  "hogehoge",'query': quote(query)}
	r = requests.get('http://chiebukuro.yahooapis.jp/Chiebukuro/V1/questionSearch', params=q)

	result = xmltodict.parse(r.text)

	print(result["ResultSet"].keys())#キーを取得
	print(result["ResultSet"]["Result"]["Question"][1])
	print(result["ResultSet"]["Result"]["Question"][2])

if __name__ == "__main__":

	sample("天気")



