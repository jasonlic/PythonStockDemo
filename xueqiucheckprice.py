#use pysnowball to check xueqiu's stock price    
# login xueqiu web
# pip install browser_cookie3 pysnowball
import pysnowball as ball
import requests
import json
import os, random, time, datetime, schedule, webbrowser
import browser_cookie3

#https://www.jisilu.cn/question/458945
#use default brower: chrome
#https://github.com/wbbyfd/UniversalRotation?tab=readme-ov-file
def get_xq_a_token():
    str_xq_a_token = ';'
    while True:
        cj = browser_cookie3.load()
        for item in cj:
            if item.name == "xq_a_token" :
                print('get token, %s = %s' % (item.name, item.value))
                str_xq_a_token = 'xq_a_token=' + item.value + ';'
                return str_xq_a_token
        if str_xq_a_token == ";" :
            print('get token, retrying ......')
            webbrowser.open("https://xueqiu.com/")
            time.sleep(60)


myxiuqiutaken = get_xq_a_token()

ball.set_token(myxiuqiutaken)

data = ball.watch_stock(-1) #class 'dict'
#print(type(data))
all = data['data'] #class 'dict'

stocks = all['stocks'] #class 'list'

num = len(stocks)

def get_requests(url):

    HEADERS = {'Host':  "stock.xueqiu.com",
               'Accept': 'application/json',
               'Cookie': myxiuqiutaken,
               'User-Agent': 'Xueqiu iPhone 11.8',
               'Accept-Language': 'zh-Hans-CN;q=1, ja-JP;q=0.9',
               'Accept-Encoding': 'br, gzip, deflate',
               'Connection': 'keep-alive'}
    response = requests.get(url, headers=HEADERS) #must has headers
    return response #<class 'requests.models.Response'>
#test get_requests
#response = get_requests("https://stock.xueqiu.com/v5/stock/alert/config/get.json?symbol=SZ000737")
for stock in stocks:
    weburl = "https://stock.xueqiu.com/v5/stock/alert/config/get.json?symbol="
    weburl = weburl + stock['symbol']
    response = get_requests(weburl)
    stock_str = response.content.decode() #str
    j_dict = json.loads(stock_str) #<class 'dict'>
    j_data = j_dict['data']

    stock_jason = ball.quotec(stock['symbol'])['data'] # <class 'list'>

    stock_price = stock_jason[0]['current']

    if (j_data['price_asc'] != None):
        if (stock_price > j_data['price_asc']):
            print(stock['name'],stock['symbol'],' is too low')

    if (j_data['price_desc'] != None):
        if (stock_price < j_data['price_desc']):
            print(stock['name'],stock['symbol'],' is too high')



