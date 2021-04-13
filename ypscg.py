# https://blog.51cto.com/youerning/2083764?wx=
# -*- coding: utf-8 -*
import pandas as pd
import tushare as ts

#通过股票代码获取股票数据,这里没有指定开始及结束日期 
df = ts.get_k_data("300104")

# 将数据的index转换成date字段对应的日期
df.index = pd.to_datetime(df.date)

# 将多余的date字段删除
df.drop("date", inplace=True, axis=1)

# 计算5,15,50日的移动平均线, MA5, MA15, MA50
days = [5, 15, 50]
for ma in days:
    column_name = "MA{}".format(ma)
    df[column_name] = df.close.rolling(ma).mean()
    
# 计算浮动比例
df["pchange"] = df.close.pct_change()
# 计算浮动点数
df["change"] = df.close.diff()

print(df.head())

df[["close", "MA5", "MA15", "MA50"]].plot(figsize  = (10,18))    
'''
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
#from matplotlib.finance import date2num, candlestick_ohlc
from mpl_finance import date2num, candlestick_ohlc

def candlePlot(data, title=""):
    data["date"] = [date2num(pd.to_datetime(x)) for x in data.index]
    dataList = [tuple(x) for x in data[
        ["date", "open", "high", "low", "close"]].values]

    ax = plt.subplot()
    ax.set_title(title)
    ax.xaxis.set_major_formatter(DateFormatter("%y-%m-%d"))
    candlestick_ohlc(ax, dataList, width=0.7, colorup="r", colordown="g")
    plt.setp(plt.gca().get_xticklabels(), rotation=50,
             horizontalalignment="center")
    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    plt.grid(True)

candlePlot(df)
'''