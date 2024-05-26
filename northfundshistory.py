#北向资金与各个指数的历史数值显示

import akshare as ak
import pandas as pd
import datetime

import matplotlib.pyplot as plt
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"]=False

#常用大盘指数 '上证综指': 'sh000001',
indexs={'深证成指': 'sz399001',
        '沪深300': 'sh000300',
       '创业板指': 'sz399006',
       '上证50': 'sh000016',
       '中证500': 'sh000905',
       '中小板指': 'sz399005',
       '上证180': 'sh000010'}
#start='2014-11-17'
start='2021-11-17'
end = str(datetime.date.today())
# get_index_data from #https://blog.csdn.net/qq_26742269/article/details/122834149 
def get_index_data(code,start,end):
    start = datetime.datetime.strptime(start,"%Y-%m-%d")
    end =datetime.datetime.strptime(end,"%Y-%m-%d")
    #print("get_index_data ",code,start,end)
    index_df = ak.stock_zh_index_daily(symbol=code)
    index_df['date']=index_df['date'].apply(lambda x:datetime.datetime.strptime(str(x),"%Y-%m-%d"))
    index_df = index_df.loc[(index_df['date']>=start) & (index_df['date']<=end)]
    index_df.reset_index(drop=True,inplace=True)
    return index_df

index_data=pd.DataFrame()
#get first data include date and close
index_data = get_index_data('sh000001',start,end)
index_data = index_data[['date','close']]
index_data.rename(columns={'close': '上证综指'},inplace=True)
#add follow index's close to index_data
for name,code in indexs.items():
    #index_data[name]=get_index_data(code,start,end)['close']
    insert = get_index_data(code,start,end)['close']
    index_data.insert(loc=len(index_data.columns),column = name,value=insert)
    #print('add ',name)
#print(index_data)

#沪深港通历史数据
stock_hsgt_hist_em_df = ak.stock_hsgt_hist_em(symbol="北向资金")
stock_hsgt_hist_em_df['日期']=stock_hsgt_hist_em_df['日期'].apply(lambda x: datetime.datetime.strptime(str(x) ,'%Y-%m-%d')) 
stock_hsgt_hist_em_df = stock_hsgt_hist_em_df.loc[(stock_hsgt_hist_em_df['日期'] >= start) & (stock_hsgt_hist_em_df['日期']<= end)]
stock_hsgt_hist_em_df.rename(columns={'日期': 'date'},inplace=True)
#print(stock_hsgt_hist_em_df)
'''
      日期       当日成交净买额  买入成交额  卖出成交额  历史累计净买额  当日资金流入       当日余额          持股市值   领涨股  领涨股-涨跌幅    沪深300  沪深300-涨跌幅     领涨股-代码
0     2014-11-17  120.8233     120.8233    0.0000    0.012082      130.0000     0.0000  0.000000e+00   唐山港     9.98  2474.01      -0.19  601000.SH
'''


stock_hsgt_hist_em_df = pd.merge(stock_hsgt_hist_em_df,index_data,how='inner',on='date') #here change 沪深300 to 沪深300_y ?
#print(stock_hsgt_hist_em_df)
stock_hsgt_hist_em_df = stock_hsgt_hist_em_df[['date','当日成交净买额','上证综指','深证成指','沪深300_y','创业板指','上证50','中证500','中小板指','上证180']]
#print(stock_hsgt_hist_em_df)
stock_hsgt_hist_em_df.plot(x='date',secondary_y =['上证综指','深证成指','沪深300_y','创业板指','上证50','中证500','中小板指','上证180'] )

plt.show()

df = stock_hsgt_hist_em_df[['当日成交净买额','上证综指']]
# 计算 Pearson 相关系数
correlation_matrix = df.corr()
print(correlation_matrix)