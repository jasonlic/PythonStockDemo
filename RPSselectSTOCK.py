
#【Python量化】如何利用欧奈尔的RPS寻找强势股？ https://zhuanlan.zhihu.com/p/59867869
#先引入后面可能用到的library
import pandas as pd  
import tushare as ts 
import matplotlib.pyplot as plt
#%matplotlib inline   

#正常显示画图时出现的中文和负号
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

#使用之前先输入token，可以从个人主页上复制出来，
#每次调用数据需要先运行该命令
token='输入你在tushare获得的token'

ts.set_token(token)
pro=ts.pro_api()

df = pro.stock_basic(exchange='', list_status='L', 
    fields='ts_code,symbol,name,area,industry,list_date')
print(len(df))

#排除掉新股次新股，这里是只考虑2017年1月1日以前上市的股票
df=df[df['list_date'].apply(int).values<20170101]
len(df)
#输出结果：3024
#获取当前所有非新股次新股代码和名称
codes=df.ts_code.values
names=df.name.values
#构建一个字典方便调用
code_name=dict(zip(names,codes))

#使用tushare获取上述股票周价格数据并转换为周收益率
#设定默认起始日期为2018年1月5日，结束日期为2019年3月19日
#日期可以根据需要自己改动
def get_data(code,start='20180101', end='20190319'):
    df=pro.daily(ts_code=code, start_date=start, end_date=end,fields='trade_date,close')
    #将交易日期设置为索引值
    df.index=pd.to_datetime(df.trade_date)
    df=df.sort_index()
    #计算收益率
    return df.close

#构建一个空的dataframe用来装数据
data=pd.DataFrame()
for name,code in code_name.items():
    data[name]=get_data(code)

#data.to_csv('daily_data.csv',encoding='gbk')
#data=pd.read_csv('stock_data.csv',encoding='gbk',index_col='trade_date')
#data.index=(pd.to_datetime(data.index)).strftime('%Y%m%d')

#计算收益率
def cal_ret(df,w=5):
    '''w:周5;月20;半年：120; 一年250
    '''
    df=df/df.shift(w)-1
    return df.iloc[w:,:].fillna(0)    