# add all stocks of funds together
# 便利所有基金，获取基金持仓股数相加。
import akshare as ak
import pandas as pd
import multiprocessing

#东方财富网-沪深京 A 股-实时行情数据
stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()
'''
       序号 代码    名称   最新价   涨跌幅 涨跌额    成交量   成交额        振幅   最高    最低   今开    昨收    量比  换手率  市盈率-动态   市净率         总市值        流通市值 涨速  5分钟涨跌  60日涨跌幅  
年初至今涨跌幅
0        1  920002  N万达  72.01  247.20  51.27   39305.0  2.518762e+08  88.43  75.00  56.66  57.00  20.74   NaN  82.75   40.09  3.66  2.286470e+09  3.420475e+08 -0.68   0.56  247.20   247.20
'''
#all stocks
stock_zh_a_spot_df = pd.DataFrame(stock_zh_a_spot_em_df, columns=['代码', '名称','最新价'])
stock_zh_a_spot_df['股数'] = 0.0
#print(stock_zh_a_spot_df)
'''
      代码     名称      最新价  股数
0     301208   中亦科技  30.94  0.0  
[5614 rows x 4 columns]'''
#print(stock_zh_a_spot_em_dfdf)
#get all funds
fund_name_em_df = ak.fund_name_em()
fund_sum = len(fund_name_em_df)
print('fund sum =',fund_sum)
'''    基金代码          拼音缩写            基金简称      基金类型                         拼音全称
0      000001            HXCZHH            华夏成长混合   混合型-灵活                      HUAXIACHENGZHANGHUNHE'''

#print(type(fund_name_em_df[['基金代码']])) #<class 'pandas.core.frame.DataFrame'>
'''
0        000001
21633    970214
'''
fundindex = 0
for fund in fund_name_em_df['基金代码']:
  #print(fund,type(fund)) #<class 'str'>
  #print(fund_name_em_df['基金代码'] == fund)
  
  '''
  fund:  000621 426    易方达现金增利货币B
  '''
  try:
    #get stocks hold by fund
    fund_portfolio_hold_em_df = ak.fund_portfolio_hold_em(symbol=fund, date="2024") #only years
  except:
    print('fund: ',fund,fund_name_em_df.loc[fund_name_em_df['基金代码'] == fund,'基金简称'].values,'get stocks NULL\n')
    continue
  print('fund: ',fund,fund_name_em_df.loc[fund_name_em_df['基金代码'] == fund,'基金简称'].values,'process = %.2f\n',fund_name_em_df.loc[fund_name_em_df['基金代码'] == fund].index/fund_sum)

  '''
    序号    股票代码   股票名称  占净值比例 持股数   持仓市值              季度
  0   1     002025    航天电器  3.46      209.92  7947.67  2024年1季度股票投资明细
  '''
  #print(fund_portfolio_hold_em_df)

  for stock in fund_portfolio_hold_em_df['股票代码']:
    '''
    002025 703    0
    '''
    print(stock)
    if (stock in stock_zh_a_spot_df['代码']):
      #only support A stock
      stock_zh_a_spot_df.loc[stock_zh_a_spot_df['代码'] == stock,'股数'] = \
        stock_zh_a_spot_df.loc[stock_zh_a_spot_df['代码'] == stock,'股数'].iloc[0] + \
        fund_portfolio_hold_em_df.loc[fund_portfolio_hold_em_df['股票代码'] == stock,'持股数'].iloc[0]

    #print(stock_zh_a_spot_df.loc[stock_zh_a_spot_df.代码 == stock,'股数'].iloc[0],'\n')

stock_zh_a_spot_df.to_csv('fundstocks.cvs')
print('end\n')