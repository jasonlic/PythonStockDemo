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
stock_zh_a_spot_em_dfdf = pd.DataFrame(stock_zh_a_spot_em_df, columns=['代码', '名称','最新价'])
stock_zh_a_spot_em_dfdf['股数'] = 0
'''
     代码    名称    最新价  股数
0     688655   迅捷兴  10.36   0
1     688653  康希通信  13.06   0
2     688603  天承科技  62.44   0
3     688720  艾森股份  42.17   0
4     688671  碧兴物联  22.20   0
...      ...   ...    ...  ..
5609  603233   大参林  17.46   0
5610  002840  华统股份  18.93   0
5611  002496  辉丰股份   2.92   0
5612  600766  退市园城   0.36   0
5613  000982  中银绒业   0.32   0

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
1        000002
2        000003
3        000004
4        000005
          ...
21629    970210
21630    970211
21631    970212
21632    970213
21633    970214
'''
for fund in fund_name_em_df['基金代码']:
  #print(fund,type(fund)) #<class 'str'>
  #print(fund_name_em_df['基金代码'] == fund)
  print('fund: ',fund,fund_name_em_df.loc[fund_name_em_df['基金代码'] == fund,'基金简称'],'\n')
  '''
  fund:  000621 426    易方达现金增利货币B
  Name: 基金简称, dtype: object'''
  try:
    fund_portfolio_hold_em_df = ak.fund_portfolio_hold_em(symbol=fund, date="2024") #only years
  except:
    print('fund: ',fund,fund_name_em_df.loc[fund_name_em_df['基金代码'] == fund,'基金简称'],'get stocks NULL\n')
    continue

  '''
    序号    股票代码   股票名称  占净值比例 持股数   持仓市值              季度
  0   1     002025    航天电器  3.46      209.92  7947.67  2024年1季度股票投资明细
  '''


  #print(fund_portfolio_hold_em_df)

  for stock in fund_portfolio_hold_em_df['股票代码']:
    #print(stock,type(stock)) #002025 <class 'str'>
    #print(stock,stock_zh_a_spot_em_dfdf.loc[stock_zh_a_spot_em_dfdf['代码'] == stock,'股数']) #002025 703    0
    '''
    002025 703    0
    Name: 股数, dtype: int64  '''
   # print(fund_portfolio_hold_em_df.loc[fund_portfolio_hold_em_df['股票代码'] == stock,'持股数'])

    stock_zh_a_spot_em_dfdf.loc[stock_zh_a_spot_em_dfdf['代码'] == stock,'股数'] = \
            stock_zh_a_spot_em_dfdf.loc[stock_zh_a_spot_em_dfdf['代码'] == stock,'股数'] + \
            fund_portfolio_hold_em_df.loc[fund_portfolio_hold_em_df['股票代码'] == stock,'持股数']
    #print(stock_zh_a_spot_em_dfdf.loc[fund_portfolio_hold_em_df['代码'] == stock,'股数'])

print(stock_zh_a_spot_em_dfdf)