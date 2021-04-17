#https://mp.weixin.qq.com/s/3QWoNBOLh-cRyB1YPVtCZA
#使用Python构建股票财务指标打分系统
import pandas as pd
import numpy as np
from datetime import datetime
#画图
import matplotlib.pyplot as plt
#%matplotlib inline
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
#获取数据
import tushare as ts 
token='输入你在tushare上获取的token'
ts.set_token(token)
pro=ts.pro_api(token)

#编写函数获取相应的财务指标，输出dataframe格式。
def get_indicators(code):
    #获取当前时间，计算当前和过去四年时间
    t0=datetime.now()
    t1=datetime(t0.year-4,t0.month,t0.day)
    end=t0.strftime('%Y%m%d')
    start=t1.strftime('%Y%m%d')
    #财务比率
    fields='ann_date,end_date,tr_yoy,op_yoy,\
         grossprofit_margin,expense_of_sales,inv_turn,eps,\
         ocfps,roe_yearly,roa2_yearly,netprofit_yoy'
    fina = (pro.fina_indicator(ts_code=code,start_date=start, end_date=end,fields=fields)
           .drop_duplicates(subset=['ann_date','end_date'],keep='first'))
    fina.set_index('end_date',inplace = True)
    fina=fina.sort_index()
    #获取市盈率和市净率指标（pe、pb数据）
    pbe=pro.daily_basic(ts_code=code, fields='trade_date,pe_ttm,pb')
    pbe.set_index('trade_date',inplace=True)
    pbe=pbe.sort_index()
    #合并数据
    df=pd.merge(fina,pbe,left_index=True,right_index=True,how='left')
    #pb缺失数据使用前值填充，pe不管，缺失值可能是因为盈利为负数
    df['pb'].fillna(method='ffill',inplace=True)
    return df
#编写各项财务指标的评分函数。    
#存在缺失值或者负值（市盈率）情况下评分直接为0
#营业收入增长率打分（0-10）
def cal_tryoy(y):
    '''y是营业收入增长率'''
    try:
        return 5+ min(round(y-10),5) if y>=10 else 5+ max(round(y-10),-5)
    except:
        return 0
#营业利润增长率打分(0-10)
def cal_opyoy(y):
    '''y是营业利润增长率'''
    try:
        return 5+ min(round((y-20)/2),5) if y>=20 else 5+ max(round((y-20)/2),-5)
    except:
        return 0
#毛利率打分
def cal_gpm(y):
    '''y是最近季度毛利率-前三季度平均毛利率'''
    try:
        return 5+min(round(y)/0.5,5) if y>0 else max(round(y)/0.5,-5)+5
    except:
        return 0
#期间费用率打分
def cal_exp(y):
    '''y是最近季度期间费用率-前三季度平均期间费用率'''
    try:
        return 5+min(round(y)/0.5,5) if y>0 else max(round(y)/0.5,-5)+5
    except:
        return 0
#存货周转率打分
def cal_inv(y):
    '''y是（最近季度存货周转率-前三季度平均存货周转率）/前三季度平均存货周转率*100'''
    try:
        return 5+min(round(y/2),5) if y>0 else max(round(y/2),-5)+5
    except:
        return 0
#每股经营性现金流打分
def cal_ocfp(y):
    '''y是（最近三季度每股经营性现金流之和-最近三季度每股收益之和）/最近三季度每股收益之和*100'''
    try:
        return 5+min(round(y/4),5) if y>0 else max(round(y/4),-5)+5
    except:
        return 0
#净资产收益率打分
def cal_roe(y):
    '''y是年化净资产收益率'''
    try:
        return 5+ min(round(y-15),5) if y>=15 else 5+ max(round(y-15),-5)
    except:
        return 0
#总资产报酬率打分
def cal_roa(y):
    '''y是最近季度年化总资产报酬率'''
    try:
        return min(round((y-5)/0.5),10) if y>=5 else max(round(y-5),0)
    except:
        return 0
#市净率打分
def cal_pb(y):
    '''y是市净率'''
    try:
        return 5-max(round((y-3)/0.4),-5) if y<=3 else 5-min(round((y-3)/0.4),5)
    except:
        return 0
#动态市盈率相对盈利增长率（PEG）打分
def cal_pe(y):
    '''y是动态市盈率相对盈利增长率'''
    try:
        return 5-max(round((y-1)/0.1),-5) if y<=1 else 5-min(round((y-1)/0.1),5)
    except:
        return 0


#计算财务指标得分
def indicator_score(code):
    data=get_indicators(code)
    '''(1)营业收入增长率打分'''
    data['营收得分']=data['tr_yoy'].apply(cal_tryoy)
    '''(2)营业利润增长率打分'''
    data['利润得分']=data['op_yoy'].apply(cal_opyoy)
    '''(3)毛利率打分'''
    #计算最近季度毛利率-前三季度平均毛利率
    data['gpm']=data['grossprofit_margin']-data['grossprofit_margin'].rolling(3).mean()
    data['毛利得分']=data['gpm'].apply(cal_gpm)
    '''(4)期间费用率打分'''
    #最近季度期间费用率-前三季度平均期间费用率
    data['exp']=data['expense_of_sales']-data['expense_of_sales'].rolling(3).mean()
    data['费用得分']=data['exp'].apply(cal_exp)
    '''(5)周转率打分'''
    #（最近季度存货周转率-前三季度平均存货周转率）/前三季度平均存货周转率*100
    data['inv']=(data['inv_turn']-data['inv_turn'].rolling(3).mean())*100/data['inv_turn'].rolling(3).mean()
    data['周转得分']=data['inv'].apply(cal_inv)
    '''(6)每股经营现金流打分'''
    #（最近三季度每股经营性现金流之和-最近三季度每股收益之和）/最近三季度每股收益之和*100
    data['ocf']=(data['ocfps'].rolling(3).sum()-data['eps'].rolling(3).sum())*100/data['eps'].rolling(3).sum()
    data['现金得分']=data['ocf'].apply(cal_ocfp)
    '''(7)净资产收益率打分'''
    data['净资产得分']=data['roe_yearly'].apply(cal_roe)
    '''(8)总资产收益率打分'''
    data['总资产得分']=data['roa2_yearly'].apply(cal_roa)
    '''(9)市净率打分'''
    data['市净率得分']=data['pb'].apply(cal_pb)
    '''(10)动态市盈率相对盈利增长率打分'''
    #动态市盈率相对盈利增长率
    data['peg']=data['pe_ttm']/data['netprofit_yoy'].rolling(3).mean()
    data['市盈率得分']=data['peg'].apply(cal_pe)
    #计算总得分
    data['总分']=data[['营收得分','利润得分','费用得分','周转得分','现金得分','净资产得分','总资产得分',\
                 '市净率得分','市盈率得分']].sum(axis=1)
    return data[['营收得分','利润得分','费用得分','周转得分','现金得分','净资产得分','总资产得分',\
                 '市净率得分','市盈率得分','总分']]

#贵州茅台
code='600519.SH'
indicator_score(code)
plot_signal(code)
plot_stock_signal(code)

#获取当前正常上市交易的股票列表
def get_code():
    t=datetime.now()
    df=pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    #排除上市日期短于4年的个股
    #获取当前年份
    year=datetime.now().strftime('%Y')
    #四年前
    year=str(int(year)-4)+'0101'
    #保留上市时间大于四年个股数据
    df=df[df.list_date<year]
    #排除银行、保险、多元金融公司
    df=df[-df.industry.isin(['银行','保险','多元金融'])]
    #排除st和*ST股
    df=df[-df.name.str.startswith(('ST'))]
    df=df[-df.name.str.startswith(('*'))]
    code=df.ts_code.values
    name=df.name
    return dict(zip(name,code))

#计算所有股票财务指标总分
def get_all_score():
    df=pd.DataFrame()
    for name,code in get_code().items():
        try:
            df[name]=indicator_score(code)['总分']
        except:
            continue
    return df

dff=get_all_score()    
#获取最近日期总分排名前15个股
dff.T.sort_values(dff.T.columns[-1],ascending=False).iloc[:15,-10:]