#https://mp.weixin.qq.com/s/VigghdPnvY-8DLlsB97PWA
#【Python量化】使用机器学习预测股票交易信号
import numpy as np
import pandas as pd  
import tushare as ts
#技术指标
import talib as ta
#机器学习模块
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier,XGBRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
import shap
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn import preprocessing
#画图
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

#%matplotlib inline   
#正常显示画图时出现的中文和负号
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

#默认以上证指数交易数据为例
def get_data(code='sh',start='2000-01-01',end='2021-03-02'):
    df=ts.get_k_data('sh',start='2005')
    df.index=pd.to_datetime(df.date)
    df=df[['open','high','low','close','volume']]
    return df
df=get_data()
df_train,df_test=df.loc[:'2017'],df.loc['2018':]

def trade_signal(data,short=10,long=60,tr_id=False):
    data['SMA1'] = data.close.rolling(short).mean()
    data['SMA2'] = data.close.rolling(long).mean() 
    data['signal'] = np.where(data['SMA1'] >data['SMA2'], 1.0, 0.0) 
    if(tr_id is not True):
        display(data['signal'].value_counts())

df_tr1 = df_train.copy(deep=True)  
df_te1 = df_test.copy(deep=True) 
trade_signal(df_tr1)  #
trade_signal(df_te1,tr_id=True)  
plt.figure(figsize=(14,12), dpi=80)
ax1 = plt.subplot(211)
plt.plot(df_tr1.close,color='b')
plt.title('上证指数走势',size=15)
plt.xlabel('')
ax2 = plt.subplot(212)
plt.plot(df_tr1.signal,color='r')
plt.title('交易信号',size=15)
plt.xlabel('')
plt.show()