#read fundstocks.cvs to draw pie 
import akshare as ak
import pandas as pd

import matplotlib.pyplot as plt
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"]=False

df = pd.read_csv('fundstocks.cvs')

#print(df)

df['总价'] = df['最新价'] * df['股数']

df.sort_values(by='总价',ascending=False,inplace=True)

df_20 = df.iloc[0:20,:]
print(df_20)
df_20['总价'].plot.pie(labels=df['名称'],autopct='%.2f')
plt.show()