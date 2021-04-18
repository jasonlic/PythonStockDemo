#资产收益率的非平稳性——为何机器学习预测效果不佳？
# https://mp.weixin.qq.com/s/B6QhZPBVVE_uk6XiB2MT_w

#先引入后面可能用到的包（package）
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
sns.set()  
#%matplotlib inline   
#正常显示画图时出现的中文和负号
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
import tushare as ts
def get_price(code,start,end):
    df=ts.get_k_data(code,start,end)
    df.index=pd.to_datetime(df.date)
    return df.close

codes=['sh','sz','cyb','zxb','hs300']
names=['上证综指','深证综指','创业板指','中小板指','沪深300']
#changed for get today's data
end_day = pd.Timestamp.today().date()
#end_day = pd.to_datetime('2021-4-17')
start_day = end_day - 10 * 252 * pd.tseries.offsets.BDay()
start=start_day.strftime('%Y-%m-%d')
end=end_day.strftime('%Y-%m-%d')
#指数收盘价数据
df = pd.DataFrame({name:get_price(code, start, end) for name,code in zip(names,codes)})
(df/df.iloc[0]-1).plot(figsize=(12,7))
"""
plt.title('指数累计收益率\n2011-2021',size=15)
plt.show()
"""
#计算股票指数的日对数收益率。
#实际上在经济金融上，采用对数收益率已经是约定俗成了，当然这样处理主要是基于对数处理的统计特性比较适合建模，如为了使数据更加平滑，克服数据本身的异方差等。
rs=(np.log(df/df.shift(1))).dropna()
"""
rs.plot(figsize=(12,5))
plt.title('指数日对数收益率',size=15)
plt.show()
"""
#下面将演示和分析在预测时间序列时当数据不满足模型假设条件的挑战。在开始之前，我们要知道，证券收益率序列往往不满足平稳性的要求。
import scipy.stats as stats

def add_mean_std_text(x, **kwargs):
    mean, std = x.mean(), x.std()
    mean_tx = f"均值: {mean:.4%}\n标准差: {std:.4%}"

    txkw = dict(size=14, fontweight='demi', color='red', rotation=0)
    ymin, ymax = plt.gca().get_ylim()
    plt.text(mean+0.025, 0.8*ymax, mean_tx, **txkw)
    return

def plot_dist(rs, ex):
    plt.rcParams['font.size'] = 14
    g = (rs
         .pipe(sns.FacetGrid, height=5,aspect=1.5)
        #.map(sns.distplot, ex, kde=False, fit=stats.norm,fit_kws={ 'lw':2.5, 'color':'red','label':'正态分布'})
         .map(sns.distplot((ex), kde=False, fit=stats.norm,fit_kws={ 'lw':2.5, 'color':'red','label':'正态分布'}))
        #.map(sns.distplot, ex, kde=False, fit=stats.laplace,fit_kws={'linestyle':'--','color':'blue', 'lw':2.5, 'label':'拉普拉斯分布'})
         .map(sns.distplot(ex, kde=False, fit=stats.laplace,fit_kws={'linestyle':'--','color':'blue', 'lw':2.5, 'label':'拉普拉斯分布'}))
         #.map(sns.distplot, ex, kde=False, fit=stats.johnsonsu,fit_kws={'linestyle':'-','color':'green','lw':2.5, 'label':'约翰逊分布'})
         .map(sns.distplot(ex, kde=False, fit=stats.johnsonsu,fit_kws={'linestyle':'-','color':'green','lw':2.5, 'label':'约翰逊分布'}))
         .map(add_mean_std_text, ex))
    g.add_legend()
    sns.despine(offset=1)
    plt.title(f'{ex}收益率',size=15)
    return
    
plot_dist(rs, '上证综指')