# -*- coding: utf-8 -*
#【Python量化】股票涨停板探索性分析与数据挖掘
# https://mp.weixin.qq.com/s/RAr4XPj8srX_iZhxIB_Gfw

import pandas as pd
import numpy as np
#画图
import matplotlib.pyplot as plt
#正确显示中文和负号
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#处理时间
from dateutil.parser import parse
from datetime import datetime,timedelta
#使用tushare获取数据
import tushare as ts 
"""
token='输入你在tushare上注册的token'

pro=ts.pro_api(token)

#获取最新交易日期
#获取交易日历
cals=pro.trade_cal(exchange='SSE')
cals=cals[cals.is_open==1].cal_date.values
def get_now_date():
    #获取当天日期时间
    d=datetime.now().strftime('%Y%m%d')
    while d not in cals:
        d1=parse(d)
        d=(d1-timedelta(1)).strftime('%Y%m%d')
    return d
d1=get_now_date()
n1=np.argwhere(cals==d1)[0][0]+1
#获取最近6年的交易日行情
#实际上tushare只能获取2016后的涨跌停数据
dates=cals[-250*6:n1]

df=pro.limit_list(trade_date=dates[0], limit_type='U')
for date in dates[1:]:
    df_tem=pro.limit_list(trade_date=date, limit_type='U')
    df=pd.concat([df,df_tem])
#查看前几行数据
#实际上tushare只能获取2016后的涨跌停数据
#数据下载3-4分钟左右
df.head()
"""
# use data from up_limit_data.csv
import csv
df=pd.read_csv('up_limit_data.csv',index_col=0)
df.iloc[:,1:].describe().round(2)
def dy_zh(data, cut_points, labels=None): 
    min_num = data.min() 
    max_num = data.max() 
    break_points = [min_num] + cut_points + [max_num]
    if not labels: 
        labels = range(len(cut_points)+1)
    else: 
         labels=[labels[i] for i in range(len(cut_points)+1)] 
    dataBin = pd.cut(data,bins=break_points,
         labels=labels,include_lowest=True)    
    return dataBin 

cut_points = [10,30,50,100] 
labels=['10元以下', '10-30元','30-50元','50-100元','100元以上'] 
#调用函数dy_zh,增加新列
df['价格区间'] = dy_zh(df['close'], cut_points, labels) 
#查看标签列，取值范围前面加上了序号，是便于后面生成表格时按顺序排列
df.head()
#使用柱状图展示不同价格区间下涨停个股数量分布。
group_price=df.groupby('价格区间')['trade_date'].count()

plt.figure(figsize=(12,5))
colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
fig=plt.bar(group_price.index,group_price.values,color=colors[:5]);
#自动添加标签
def autolabel(fig):
    for f in fig:
        h=f.get_height()
        plt.text(f.get_x()+f.get_width()/2,1.02*h,
        f'{int(h)}',ha='center',va='bottom')
autolabel(fig)

def plot_bar(group_data):
    plt.figure(figsize=(16,5))
    fig=plt.bar(group_data.index,group_data.values);
    autolabel(fig)
    plt.title('2016-2021涨停板排名前20',size=15);

group_name=df.groupby('name')['ts_code'].count().sort_values(ascending=False)[:20]
plot_bar(group_name)

#分别剔除ST、*ST和新股（N开头）
df_st=df[-(df.name.str.startswith('ST') | df.name.str.startswith('*ST')|df.name.str.startswith('N'))]
group_name_st=df_st.groupby('name')['ts_code'].count().sort_values(ascending=False)[:20]
plot_bar(group_name_st)

#使用0.5.11版本的pyecharts
#from pyecharts import Bar
from pyecharts.charts import Bar

from pyecharts import options as opts

count_=df.groupby('trade_date')['trade_date'].count()
attr=count_.index
v1=count_.values
#bar=Bar('每日涨停板个数','2016-2021',title_text_size=15)
bar=Bar()
bar.set_global_opts(title_opts=opts.TitleOpts(
    title="每日涨停板个数",
    subtitle="2016-2021",
    pos_left="20%"
))
#bar.add('',attr,v1,is_splitline_show=False,is_datazoom_show=True,linewidth=2)
#bar
bar.render()
"""
#获取股票列表
stocks=pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
#排除新股
stocks=stocks[stocks.list_date<(parse(get_now_date())-timedelta(60)).strftime('%Y%m%d')]
dff=pd.merge(df[['trade_date','ts_code','name','close','pct_chg','fc_ratio','fl_ratio']],stocks[['ts_code','name','industry','list_date']])
#dff.head()

(dff.groupby('industry')['name'].count().sort_values(ascending=False)[:10]
 .plot.bar(figsize=(14,5),rot=0));
plt.title('2016-2021涨停板行业排名前十',size=15);

dff['year']=(pd.to_datetime(dff['trade_date'].astype(str))).dt.strftime('%Y')
#对每年行业涨停板个数排名进行可视化
#生成2*3六个子图
plot_pos=list(range(321,327))
#每个子图颜色
colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
fig=plt.figure(figsize=(18,14))
fig.suptitle('2016-2021行业涨停排名前十',size=15)
years=sorted(dff['year'].unique())
for i in np.arange(len(plot_pos)):
    ax=fig.add_subplot(plot_pos[i])
    (dff[dff.year==years[i]].groupby('industry')['name']
                     .count()
                     .sort_values(ascending=False)[:10]
                     .plot.bar(rot=0,color=colors[i]));
    ax.set_title(years[i])
    ax.set_xlabel('')
plt.show()

new_name=['汽车','电力','有色金融','钢铁','农林牧渔','医药生物','房地产','交通运输','煤炭','金融','食品饮料',
          '石油','公用事业','计算机','电子','通信','休闲服务','纺织服装','商业贸易','建筑装饰','机械设备','轻工制造','化工']
old_name=[('汽车配件', '汽车整车','汽车服务','摩托车',),('火力发电','新型电力', '水力发电'),('黄金', '铝','小金属','铅锌','铜',),
         ('普钢','特种钢','钢加工',),( '渔业', '种植业','林业','农业综合','饲料', '农药化肥','橡胶', ),('医疗保健','生物制药','医药商业','中成药','化学制药',),
         ('房产服务', '区域地产','全国地产','园区开发',),('公路','铁路','水运', '航空','空运','公共交通','路桥','港口','船舶', '仓储物流',  ),
         ('煤炭开采','焦炭加工',),('证券','保险','多元金融','银行'),('啤酒','食品', '乳制品', '红黄酒','白酒','软饮料',),
         ('石油开采','石油加工','石油贸易'),('供气供热','水务','环境保护', ),
         ('互联网', '软件服务', 'IT设备', ),('半导体', '元器件',),('通信设备','电信运营',),( '文教休闲','旅游服务','旅游景点','酒店餐饮','影视音像','出版业',),
         ('染料涂料','服饰','纺织','纺织机械','家居用品'), ('商品城','百货', '批发业', '超市连锁','电器连锁', '其他商业','商贸代理','广告包装'),
         ('建筑工程','装修装饰','其他建材','水泥'),('专用机械','轻工机械','化工机械','机械基件','运输设备','机床制造','农用机械','工程机械', '电器仪表'),
         ('造纸','陶瓷','玻璃', '塑料','矿物制品',),('化工原料','化纤','日用化工')]

#将某些细分行业合并成大类
for i in range(len(old_name)):
    for j in old_name[i]:
        dff.replace(j,new_name[i],inplace=True)
industry_up=pd.DataFrame()
#获取最近10日各行业涨停板数据
for d in dates[-10:]:
    industry_up[d]=dff[dff.trade_date==d].groupby('industry')['name'].count()
industry_up.fillna(0).sort_values(dates[-1],ascending=False).astype(int)

#近期滚动5天行业涨停个股数
(industry_up.fillna(0).T.rolling(5).sum()).T.dropna(axis=1).sort_values(dates[-1],ascending=False)

"""