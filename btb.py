#coding:utf-8
# Introduction to Sell-Off Analysis for Crypto-Assets: Triggered by Bitcoin?
# (c) 2021 QuantAtRisk.com
# https://quantatrisk.com/2021/03/15/introduction-sell-off-analysis-crypto-assets-triggered-by-bitcoin-python/
# https://mp.weixin.qq.com/s/ez-1FZXGHTzimRm-xnvGfg
import ccrypto  as cc
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
 
grey = (.8,.8,.8)
 
# 下载 BTC OHLC 
#b = cc.getCryptoSeries('BTC', 'USD', freq='h', ohlc=True, exch='Coinbase', start_date='2021-03-13 00:00')
b = cc.getcryptoseries('BTC', 'USD', freq='h', ohlc=True, exch='Coinbase', start_date='2021-03-13 00:00')
fig, ax1 = plt.subplots(1,1, figsize=(12,7))
ax1.plot(b.BTCUSD_H, 'b:')
ax1.plot(b.BTCUSD_C, 'k')
ax1.plot(b.BTCUSD_L, 'r:')
ax1.grid()
plt.title('BTC Price: Max %.2f, Min %.2f' % (b.BTCUSD_H.max(), b.BTCUSD_L.min()))