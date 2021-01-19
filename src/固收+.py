# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 23:22:00 2021

@author: xingxing
"""

from src.common import *

# ================简单组合策略 15% - 85%====================

guozhai = pd.read_csv('data/000012.csv', encoding='gbk')
hs_300 = pd.read_csv('data/000300.csv', encoding='gbk')


guozhai = pd.read_csv('data/217011.csv')
guozhai = guozhai.set_index('净值日期').sort_index()

hs_300.set_index('日期', inplace=True)
hs_300.sort_index(inplace=True)

total = pd.concat([guozhai['累计净值'].rename('债券'), hs_300.收盘价.rename('沪深300')], axis = 1)

total.sort_index(inplace=True)

total_2015 = total[total.index >= '2015-01-01']

total_2015 = total_2015.dropna()

total_2015

pnl = simple_holding_strategy_balance(total_2015, np.array([0.85, 0.15]))

get_annual_return(pnl)
get_max_drawdown(pnl)
get_mar(pnl)

plt.plot(pnl)

'''
get_annual_return(pnl)
Out[40]: 0.047026454832702624

get_max_drawdown(pnl)
Out[41]: 0.10226021189244701

get_mar(pnl)
Out[42]: 0.45987050058299384
'''

# ============================================

guozhai_rate = pd.read_csv('data/中国十年期国债收益率历史数据.csv')

guozhai_rate

def to_time_str(date):
    year_index = date.find('年')
    month_index= date.find('月')
    day_index = date.find('日')
    # print(year_index, month_index, day_index)
    year = int(date[0:year_index])
    month = int(date[year_index+1:month_index])
    day = int(date[month_index+1:day_index])
    date = datetime.datetime(year, month, day).strftime('%Y-%m-%d')
    return date

guozhai_rate['date'] = guozhai_rate['日期'].apply(to_time_str)
guozhai_rate.set_index('date', inplace=True)
guozhai_rate.sort_index(inplace=True)

guozhai_rate

guozhai_rate.收盘.plot(figsize=(16, 9))

ts.set_token('ec4caba8049b4697b0d6006052c0ac3c6aae5ce8927463b296cbd2d6')

pro = ts.pro_api()

hs300_pe = pd.read_csv(r'data/000300_pe.csv')
hs300_pe.head()

hs300_pe.set_index('trade_date', inplace=True)

hs300_pe.plot(subplots=True)

indicator_股债性价比 = pd.concat([hs300_pe.pe_ttm.rename('pe'), guozhai_rate.收盘.rename('利率')], axis=1)

indicator_股债性价比.sort_index(inplace=True)

indicator_股债性价比['股债性价比'] = (1.0 / indicator_股债性价比.pe) / indicator_股债性价比.利率

indicator_股债性价比.股债性价比.plot(figsize = (16, 9))

indicator_股债性价比.info()

indicator_股债性价比.shape


pd.concat([indicator_股债性价比.股债性价比, hs_300.收盘价], axis = 1).sort_index().dropna().plot(figsize=(16, 9), subplots=True)


indicator_股债性价比.股债性价比
data_股债性价比 = total_2015.copy()
data_股债性价比['cash'] = 1
data_股债性价比

def is_balance_day(today, yesterday):
    today = get_datetime_from_str(today).month
    yesterday = get_datetime_from_str(yesterday).month
    if today != yesterday:
        return True
    
    return False

# 债券 股票 现金
s_w = np.array([0, 0, 1])
s_ret = np.array([])

for i in np.arange(0, total_2015.shape[0]):
    if i == 0:
        s_w = np.array([0, 0, 1])
        s_ret = np.append(s_ret, 1)
        continue
    else:
        yd_md = data_股债性价比.values[i-1, ]
        td_md = data_股债性价比.values[i, ]
        s_rtn = td_md / yd_md
        
        s_w = s_w * s_rtn
        s_ret = np.append(s_ret, s_w.sum())
    
    if is_balance_day(data_股债性价比.index[i], data_股债性价比.index[i-1]):
        print(data_股债性价比.index[i])
        cur_股债性价比 = indicator_股债性价比.股债性价比[data_股债性价比.index[i]]
        cur_ret = s_ret[-1]
        
        zhaiquan_ratio = cur_ret * 0.8
        gupiao_ratio = (cur_ret - zhaiquan_ratio) * ((cur_股债性价比 - 0.015) / (0.035 - 0.015))
        cash_ratio = cur_ret - zhaiquan_ratio - gupiao_ratio
        
        s_w = np.array([zhaiquan_ratio, gupiao_ratio, cash_ratio])
        print(s_w)
    # print(s_w)
        
get_annual_return(s_ret)
get_max_drawdown(s_ret)
get_mar(s_ret)
