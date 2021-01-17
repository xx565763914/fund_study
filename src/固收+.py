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


'''
get_annual_return(pnl)
Out[133]: 0.047026454832702624

get_max_drawdown(pnl)
Out[134]: 0.11390852143026081

get_mar(pnl)
Out[135]: 0.4128440457502912
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


