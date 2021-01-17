# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 21:10:45 2021

@author: xingxing
"""

import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime

def simple_holding_strategy(data: pd.DataFrame, weight = None):
    
    if weight is None:
        print('请指定weight')
        return 
    
    strategy_pnl = np.array([])
    strategy_weight = None
    
    for i in np.arange(data.shape[0]):
        
        if i == 0:
            strategy_pnl = np.append(strategy_pnl, 1)
            strategy_weight = weight.copy()
        else:
            
            # 1. 计算收益率
            yesterday_md = data.values[i-1, ]
            today_md = data.values[i, ]
            simple_ret = today_md / yesterday_md
            
            # 2。 更新仓位
            strategy_weight = strategy_weight * simple_ret
            
            # 3. 添加净值
            strategy_pnl = np.append(strategy_pnl, strategy_weight.sum())
        
    return strategy_pnl


def get_datetime_from_str(date):
    '''
        2015-01-05转为datetime
    '''
    return datetime.datetime.strptime(date, '%Y-%m-%d')

def is_rebalance_day(today, yesterday):
    
    today = get_datetime_from_str(today)
    yesterday = get_datetime_from_str(yesterday)
    if today.month == 7 and yesterday.month == 6:
        return False
    elif today.month == 1 and yesterday.month == 12:
        return True
    return False
    
    

def simple_holding_strategy_balance(data: pd.DataFrame, weight = None):
    
    if weight is None:
        print('请指定weight')
        return 
    
    strategy_pnl = np.array([])
    strategy_weight = None
    
    for i in np.arange(data.shape[0]):
        
        if i == 0:
            strategy_pnl = np.append(strategy_pnl, 1)
            strategy_weight = weight.copy()
            continue
        else:
            
            # 1. 计算收益率
            yesterday_md = data.values[i-1, ]
            today_md = data.values[i, ]
            simple_ret = today_md / yesterday_md
            
            # 2。 更新仓位
            strategy_weight = strategy_weight * simple_ret
            
            # 3. 添加净值
            strategy_pnl = np.append(strategy_pnl, strategy_weight.sum())
        
        yesterday = data.index[i-1]
        today = data.index[i]
        if is_rebalance_day(today, yesterday):
            current_pnl = strategy_pnl[-1]
            strategy_weight = np.array([current_pnl*weight[0], current_pnl*weight[1]])
        
    return strategy_pnl


# ===========================策略指标==================================
    
def get_max_drawdown(pnl):
    '''
    pnl: numpy 数组
    '''
    max_drawdown = 0.0
    max_pnl = 0.0
    
    for day_pnl in pnl:
        max_pnl = max(max_pnl, float(day_pnl))
        current_drawdown = (max_pnl - day_pnl) / day_pnl
        max_drawdown = max(max_drawdown, current_drawdown)
    return max_drawdown

def get_annual_return(pnl):
    first_pnl = pnl[0]
    last_pnl = pnl[-1]
    
    years = len(pnl) / 220.0
    return (last_pnl / first_pnl) ** (1 / years) - 1

def get_mar(pnl):
    
    return get_annual_return(pnl) / get_max_drawdown(pnl)

# ===============================策略指标结束=====================================