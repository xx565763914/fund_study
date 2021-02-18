#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 18:55:27 2021

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
from functools import reduce

def get_datetime_from_str(date):
    '''
        2015-01-05转为datetime
    '''
    return datetime.datetime.strptime(date, '%Y-%m-%d')

class FundPortfolio(object):
    '''
        基金投资组合回测基类
    '''
    def __init__(self, assert_data: pd.DataFrame):
        self.portfolio_pnl = []    # 组合pnl曲线
        self.portfolio_weight = None # 组合权重
        self.portfolio_cash = 0.0 # 组合现金保有量
        self.portfolio_risk = []
        self.assert_data = assert_data
        self.history_weights = []
        
    
    def backtest(self):
        days, assert_num = self.assert_data.shape
        data = self.assert_data.copy()
        
        for i in range(days):
            if i == 0:
                self.portfolio_pnl.append(dict(date=self.assert_data.index[i], pnl = 1.0))
                self.portfolio_weight = np.array([1.0/assert_num] * assert_num)
                self.portfolio_cash = 0.0
                continue
            else:
                y_md = data.values[i-1, ]
                t_md = data.values[i, ]
                simple_return = t_md / y_md
                
                self.portfolio_weight = self.portfolio_weight * simple_return
                self.portfolio_pnl.append(dict(date=self.assert_data.index[i], pnl = self.portfolio_weight.sum() + self.portfolio_cash))
            
            # 调仓相关
            if self.is_balance_day(i):
                self.portfolio_weight, self.portfolio_cash = self.get_rebalanced_weight(i)
            
            # 计算组合风险 i > 30
            current_portfolio_risk = np.nan
            if i > 30:
                cov = self.assert_data.iloc[i-30:i, :].pct_change().cov()
                assert_weight = self.portfolio_weight / self.portfolio_weight.sum()
                current_portfolio_risk = np.sqrt(reduce(np.dot, [assert_weight, cov, assert_weight.T])) * np.sqrt(250)
                current_portfolio_risk = current_portfolio_risk * self.portfolio_weight.sum() / \
                    self.get_current_portfolio_pnl()
            self.portfolio_risk.append(dict(date = self.assert_data.index[i], portfolio_risk = current_portfolio_risk))
            
            # 记录投资组合历史权重
            self.history_weights.append(dict(date = self.assert_data.index[i], weight = self.portfolio_weight, cash = self.portfolio_cash))
    
    def get_current_portfolio_pnl(self):
        return self.portfolio_weight.sum() + self.portfolio_cash
    
    def is_balance_day(self, current_day):
        raise NotImplementedError("调仓日判断逻辑没有实现！！！！")
    
    def get_rebalanced_weight(self, current_day):
        '''
            可以通过currentDay做一个预热逻辑
        '''
        raise NotImplementedError("调仓逻辑没有实现！！！！")
        
    def get_portfolio_pnl(self):
        return self.portfolio_pnl
    
    def get_daily_std(self):
        '''
            获取组合每日风险
        '''
        return self.portfolio_risk
    
    def get_history_weight(self):
        '''
             获取历史仓位权重
        '''
        return self.history_weights

class RebalanceWhenExceed(FundPortfolio):
    '''
        组合波动率阈值触发调仓
    '''
    
    def __init__(self, assert_data: pd.DataFrame, min_vol: float, max_vol: float):
        super(RebalanceWhenExceed, self).__init__(assert_data)
        self.min_vol = min_vol
        self.max_vol = max_vol
    
    def is_balance_day(self, current_day):
        if current_day < 30:
            return False

        cov = self.assert_data.iloc[current_day-30:current_day, :].pct_change().cov()
        assert_weight = self.portfolio_weight / self.portfolio_weight.sum()
        current_portfolio_risk = np.sqrt(reduce(np.dot, [assert_weight, cov, assert_weight.T])) * np.sqrt(250)
        current_portfolio_risk = current_portfolio_risk * self.portfolio_weight.sum() / \
            self.get_current_portfolio_pnl()
            
        # print(self.assert_data.index[current_day], current_portfolio_risk)
        if not (current_portfolio_risk > self.min_vol and current_portfolio_risk < self.max_vol):
            # print('调仓')
            return True
        
        # print('不调仓')
        return False
    
    
class MonthRebalancePortfolio(FundPortfolio):
    '''
        按月调仓投资组合基类
    '''
    def __init__(self, assert_data: pd.DataFrame):
        super(MonthRebalancePortfolio, self).__init__(assert_data)
    
    def get_datetime_from_str(self, date):
        '''
            2015-01-05转为datetime
        '''
        return datetime.datetime.strptime(date, '%Y-%m-%d')

    def _is_balance_day(self, today, yesterday):
        today = get_datetime_from_str(today).month
        yesterday = get_datetime_from_str(yesterday).month
        if today != yesterday:
            return True

        return False
    
    def is_balance_day(self, current_day):
        yd = self.assert_data.index[current_day-1]
        td = self.assert_data.index[current_day]
        return self._is_balance_day(td, yd)
    
    def get_rebalanced_weight(self, current_day):
        raise NotImplementedError("调仓逻辑没有实现！！！")

class EqualWeightPortfolio(MonthRebalancePortfolio):
    '''
        等权投资组合类
    '''
    
    def __init__(self, assert_data: pd.DataFrame):
        super(EqualWeightPortfolio, self).__init__(assert_data)
    
    def get_rebalanced_weight(self, currentDay):
        portfolio_pnl = self.get_current_portfolio_pnl()
        assert_num = self.assert_data.shape[1]
        
        new_weight = np.array([portfolio_pnl/assert_num] * assert_num)
        new_cash = 0.0
        
        return new_weight, new_cash


class VolatilityPortfolio(MonthRebalancePortfolio):
    
    '''
        波动率倒数调仓
    '''
    
    def __init__(self, assert_data: pd.DataFrame):
        super(VolatilityPortfolio, self).__init__(assert_data)
        
    def get_rebalanced_weight(self, current_day):
        if current_day < 30:
            return self.portfolio_weight, self.portfolio_cash
        
        portfolio_pnl = self.get_current_portfolio_pnl()
        
        data = self.assert_data.iloc[current_day-30:current_day, :]
        weight = 1.0 / data.pct_change().std().values
        weight = weight / weight.sum()
        
        weight = weight * portfolio_pnl
        return weight, 0.0

