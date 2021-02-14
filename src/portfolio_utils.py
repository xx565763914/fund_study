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



class FundPortfolio(object):
    '''
        基金投资组合回测基类
    '''
    def __init__(self, assert_data: pd.DataFrame):
        self.portfolio_pnl = np.array([])
        self.portfolio_weight = None
        self.portfolio_cash = 0.0
        self.portfolio_risk = []
        self.assert_data = assert_data
    
    def backtest(self):
        days, assert_num = self.assert_data.shape
        data = self.assert_data.copy()
        
        for i in range(days):
            if i == 0:
                self.portfolio_pnl = np.append(self.portfolio_pnl, 1.0)
                self.portfolio_weight = np.array([1.0/assert_num] * assert_num)
                self.portfolio_cash = 0.0
                continue
            else:
                y_md = data.values[i-1, ]
                t_md = data.values[i, ]
                simple_return = t_md / y_md
                
                self.portfolio_weight = self.portfolio_weight * simple_return
                self.portfolio_pnl = np.append(self.portfolio_pnl, self.portfolio_weight.sum() + self.portfolio_cash)
            
            # 调仓相关
            if self.is_balance_day(i):
                self.portfolio_weight, self.portfolio_cash = self.get_rebalanced_weight(i)
            
            # 计算组合风险 i > 30
            current_portfolio_risk = np.nan
            if i > 30:
                cov = self.assert_num.iloc[i-30:i, :].pct_change().cov()
                current_portfolio_risk = reduct(np.dot, self.portfolio_weight, cov, self.portfolio_weight.T)
            self.portfolio_risk.append(dict(date = self.assert_data.index[i], portfolio_risk = current_portfolio_risk))
            
    
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
    
    def get_daily_std():
        '''
            获取组合每日风险
        '''
        return self.portfolio_risk
    
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


