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
from scipy.optimize import minimize



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


def get_datetime_from_str(date):
    '''
        2015-01-05转为datetime
    '''
    return datetime.datetime.strptime(date, '%Y-%m-%d')

def is_balance_day(today, yesterday):
    today = get_datetime_from_str(today).month
    yesterday = get_datetime_from_str(yesterday).month
    if today != yesterday:
        return True
    
    return False

def equal_weight_strategy(data:pd.DataFrame, target_year_vol = 0.08):
    row, col = data.shape
    data1 = data.copy()
    data1['cash'] = 1
    
    s_pnl = np.array([])
    s_weight = np.array([1.0/col] * col)
    s_weight = np.append(s_weight, 0)
    
    for i in np.arange(row):
        if i == 0:
            s_pnl = np.append(s_pnl, 1)
            continue
        else:
            y_md = data1.values[i-1, ]
            t_md = data1.values[i, ]
            
            simple_return = t_md / y_md
            
            s_weight = s_weight * simple_return
            s_pnl = np.append(s_pnl, s_weight.sum())
        
        yd = data1.index[i-1]
        td = data1.index[i]
        if is_balance_day(td, yd):
            cur_pnl = s_pnl[-1]
            
            zuhe_weight = 1
            if i > 45:
                s_std = np.std(s_pnl[i-35:i])
                year_vol = s_std * np.sqrt(250)
                if np.abs(year_vol) > 1e-6:
                    zuhe_weight = target_year_vol / year_vol
                if zuhe_weight > 1.0:
                    zuhe_weight = 1.0
            zuhe_weight = zuhe_weight * cur_pnl
            
            s_weight = np.array([zuhe_weight / col] * col)
            s_weight = np.append(s_weight, cur_pnl - zuhe_weight)
    return s_pnl

def risk_parity_strategy(data:pd.DataFrame, target_year_vol = 0.08):
    row, col = data.shape
    data1 = data.copy()
    data1['cash'] = 1
    
    s_pnl = np.array([])
    s_weight = np.array([1.0/col] * col)
    s_weight = np.append(s_weight, 0)
    weights = []
    
    for i in np.arange(row):
        if i == 0:
            s_pnl = np.append(s_pnl, 1)
            continue
        else:
            y_md = data1.values[i-1, ]
            t_md = data1.values[i, ]
            
            simple_return = t_md / y_md
            
            s_weight = s_weight * simple_return
            s_pnl = np.append(s_pnl, s_weight.sum())
        
        yd = data1.index[i-1]
        td = data1.index[i]
        if is_balance_day(td, yd):
            cur_pnl = s_pnl[-1]
            
            risk_parity_weight = np.array([1.0/col] * col)
            
            if i > 80:
                count = 1
                risk_weight = calc_equal_risk_contributions_weights(data1.iloc[i-30:i, 0:col].pct_change().cov().values)
                for period in range(40, 70, 2):
                    count = count + 1
                    risk_weight = risk_weight + calc_equal_risk_contributions_weights(data1.iloc[i-period:i, 0:col].cov().values)
                risk_parity_weight = risk_weight / count
            zuhe_weight = 1
            if i > 90:
                s_std = np.std(s_pnl[i-35:i])
                year_vol = s_std * np.sqrt(250)
                if np.abs(year_vol) > 1e-6:
                    zuhe_weight = target_year_vol / year_vol
                if zuhe_weight > 1.0:
                    zuhe_weight = 1.0
            
            old_weight = s_weight.copy()
            new_weight = np.array(risk_parity_weight * zuhe_weight * cur_pnl)
            new_weight = np.append(new_weight, cur_pnl - (risk_parity_weight * zuhe_weight * cur_pnl).sum())
            
            commission = np.nansum(np.abs(new_weight - old_weight)) * 1 / 10000.0
            cur_pnl = cur_pnl - commission
            
            # 再次计算仓位
            s_weight = np.array(risk_parity_weight * zuhe_weight * cur_pnl)
            s_weight = np.append(s_weight, cur_pnl - (risk_parity_weight * zuhe_weight * cur_pnl).sum())
            # print(s_weight.round(2
        weights.append(s_weight)
    return s_pnl, weights


# ============================风险平价策略==============================

def calc_portfolio_risk_contribution(weights, covmat):
    """
    Compute the contributions to risk of the constituents of a portfolio, given a set of portfolio weights and a covariance matrix
    """
    total_portfolio_var = calc_portfolio_vol(weights, covmat)**2
    # Marginal contribution of each constituent
    marginal_contrib = covmat@weights
    risk_contrib = np.multiply(marginal_contrib, weights.T)/total_portfolio_var
    return risk_contrib

def calc_target_risk_contributions_weights(target_risk, cov):
    """
    Returns the weights of the portfolio that gives you the weights such
    that the contributions to portfolio risk are as close as possible to
    the target_risk, given the covariance matrix
    """
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    
    def msd_risk(weights, target_risk, cov):
        """
        Returns the Mean Squared Difference in risk contributions
        between weights and target_risk
        """
        w_contribs = calc_portfolio_risk_contribution(weights, cov)
        return ((w_contribs-target_risk)**2).sum()
    
    weights = minimize(msd_risk, init_guess,
                       args=(target_risk, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def calc_equal_risk_contributions_weights(cov):
    """
    Returns the weights of the portfolio that equalizes the contributions
    of the constituents based on the given covariance matrix
    """
    n = cov.shape[0]
    return calc_target_risk_contributions_weights(target_risk=np.repeat(1/n,n), cov=cov)

def calc_portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    vol = (weights.T @ covmat @ weights)**0.5
    return vol 

# ===========================策略指标==================================
    
def get_max_drawdown(pnl):
    '''
    pnl: numpy 数组
    '''
    max_drawdown = 0.0
    max_pnl = 0.0
    
    for day_pnl in pnl:
        max_pnl = max(max_pnl, float(day_pnl))
        current_drawdown = (max_pnl - day_pnl) / max_pnl
        max_drawdown = max(max_drawdown, current_drawdown)
    return max_drawdown

def get_annual_return(pnl):
    first_pnl = pnl[0]
    last_pnl = pnl[-1]
    
    years = len(pnl) / 250.0
    return (last_pnl / first_pnl) ** (1 / years) - 1

def get_mar(pnl):
    
    return get_annual_return(pnl) / get_max_drawdown(pnl)

def performance(pnl):
    return dict(max_drawdown = get_max_drawdown(pnl), 
                annual_return = get_annual_return(pnl),
                mar = get_mar(pnl))

# ===============================策略指标结束=====================================
    

# ===================数据拉取程序============================
# 抓取网页
def get_url(url, params=None, proxies=None):
    rsp = requests.get(url, params=params, proxies=proxies)
    rsp.raise_for_status()
    return rsp.text

# 从网页抓取数据
def get_fund_data(code,per=300,sdate='',edate='',proxies=None):
    url = 'http://fund.eastmoney.com/f10/F10DataApi.aspx'
    params = {'type': 'lsjz', 'code': code, 'page':1,'per': per, 'sdate': sdate, 'edate': edate}
    html = get_url(url, params, proxies)
    soup = BeautifulSoup(html, 'html.parser')

    # 获取总页数
    pattern=re.compile(r'pages:(.*),')
    result=re.search(pattern,html).group(1)
    pages=int(result)

    # 获取表头
    heads = []
    for head in soup.findAll("th"):
        heads.append(head.contents[0])

    # 数据存取列表
    records = []

    # 从第1页开始抓取所有页面数据
    print(pages)
    page=1
    while page<=pages:
        print(page)
        params = {'type': 'lsjz', 'code': code, 'page':page,'per': per, 'sdate': sdate, 'edate': edate}
        html = get_url(url, params, proxies)
        soup = BeautifulSoup(html, 'html.parser')

        # 获取数据
        for row in soup.findAll("tbody")[0].findAll("tr"):
            row_records = []
            for record in row.findAll('td'):
                val = record.contents

                # 处理空值
                if val == []:
                    row_records.append(np.nan)
                else:
                    row_records.append(val[0])

            # 记录数据
            records.append(row_records)

        # 下一页
        page=page+1
        time.sleep(2)

    # 数据整理到dataframe
    np_records = np.array(records)
    data= pd.DataFrame()
    for col,col_name in enumerate(heads):
        data[col_name] = np_records[:,col]

    return data


