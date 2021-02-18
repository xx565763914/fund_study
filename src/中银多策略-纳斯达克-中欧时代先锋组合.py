#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 23:31:44 2021

@author: xingxing
"""

#%% 要引入的包

import pandas as pd
import numpy as np
from src.tiantian_funds_datasource import *
from src.portfolio_utils import *
from src.performance_utils import *

#%% 拉取数据

中银多策略混合 = get_fund_data('000572', sdate = '2014-01-01', edate = '2021-02-01')
中欧时代先锋 = get_fund_data('001938', sdate = '2014-01-01', edate = '2021-02-01')

中银多策略混合.head()
中银多策略混合.to_csv('./data/中银多策略混合A.csv', index = 0)

中欧时代先锋.head()
中欧时代先锋.to_csv('./data/中欧时代先锋A.csv', index = 0)

易方达安心回馈组合 = get_fund_data('001182', sdate = '2014-01-01', edate = '2021-02-01')

易方达安心回馈组合.head()
易方达安心回馈组合.to_csv('./data/易方达安心回馈组合.csv', index = 0)

交银优势行业混合 = get_fund_data('519697', sdate = '2014-01-01', edate = '2021-02-01')

交银优势行业混合.head()
交银优势行业混合.to_csv(r'./data/交银优势行业混合.csv', index = 0)

易方达消费行业股票 = get_fund_data('110022', sdate = '2014-01-01', edate = '2021-02-01')

易方达消费行业股票.head()
易方达消费行业股票.to_csv(r'./data/易方达消费行业股票.csv', index = 0)


#%% 整理数据

def read_fund_frame(frame_path: str):
    with open(frame_path, 'r', encoding='utf-8') as fp:
        frame = pd.read_csv(fp)
    return frame

def normalize_frame(fund_frame: pd.DataFrame):
    return fund_frame.set_index('净值日期').sort_index()

纳斯达克_数据路径 = r'./data/513100_纳斯达克etf.csv'
纳斯达克 = normalize_frame(read_fund_frame(纳斯达克_数据路径))

中银多策略混合路径 = r'./data/中银多策略混合A.csv'
中银多策略混合 = normalize_frame(read_fund_frame(中银多策略混合路径))

中银多策略混合


中欧时代先锋 = r'./data/中欧时代先锋A.csv'
中欧时代先锋 = normalize_frame(read_fund_frame(中欧时代先锋))

中欧时代先锋

易方达安心回馈组合路径 = r'./data/易方达安心回馈组合.csv'
易方达安心回馈组合 = normalize_frame(read_fund_frame(易方达安心回馈组合路径))

交银优势行业混合路径= r'./data/交银优势行业混合.csv'
交银优势行业混合 = normalize_frame(read_fund_frame(交银优势行业混合路径))

交银优势行业混合

易方达消费行业股票路径 = r'./data/易方达消费行业股票.csv'
易方达消费行业股票 = normalize_frame(read_fund_frame(易方达消费行业股票路径))
易方达消费行业股票

total = pd.concat([
        纳斯达克.累计净值.rename('纳斯达克'),
        中银多策略混合.累计净值.rename('中银多策略混合'),
        中欧时代先锋.累计净值.rename('中欧时代先锋'),
        易方达安心回馈组合.累计净值.rename('易方达安心回馈组合'),
        交银优势行业混合.累计净值.rename('交银优势行业混合'),
        易方达消费行业股票.累计净值.rename('易方达消费行业股票')
    ], axis = 1).sort_index().fillna(method = 'ffill').dropna()

(total / total.iloc[0, :]).plot(figsize = (16, 9))


#%% 等权重策略

s = EqualWeightPortfolio(total)
s.backtest()

#s_pnl = s.get_portfolio_pnl()
# pd.Series(data = s_pnl).plot(figsize = (16, 9))     

pd.DataFrame(s.get_portfolio_pnl()).set_index('date').sort_index().plot(figsize = (16, 9))
pd.DataFrame(s.get_daily_std()).set_index('date').plot(figsize = (16, 9))

performance(pd.DataFrame(s.get_portfolio_pnl()).set_index('date').pnl.values)


#%% 波动率倒数仓位

s = VolatilityPortfolio(total)
s.backtest()

pd.DataFrame(s.get_portfolio_pnl()).set_index('date').sort_index().plot(figsize = (16, 9))
pd.DataFrame(s.get_daily_std()).set_index('date').plot(figsize = (16, 9))

pd.DataFrame(s.get_history_weight()).set_index('date').sort_index().to_csv('./data/weights.csv')

performance(pd.DataFrame(s.get_portfolio_pnl()).set_index('date').pnl.values)

#%% 季度调仓


class RebalanceStrategy1Portfolio(FundPortfolio):
    '''
        按季度调仓投资组合基类
    '''
    def __init__(self, assert_data: pd.DataFrame):
        super(RebalanceStrategy1Portfolio, self).__init__(assert_data)
    
    def get_datetime_from_str(self, date):
        '''
            2015-01-05转为datetime
        '''
        return datetime.datetime.strptime(date, '%Y-%m-%d')

    def _is_balance_day(self, today, yesterday):
        today = get_datetime_from_str(today).month
        yesterday = get_datetime_from_str(yesterday).month
        # if today != yesterday and (today == 1 or today == 4 or today == 7 or today == 10): # 一个季度调节一次
        if today != yesterday and (today == 1): # 一年调节一次
            return True

        return False
    
    def is_balance_day(self, current_day):
        yd = self.assert_data.index[current_day-1]
        td = self.assert_data.index[current_day]
        return self._is_balance_day(td, yd)
    
    def get_rebalanced_weight(self, current_day):
        raise NotImplementedError("调仓逻辑没有实现！！！")


class EqualWeight1Portfolio(RebalanceStrategy1Portfolio):
    '''
        等权投资组合类
    '''
    
    def __init__(self, assert_data: pd.DataFrame):
        super(EqualWeight1Portfolio, self).__init__(assert_data)
    
    def get_rebalanced_weight(self, currentDay):
        portfolio_pnl = self.get_current_portfolio_pnl()
        assert_num = self.assert_data.shape[1]
        
        new_weight = np.array([portfolio_pnl/assert_num] * assert_num)
        new_cash = 0.0
        
        return new_weight, new_cash


class Volatility1Portfolio(RebalanceStrategy1Portfolio):
    
    '''
        波动率倒数调仓
    '''
    
    def __init__(self, assert_data: pd.DataFrame):
        super(Volatility1Portfolio, self).__init__(assert_data)
        
    def get_rebalanced_weight(self, current_day):
        if current_day < 30:
            return self.portfolio_weight, self.portfolio_cash
        
        portfolio_pnl = self.get_current_portfolio_pnl()
        
        data = self.assert_data.iloc[current_day-30:current_day, :]
        weight = 1.0 / data.pct_change().std().values
        weight = weight / weight.sum()
        
        weight = weight * portfolio_pnl
        return weight, 0.0


class EqualWeightVolitilityPortfolio(RebalanceStrategy1Portfolio):
    
    '''
        等权波动率倒数权重
    '''
    def __init__(self, assert_data: pd.DataFrame):
        super(EqualWeightVolitilityPortfolio, self).__init__(assert_data)
        
    def get_rebalanced_weight(self, current_day):
        if current_day < 40:
            return self.portfolio_weight, self.portfolio_cash
        
        print('调仓', self.assert_data.index[current_day])
        
        portfolio_pnl = self.get_current_portfolio_pnl()
        assert_num = self.assert_data.shape[1]
        
        data = self.assert_data.iloc[current_day-40:current_day, :]
        weight = 1.0 / data.pct_change().std().values
        weight = weight / weight.sum()
        
        equal_weight = np.array([1 / assert_num] * assert_num)
        
        
        weight = (weight + equal_weight) / 2.0
        weight = weight * portfolio_pnl
        return weight, 0.0


pd.DataFrame(s.get_history_weight()).set_index('date').sort_index().to_csv('./data/weights.csv')


s = EqualWeightVolitilityPortfolio(total)
s.backtest()

pd.DataFrame(s.get_portfolio_pnl()).set_index('date').sort_index().plot(figsize = (16, 9))
pd.DataFrame(s.get_daily_std()).set_index('date').plot(figsize = (16, 9))

performance(pd.DataFrame(s.get_portfolio_pnl()).set_index('date').pnl.values)

#%% 根据波动率调仓


class EqualWeightVolTragger(RebalanceWhenExceed):
    
    
    def __init__(self, assert_data: pd.DataFrame, min_vol: float, max_vol: float):
        super(EqualWeightVolTragger, self).__init__(assert_data, min_vol, max_vol)
        self.min_vol = min_vol
        self.max_vol = max_vol
    
    def get_rebalanced_weight(self, currentDay):
        
        if currentDay < 30:
            return self.portfolio_weight, self.portfolio_cash
        
        portfolio_pnl = self.get_current_portfolio_pnl()
        assert_num = self.assert_data.shape[1]
        
        new_weight = np.array([1/assert_num] * assert_num)
        new_cash = 0.0
        
        # 计算组合波动率
        cov = self.assert_data.iloc[currentDay-30:currentDay, :].pct_change().cov()
        assert_weight = new_weight / new_weight.sum()
        current_portfolio_risk = np.sqrt(reduce(np.dot, [assert_weight, cov, assert_weight.T])) * np.sqrt(250)


        杠杆率 = 0.15 / current_portfolio_risk
        
        if 杠杆率 > 1:
            return self.portfolio_weight, self.portfolio_cash
        
        print(self.assert_data.index[currentDay], '调仓')
        new_cash = (1 - 杠杆率) * portfolio_pnl
        new_weight = portfolio_pnl * 杠杆率 * new_weight
        
        return new_weight, new_cash

s = EqualWeightVolTragger(total, 0.10, 0.18)
s.backtest()

pd.DataFrame(s.get_portfolio_pnl()).set_index('date').sort_index().plot(figsize = (16, 9))
pd.DataFrame(s.get_daily_std()).set_index('date').plot(figsize = (16, 9))

performance(pd.DataFrame(s.get_portfolio_pnl()).set_index('date').pnl.values)

pd.DataFrame(s.get_history_weight()).set_index('date').sort_index().to_csv('./data/weights.csv')
