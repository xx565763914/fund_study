# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 19:55:15 2021

@author: qinxi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.tiantian_funds_datasource import *
from src.portfolio_utils import *
from src.performance_utils import *
%matplotlib inline


#%% 整理数据
def read_fund_frame(frame_path: str):
    with open(frame_path, 'r', encoding='utf-8') as fp:
        frame = pd.read_csv(fp)
    return frame

def normalize_frame(fund_frame: pd.DataFrame):
    return fund_frame.set_index('净值日期').sort_index()

纳斯达克_数据路径 = r'./data/513100_纳斯达克etf.csv'
纳斯达克 = normalize_frame(read_fund_frame(纳斯达克_数据路径))

国债_数据路径 = r'./data/159926_国债etf.csv'
国债 = normalize_frame(read_fund_frame(国债_数据路径))

沪深300_数据路径 = r'./data/159919_沪深300etf.csv'
沪深300 = normalize_frame(read_fund_frame(沪深300_数据路径))

total = pd.concat([纳斯达克.累计净值.rename('纳斯达克'),
                   国债.累计净值.rename('国债'),
                   沪深300.累计净值.rename('沪深300')], axis = 1).sort_index().fillna(method = 'ffill')


(total / total.iloc[0, :]).plot(figsize = (16, 9))
    

#%% 等权策略

from functools import reduce

s = EqualWeightPortfolio(total)
s.backtest()

#s_pnl = s.get_portfolio_pnl()
# pd.Series(data = s_pnl).plot(figsize = (16, 9))     

pd.DataFrame(s.get_portfolio_pnl()).set_index('date').sort_index().plot(figsize = (16, 9))
pd.DataFrame(s.get_daily_std()).set_index('date').plot(figsize = (16, 9))


#%% 波动率倒数

s = VolatilityPortfolio(total)
s.backtest()

pd.DataFrame(s.get_portfolio_pnl()).set_index('date').sort_index().plot(figsize = (16, 9))
pd.DataFrame(s.get_daily_std()).set_index('date').plot(figsize = (16, 9))



#%% 风险平价




#%% 目标波动率15% 沪深300


class TargetVolHs300(MonthRebalancePortfolio):
    
    def __init__(self, assert_data):
        super(TargetVolHs300, self).__init__(assert_data)
    
    def get_rebalanced_weight(self, currentDay):
        
        if currentDay < 31:
            return self.portfolio_weight, self.portfolio_cash
        
        # 计算当前的波动率
        data = self.assert_data.iloc[currentDay - 30:currentDay, :]
        annual_vol = data.pct_change().std().values * np.sqrt(250)
        杠杆率 = (0.1 / annual_vol)[0]
        if 杠杆率 > 1:
            杠杆率 = 1
        
        current_pnl = self.get_current_portfolio_pnl()
        portfolio_weight = current_pnl * 杠杆率
        portfolio_cash = current_pnl - portfolio_weight
        
        return np.array([portfolio_weight]), portfolio_cash

s = TargetVolHs300(pd.DataFrame(data = total.沪深300, index = total.index).fillna(method = 'ffill'))
s.backtest()


pd.DataFrame(s.get_portfolio_pnl()).set_index('date').sort_index().plot(figsize = (16, 9))
pd.DataFrame(s.get_daily_std()).set_index('date').plot(figsize = (16, 9))

