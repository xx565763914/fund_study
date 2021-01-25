# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 21:41:37 2021

@author: xingxing
"""

## 仓位计算器

from src.common import *


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
    
    huangjin_less_0_15 = {
         'type': 'ineq',
        'fun': lambda weights: -weights[2] + 0.2
    }
    
    duanzhai_less_0_75 = {
        'type': 'ineq',
        'fun': lambda weights: -weights[3] + 0.65
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
                       constraints=(weights_sum_to_1, duanzhai_less_0_75, huangjin_less_0_15),
                       bounds=bounds)
    return weights.x


start_day = '2020-08-01'
today = '2021-01-24'
total_money = 15000
target_year_vol = 0.08

def get_close(data):
    data = data.set_index('净值日期').sort_index()
    return data.累计净值


au_etf = get_fund_data('518880', per=60, sdate=start_day, edate=today)
a_gu = get_fund_data('006649', per=60, sdate=start_day, edate=today)
qqq = get_fund_data('513100', per=60, sdate=start_day, edate=today)
duanzhai = get_fund_data('000085', per=60, sdate=start_day, edate=today)


total = pd.concat([ 
                   get_close(a_gu).rename('汇安多因子').astype('float'), 
                   get_close(qqq).rename('纳斯达克').astype('float'), 
                   get_close(au_etf).rename('黄金').astype('float'),
                   get_close(duanzhai).rename('短债').astype('float')], axis = 1)

total.sort_index(inplace=True)
total = total.fillna(method = 'ffill')
    
total.head()

    
i = total.shape[0]
data1 = total.copy()
col = data1.shape[1]



count = 1
risk_weight = calc_equal_risk_contributions_weights(data1.iloc[i-30:i, 0:col].pct_change().cov().values)
for period in range(40, 70, 2):
    count = count + 1
    risk_weight = risk_weight + calc_equal_risk_contributions_weights(data1.iloc[i-period:i, 0:col].pct_change().cov().values)
risk_parity_weight = risk_weight / count

risk_parity_weight

total_money * risk_parity_weight

# 计算组合年化波动率
count = 1
s_std = calc_portfolio_vol(risk_parity_weight, data1.iloc[i-30:i, 0:col].pct_change().cov().values)
for period in range(40, 70, 2):
    s_std += calc_portfolio_vol(risk_parity_weight, data1.iloc[i-period:i, 0:col].pct_change().cov().values)
    count += 1
# print(s_std)
s_std = s_std / count
year_vol = s_std * np.sqrt(250)
year_vol
if np.abs(year_vol) > 1e-6:
    zuhe_weight = target_year_vol / year_vol
if zuhe_weight > 1.0:
    zuhe_weight = 1.0

