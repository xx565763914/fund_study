#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 18:57:17 2021

@author: xingxing
"""

import numpy as np

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

