# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 22:45:33 2021

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

# 读取所有基金数据
with open(r'all_fund_2021-01-15.json', 'r', encoding='utf-8') as fp:
    all_fund_list = json.load(fp)


all_fund_frame = pd.DataFrame(all_fund_list)
all_fund_frame.columns = ['code', 'simple_name', 'chinese_name', 'type', 'full_name']

all_fund_frame.head()

all_fund_frame.groupby(by='type').count()
    
    
# 抓取网页
def get_url(url, params=None, proxies=None):
    rsp = requests.get(url, params=params, proxies=proxies)
    rsp.raise_for_status()
    return rsp.text

# 从网页抓取数据
def get_fund_data(code,per=10,sdate='',edate='',proxies=None):
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

hs300_460300 = get_fund_data('460300',per=60,sdate='2015-01-01',edate='2020-01-01')
hs300_460300.to_csv('data/460300.csv', index = 0)

all_fund_frame[all_fund_frame.code == '460300']


guozhai = pd.read_csv('data/000012.csv', encoding='gbk')
hs_300 = pd.read_csv('data/000300.csv', encoding='gbk')

guozhai

hs_300

guozhai.set_index('日期', inplace=True)
guozhai.sort_index(inplace=True)

hs_300.set_index('日期', inplace=True)
hs_300.sort_index(inplace=True)

total = pd.concat([guozhai.收盘价.rename('国债'), hs_300.收盘价.rename('沪深300')], axis = 1)

total.sort_index(inplace=True)

total_2015 = total[total.index >= '2015-01-01']
total_2015

total_2015


total_2015.head()

jinzhi = np.array([])
weight = np.array([0.5, 0.5])

for i, row in enumerate(total_2015.values):
    if i == 0:
        # 第一天净值为1
        jinzhi = np.append(jinzhi, 1)
    else:
        yesterday_md = total_2015.values[i-1, ]
        today_md = total_2015.values[i, ]
        ret = today_md / yesterday_md
        
        # 计算仓位
        weight = weight * ret
        # 计算净值
        today_jinzhi = np.dot(weight, ret)
        jinzhi = np.append(jinzhi, today_jinzhi)
    
    # 添加换仓逻辑
    
jinzhi

plt.plot(jinzhi)

