# !/usr/bin/python 
# -*- coding: utf-8 -*-
import sys
import datetime
import urllib2
import sys
import json
import glob
	
# �Ƿ�����list��
def get_index(fund_code, all_fund_list):
	fund_num = len(all_fund_list)
	fund_index = 0
	while fund_index < fund_num:
		if fund_code in all_fund_list[fund_index]:
			break;
		fund_index += 1
	
	return fund_index

# ��ȡ��������
def get_type(fund_code, all_fund_list):
	fund_type = 'none'
	for fund in all_fund_list:
		if fund_code in fund:
			fund_type = fund[3]
			break
	
	return fund_type

def main():
	# ��ǰ����	
	strtoday = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
	tdatetime = datetime.datetime.strptime(strtoday, '%Y-%m-%d')
	print strtoday

	# ȥ�����
	sdatetime = tdatetime - datetime.timedelta(days=365)
	strsdate = datetime.datetime.strftime(sdatetime, '%Y-%m-%d')
	print strsdate
	
	# ��ȡ���л����б� ���ڲ�ѯ����
	all_fund = []
	fundlist_files = glob.glob('fundlist-*.txt')
	file_object = open(fundlist_files[0], 'r')
	try:
		all_funds_txt = file_object.read()
		#print all_funds_txt
	finally:
		file_object.close()
	
	all_funds_txt = all_funds_txt[all_funds_txt.find('=')+2:all_funds_txt.rfind(';')]
	all_fund = json.loads(all_funds_txt.decode('utf-8'))
	
	
	
	# 1�� ��ȡ�� 1 3 6 ������top50
	
	month_num = 1
	month_list = [1, 3, 6, 12]
	all_fund_list = []
	
	for int_month in month_list:
		try:
			if int_month == 12:
				# 1������
				print 'get nearly 1 year top 50 funds'
				url = 'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=all&rs=1nzf,50&gs=0&sc=1nzf&st=desc&sd=' + \
				strsdate + '&ed=' + strtoday + '&qdii=&tabSubtype=,,,,,&pi=1&pn=50&dx=1'
			else:
				# ǰ n ������
				print 'get nearly ' + str(int_month) + ' months top 50 funds'
				url = 'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=all&rs=' + str(int_month) + \
				'yzf,50&gs=0&sc=' + str(int_month) + 'yzf&st=desc&sd=' + strsdate + '&ed=' + strtoday + \
				'&qdii=&tabSubtype=,,,,,&pi=1&pn=50&dx=1'
			#print url + '\n'
			response = urllib2.urlopen(url)
		except urllib2.HTTPError, e:
			print e
			urllib_error_tag = True
		except StandardError, e:
			print e
			urllib_error_tag = True
		else:
			urllib_error_tag = False
		
		if urllib_error_tag == True:
			print 'error to get date,check network!\n'
			sys.exit(1)
	
		#print response	
		all_rank_txt = response.read().decode('utf-8')		
		all_rank_txt = all_rank_txt[all_rank_txt.find('["'):all_rank_txt.rfind('"]')+2]
		#print all_rank_txt
		# �������� NND utf-8 ���а�
		all_rank_list = json.loads(all_rank_txt)
		#print 'rank sum:' + str(len(all_rank_list)) + '\n\n'
		
		fund_rank = 1
		for rank_txt in all_rank_list:
			#print rank_txt + '\n'
			#    0          1        2         3        4    5     5   6     7      8      8     10   11 12  13
			#                                          ��λ �ۼ� ����  ��1                           ��2 ��3 ����
			# ������� ������ 	            ���� 	   ��ֵ ��ֵ ����  ��   ��1��  ��3��  ��6�� ��1��  �� �� �� ������ 	                           ������
			# 002425,��ӥ�������C,JYBBHHC,2017-03-28,1.1120,1.6120,0,0.09,-0.4477,0.6335,0.5082,61.5167,,,0.09,61.5167,2016-03-07,1,61.5167,1.00%,0.10%,1,0.10%,1,
			# 002441,�°�������ծȯC,DBXTLZQC,2017-03-28,1.1365,1.5165,0.0264,0.1409,0.4241,1.2923,1.1904,50.4788,,,1.1571,50.8371,2016-02-17,1,50.4788,,0.00%,,,,
			# ��25��
			rank_list = rank_txt.split(',')
			#print str(len(rank_list)) + '\n'	
			# ����ǵ�1���� ֱ��append
			if 1 == month_num:
				fund_list= []
				
				fund_type = get_type(rank_list[0], all_fund)
				
				fund_list.append(rank_list[0])
				fund_list.append(rank_list[1] + '\t' + fund_type)
				fund_list.append(str(fund_rank))
					
				all_fund_list.append(fund_list)
			else:
				# �����Ƿ�����list��
				fund_num = len(all_fund_list)
				fund_index = get_index(rank_list[0], all_fund_list)
				if fund_index < fund_num:
					# list���Ѵ��� ֻappend rank �� rate
					#print fund_code + '\t' + str(fund_index) + '\t' + str(all_fund_list[fund_index])
					all_fund_list[fund_index].append(str(fund_rank))
				else:
					# ��������� ������Ҫ�������list�� ͬ����Ҫ���������� rank �� rate ����
					# �����fundcode��ǰ�����ļ��в����� ���ǵ�����һ ��100���� rank Ĭ��100 rate Ĭ��0
					#print fund_code + '\tnot found!'
					fund_list= []
					# code name type
					fund_type = get_type(rank_list[0], all_fund)
					
					fund_list.append(rank_list[0])
					fund_list.append(rank_list[1] + '\t' + fund_type)
					# ���� ǰ�����ļ��� rank �� rate
					for i in range(month_num - 1):
						fund_list.append('100')
					# ���ϵ�ǰ��rank �� rate
					fund_list.append(str(fund_rank))
						
					#��������б�
					all_fund_list.append(fund_list)						
						
			# ������һ�� 
			fund_rank += 1
		
		# ���� �ӵ�2���¿�ʼ �����fund ���ڱ��ļ��г��� ���ǵ�����һ ��100���� ��Ҫ��rank �� rate ���� �� ͦ����
		if month_num > 1:
			# ���len Ϊ 2 + month_num
			fund_len = 2 + month_num
			fund_num = len(all_fund_list)
			fund_index = 0
			while fund_index < fund_num:
				if len(all_fund_list[fund_index]) < fund_len:
					all_fund_list[fund_index].append('100')
				
				fund_index +=1
		
		# ������һ����
		month_num += 1
	
	print month_num
	print all_fund_list[0]
	print all_fund_list[1]
	
	# 2������ƽ������	
	fund_num = len(all_fund_list)
	print fund_num
	print '\n\n'
	fund_index = 0
	while fund_index < fund_num:
		sum_rank = 0
		# �м����ļ� ���м������� rank��index 2 3 4 5
		for i in range(month_num - 1):
			sum_rank += int(all_fund_list[fund_index][2 + i])
			
		# ����ƽ������
		avg_rank = float('%.2f' %(float(sum_rank) / (month_num - 1)))
	
		# ����avg_rank
		all_fund_list[fund_index].append(avg_rank)
		
		# ������һ��
		fund_index +=1
		
	# 3������ д�ļ� ��ӡ
	# avg_rank��index 1 + month_num
	all_fund_list.sort(key=lambda fund: fund[1 + month_num])
	
	print all_fund_list[0]
	print all_fund_list[1]
	
	file_object = open('results-zf.txt', 'w')
	int_rank = 1
	try:
		for fund_list in all_fund_list:
			print str(int_rank) + '\t' + '\t'.join('{0}'.format(n) for n in fund_list)
			file_object.write(str(int_rank) + '\t' + '\t'.join('{0}'.format(n) for n in fund_list) + '\n')
			int_rank += 1
	finally:
			file_object.close()
	
	
	sys.exit(0)


def test():
	month_num = 1
	
			
	print month_num

if __name__ == "__main__":
	reload(sys)
	#sys.setdefaultencoding('utf-8')
	sys.setdefaultencoding('GBK')
	
	#test()
	main()

	
