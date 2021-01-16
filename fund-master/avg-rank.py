# !/usr/bin/python 
# -*- coding: utf-8 -*-
import glob
import sys
	
# �Ƿ�����list��
def get_index(fund_code, all_fund_list):
	fund_num = len(all_fund_list)
	fund_index = 0
	while fund_index < fund_num:
		if fund_code in all_fund_list[fund_index]:
			break;
		fund_index += 1
	
	return fund_index

def main():
		
	# 1����ȡ���еĽ���ļ�
	result_files = glob.glob('result_*.txt')
	if (len(result_files) == 0) :
		print 'no result file to process!'
		sys.exit(1)
	
	# ��������ŵ�һ��list�� �����ĸ�fund ƽ���������
	# code name type rank1 rate1 rank2 rate2 ... rankn raten average_rank
	
	file_num = 0
	all_fund_list = []
	
	# ѭ�������ļ�
	for filename in result_files:
		print 'process file:\t' + str(file_num) + '\t' + filename
		file_object = open(filename, 'r')
		try:
			# ��1�� ��ͷ������ �ӵ�2�п�ʼ����
			file_object.readline()
			while 1:
				funds_txt = file_object.readline()
				if not funds_txt:
					break;
					
				# ת��list
				fund = funds_txt.split()
				
				# ����ǵ�1���ļ� ֱ��append
				if 0 == file_num:
					fund_list= []
					fund_list.append(fund[1])
					fund_list.append(fund[2])
					fund_list.append(fund[3])
					fund_list.append(fund[0])
					fund_list.append(fund[7])
					
					all_fund_list.append(fund_list)
				else:
					# �����Ƿ�����list��
					fund_num = len(all_fund_list)
					fund_index = get_index(fund[1], all_fund_list)
					if fund_index < fund_num:
						# list���Ѵ��� ֻappend rank �� rate
						#print fund_code + '\t' + str(fund_index) + '\t' + str(all_fund_list[fund_index])
						all_fund_list[fund_index].append(fund[0])
						all_fund_list[fund_index].append(fund[7])
					else:
						# ��������� ������Ҫ�������list�� ͬ����Ҫ���������� rank �� rate ����
						# �����fundcode��ǰ�����ļ��в����� ���ǵ�����һ ��100���� rank Ĭ��100 rate Ĭ��0
						#print fund_code + '\tnot found!'
						fund_list= []
						# code name type
						fund_list.append(fund[1])
						fund_list.append(fund[2])
						fund_list.append(fund[3])
						# ���� ǰ�����ļ��� rank �� rate
						for i in range(file_num):
							fund_list.append('100')
							fund_list.append('0')
						# ���ϵ�ǰ��rank �� rate
						fund_list.append(fund[0])
						fund_list.append(fund[7])
						
						#��������б�
						all_fund_list.append(fund_list)						
						
		finally:
			file_object.close()
		
		# ���� �ӵ�2���ļ���ʼ �����fund ���ڱ��ļ��г��� ���ǵ�����һ ��100���� ��Ҫ��rank �� rate ���� �� ͦ����
		if file_num > 0:
			# ���len Ϊ 5 + 2 * file_num
			fund_len = 5 + 2 * file_num
			fund_num = len(all_fund_list)
			fund_index = 0
			while fund_index < fund_num:
				if len(all_fund_list[fund_index]) < fund_len:
					all_fund_list[fund_index].append('100')
					all_fund_list[fund_index].append('0')
				
				fund_index +=1
		
		# ������һ���ļ�
		file_num += 1
	
	'''
	print 'filenum:' + str(file_num)
	print len(all_fund_list)
	for fund_list in all_fund_list:
		print fund_list
	'''
	
	# 2������ƽ������
	fund_num = len(all_fund_list)
	fund_index = 0
	while fund_index < fund_num:
		sum_rank = 0
		# �м����ļ� ���м������� rank��index 3 5 7 ... ...
		for i in range(file_num):
			sum_rank += int(all_fund_list[fund_index][3 + 2 * i])
			
		# ����ƽ������
		avg_rank = float('%.2f' %(float(sum_rank) / file_num))
	
		# ����avg_rank
		all_fund_list[fund_index].append(avg_rank)
		
		# ������һ��
		fund_index +=1
		
	# 3������ д�ļ� ��ӡ
	# avg_rank��index 5 + 2 * file_num
	all_fund_list.sort(key=lambda fund: fund[3 + 2 * file_num])
	
	file_object = open('results.txt', 'w')
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
	all_fund_list = []
	
	file_object = open('test.txt', 'r')
	try:
		file_object.readline()
		while 1:
			funds_txt = file_object.readline()
			if not funds_txt:
				break;
				
			fund = funds_txt.split()
			#print fund[2] + '\t' + fund[3]
			fund_list= []
			fund_list.append(fund[1])
			fund_list.append(fund[2])
			fund_list.append(fund[3])
			fund_list.append(fund[0])
			fund_list.append(fund[7])
			
			all_fund_list.append(fund_list)
	finally:
		file_object.close()

	fund_code_list = ['150294', '002534', '0004', '502008', '000114']
	fund_num = len(all_fund_list)
	print fund_num
	for fund_code in fund_code_list:
		fund_index = get_index(fund_code, all_fund_list)
		if fund_index < fund_num:
			print fund_code + '\t' + str(fund_index) + '\t' + str(all_fund_list[fund_index])
		else:
			print fund_code + '\tnot found!'
		
	
	print '\n\n\n\n'
	#print all_fund_list.index('150294')

if __name__ == "__main__":	
	#test()
	main()

	
