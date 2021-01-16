



---------------------------------------------------------------------------------
readme

һ��fund-rank.py

(1)��ȡ��һ��ʱ����ڣ��ض������������
(2)�����л����в�ѯ��һ��ʱ����ڣ�top50����������ߵĻ��𣬽���浽�ļ��С�

fund-rank.py usage:
        python fund.py start-date end-date fund-code=none

        date format ****-**-**
                start-date must before end-date
        fund-code default none
                if not input, get top 20 funds from all more than 6400 funds
                else get that fund's rate of rise

        eg:     python fund-rank.py 2017-03-01 2017-03-25
        eg:     python fund-rank.py 2017-03-01 2017-03-25 377240


����
(1)����ȡ�����б�
	��������ļ� fundlist-*.txt �ļ������ȡ���ļ�
	������ļ������� url��ȡ�б� Ȼ����ļ�
	
(2)��forѭ����ѯ����ֵ
	Ϊ�˼򻯴��� ��ѯ2�ξ�ֵ ֻ��ѯʱ��ο�ʼ�ͽ���2��ľ�ֵ
	
	�ۼƾ�ֵ����
	
	����ŵ�����λ�� ֻ�洢ǰ50������

����avg-rank.py
�Զ��top50����ļ����д�������ƽ���������������򣬽���浽�ļ��С�

����fund-zf.py
������������Ի�ȡ�������� ���Ի�ȡ�������� ���Ի�ȡ��������
MD ���ü��� ���ü��� ���ü��� ����2���ļ� ���Ű� ��ʾһ��
��ȡ��1�� ��3�� ��6�� ��12���µ���������ߵ�50������
����ƽ���������������򣬽���浽�ļ��С�


�ġ�����������Դ
��Ҫ���3�����ݣ����ݾ����������������
(1)�����б�
http://fund.eastmoney.com/js/fundcode_search.js
��ʽ��["000001","HXCZ","���ĳɳ�","�����","HUAXIACHENGZHANG"]

(2)����ֵ����
http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code=377240
http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code=160220&page=1
http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code=160220&page=1&per=50
http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code=377240&page=1&per=20&sdate=2017-03-01&edate=2017-03-01

��ʽ��var apidata={ content:"<table class='w782 comm lsjz'><thead><tr><th class='first'>��ֵ����</th><th>��λ��ֵ</th><th>�ۼƾ�ֵ</th><th>��������</th><th>�깺״̬</th><th>���״̬</th><th class='tor last'>�ֺ�����</th></tr></thead><tbody><tr><td>2017-03-01</td><td class='tor bold'>2.1090</td><td class='tor bold'>2.1090</td><td class='tor bold red'>0.29%</td><td>�����깺</td><td>�������</td><td class='red unbold'></td></tr></tbody></table>",records:1,pages:1,curpage:1};

��ʽ���Ժ�
��ֵ����	��λ��ֵ	�ۼƾ�ֵ	��������	�깺״̬	���״̬	�ֺ�����
2017-03-01	2.1090	2.1090			0.29%		�����깺	�������	

(3)������������
http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=gp&rs=&gs=0&sc=zzf&st=desc&sd=2016-03-29&ed=2017-03-29&qdii=&tabSubtype=,,,,,&pi=1&pn=50&dx=1&v=0.6370068000914493
ft�� fund type���� ����-all ��Ʊ��-gp �����-hh ծȯ��-zq ָ����-zs ������-bb QDII-qdii LOF-lof


����ɸѡ
http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=all&rs=3yzf,50&gs=0&sc=3yzf&st=desc&sd=2016-03-29&ed=2017-03-29&qdii=&tabSubtype=,,,,,&pi=1&pn=50&dx=1&v=0.013834315347261095
http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=all&rs=6yzf,20&gs=0&sc=6yzf&st=desc&sd=2016-03-29&ed=2017-03-29&qdii=&tabSubtype=,,,,,&pi=1&pn=50&dx=1&v=0.5992681832027366
http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=all&rs=1nzf,20&gs=0&sc=1nzf&st=desc&sd=2016-03-29&ed=2017-03-29&qdii=&tabSubtype=,,,,,&pi=1&pn=50&dx=1&v=0.6093838416906625

rs=3yzf,50 ��3���Ƿ�����ǰ50
rs=1nzf,20 ��1���Ƿ�����ǰ20

�塢�������

python fund.py 2016-01-21 2017-03-24
����2017.03.27������6400�������ȫ����һ�飬���˰�Сʱ���������ٲ�ͬ����ʱ�в��졣
����	����				����										����		2016-01-21	2017-03-24	������	������
1		502022	������֤50�ּ�B							�ּ��ܸ�		0.0118		0.4511			0.44		3728.81%
2		150296	�Ϸ���֤������ҵ�ĸ�ּ�B		�ּ��ܸ�		0.0290		0.4494			0.42		1448.28%
3		150294	�Ϸ���֤������ҵָ���ּ�B		�ּ��ܸ�		0.0404		0.5472			0.51		1262.38%
4		502008	�׷������ĸ�ָ���ּ�B			�ּ��ܸ�		0.0562		0.5280			0.47		836.3%
5		502015	��ʢ��֤����һ��һ·�ּ�B		�ּ��ܸ�		0.0510		0.3945			0.34		666.67%



python fund-zf.py
1	161725	������֤�׾�ָ���ּ�	��Ʊָ��	19	4	3	4	7.5
2	002230	���Ĵ��л����(QDII)	QDII	8	7	21	6	10.5
3	110022	�׷���������ҵ	��Ʊ��	30	11	10	9	15.0
4	002534	�����ȹ�����ծȯA	ծȯ��	100	1	2	3	26.5
5	160632	�����Ʒּ�	��Ʊָ��	100	26	11	11	37.0
6	180012	������ԣ������	�����	100	23	20	10	38.25
7	050015	��ʱ���л���̫��ѡ��Ʊ	QDII	100	9	27	20	39.0
8	000988	��ʵȫ��������Ʊ�����	QDII	25	21	100	24	42.5
9	050018	��ʱ��ҵ�ֶ����	�����	27	18	100	25	42.5
10	110011	�׷�����С�̻��	�����	38	19	100	18	43.75


---------------------------------------------------------------------------------
ChangeLog:
V1.0  2017.03.27
