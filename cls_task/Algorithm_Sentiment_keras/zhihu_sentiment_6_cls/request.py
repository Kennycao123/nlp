import requests
import json
import pprint
import time

payload = {"text": '公司还不错，比较人性化，但是关键要运气好'}
# payload = {"text": '加班太严重！！！'}
payload = json.dumps(payload)
headers = {'Content-type': 'application/json'}
t1 = time.time()
x = requests.post('http://127.0.0.1:20001/sentiment', data=payload,headers = {'Content-type': 'application/json'})
# x = requests.post('http://10.8.204.89:20001/sentiment', data=payload,headers = {'Content-type': 'application/json'})
json_data = json.loads(x.text)  # json.loads()把Json格式字符串解码转换成Python对象
pprint.pprint(json_data)
t2 = time.time()
print('time_cost:%.6f'%(t2-t1))