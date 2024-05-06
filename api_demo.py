from rich import traceback,print
traceback.install()

import pandas as pd
from api.tx_api import *
import json
from tqdm import tqdm
import asyncio

# #### 测试是否可用
'''
print("")
print("-----------------")
dfs = ["我恨我自己！！", "我一般般", "我是最棒的！！！"]
for df in dfs:
    response_json = tx_aip(df)
    response_data = json.loads(response_json)

    sentiment = response_data['Response']['Sentiment']
    request_id = response_data['Response']['RequestId']

    print(df)
    print("这条语句的情感(Sentiment):", sentiment)
    print("Request ID:", request_id)
    print("") 
print("-----------------")
'''
ori_data = r"/Users/surui/Desktop/ori data(1) 2.xlsx"
data = pd.read_excel(ori_data).astype(str)
#不放心可以先试一下前20
#first_column = data.iloc[:20, 0]
first_column = data.iloc[:, 0]
second_column = data.iloc[:, 1]
#print(first_column)

def api(text):
    response_json = tx_aip(text)
    response_data = json.loads(response_json)
    try:
        sentiment = response_data['Response']['Sentiment']
    except KeyError:
        sentiment = None
    return sentiment



if(1==1):
    #data1 = list(tqdm(first_column.apply(api)))
    data1 = []
    lst=first_column.tolist()
    #print(lst)
    
    async def process_data(lst):
        data1 = []
        for text in tqdm(lst):
            #sentiment = 
            data1.append(await asyncio.to_thread(api, text))
        return data1

    async def main():
        lst = first_column
        data1 = await process_data(lst)
        
        df = pd.DataFrame({
            'Comment': first_column,
            'Sentiment': second_column,
            'Sentiment-API': data1
        })
        df.to_excel('output.xlsx', index=False)
    #main()
    asyncio.run(main())

    print(data1)
    
    df = pd.DataFrame({
        'Comment': first_column,
        'Sentiment': second_column,
        'Sentiment-API': data1
    })

    df.to_excel('output.xlsx', index=False)
    
    

