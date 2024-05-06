from rich import traceback,print
traceback.install()

import pandas as pd
from api.tx_api import *
import json
from tqdm import tqdm
import asyncio
import sys
import time
from ratelimit import limits, sleep_and_retry
from ratelimit import limits, sleep_and_retry

# ####if试是否可用
if(1==0):
    print("")
    print("-----------------")
    dfs = ["我恨我自己！！", "我一般般", "我是最棒的！！！"]
    for df in dfs:
        response_json = tx_api(df)
        response_data = json.loads(response_json)

        sentiment = response_data['Response']['Sentiment']
        request_id = response_data['Response']['RequestId']

        print(df)
        print("这条语句的情感(Sentiment):", sentiment)
        print("Request ID:", request_id)
        print("") 
    print("-----------------")

ori_data = r"/Users/surui/Desktop/ori data(1) 2.xlsx"
data = pd.read_excel(ori_data).astype(str)
#不放心可以先试一下前20
#first_column = data.iloc[:20, 0]
first_column = data.iloc[:, 0]
second_column = data.iloc[:, 1]
#print(first_column)

def api(text):
    
    response_json = tx_api(text)
    response_data = json.loads(response_json)
    try:
        sentiment = response_data['Response']['Sentiment']
    except KeyError:
        sentiment = "Error"
        print(response_data)
    return sentiment


if(1==1):
    import concurrent.futures

    data1 = []

    @sleep_and_retry
    @limits(calls=10, period=1)
    def process_text(text):
        try:
            sentiment = api(text)
        except:
            sentiment = "Error"
        return sentiment

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_text, first_column)
        sys.setrecursionlimit(10**9)
        for result in results:
            data1.append(result)
            tqdm.write(f"Processed: {len(data1)} / {len(first_column)}")

    print(data1)

    df = pd.DataFrame({
        'Comment': first_column,
        'Sentiment': second_column,
        'Sentiment-API': data1
    })
    df.to_excel('output.xlsx', index=False)

    

