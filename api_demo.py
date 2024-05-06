from rich import traceback,print
traceback.install()

import pandas as pd
from api.tx_api import *
import json
from tqdm import tqdm
#import asyncio #失败，好心人帮看看
import sys
#import time
from ratelimit import limits, sleep_and_retry
import time

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
    text = text.strip()  # Remove leading and trailing whitespace
    text = text.lower()  # Convert text to lowercase
    text = text.replace(" ", "")  # Remove all spaces from text
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    if not text:
        return "ERROR 文本为空！"
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
        sys.setrecursionlimit(10**9)#这个才有用，不然会报错，浪费了5000多条
        timeout = 100000
        start_time = time.time()
        # Iterate over results 这个枚举有时间限制，上面那个timeout是骗他的，是纯忽悠人的
        for result in results:
            data1.append(result)
            tqdm.write(f"Processed: {len(data1)} / {len(first_column)}")


            # Check if timeout has been reached
            if time.time() - start_time >= timeout:
                break
    print(data1)

    df = pd.DataFrame({
        'Comment': first_column,
        'Sentiment': second_column,
        'Sentiment-API': data1
    })
    df.to_excel('output.xlsx', index=False)

    

