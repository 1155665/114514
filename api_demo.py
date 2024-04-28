from api.tx_api import *
import json
import pretty_errors
pretty_errors.activate()


dfs = ["我恨我自己！！", "我一般般", "我是最棒的！！！"]
for df in dfs:
    response_json = tx_aip(df)
    response_data = json.loads(response_json)

    sentiment = response_data['Response']['Sentiment']
    request_id = response_data['Response']['RequestId']

    print(f"\033[1;32m{df}\033[0m")
    print("这条语句的情感(Sentiment):", sentiment)
    print("Request ID:", request_id)