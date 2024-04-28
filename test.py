from api.bd_api import *
import pretty_errors
pretty_errors.activate()

while True:
    input_text = input("请输入要分析的文本 (输入 'exit' 退出)：")
    if input_text.lower() == 'exit':
        print("感谢使用，再见！")
        break
    result = analyze_sentiment(input_text)
    if result:
        sentiment, polarity, subjectivity = result
        print(f"情感倾向: {sentiment}, 极性: {polarity}, 主观性: {subjectivity}")
    else:
        print("翻译失败，请稍后重试。")

