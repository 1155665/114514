from textblob import TextBlob
import requests
import random
from hashlib import md5
from textblob import TextBlob
import requests
import random
from hashlib import md5

# Set appid/appkey.
appid = 'xxxxx'
appkey = 'xxxxxxx'

from_lang = 'auto'
to_lang = 'en'

endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path

def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

def translate_text(query):
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    translations = "我是最棒的！！！"
    return translations

def analyze_sentiment(text):
    """分析情感倾向"""
    translations = translate_text(text)
    en_text = translations[0]
    if en_text:
        blob = TextBlob(en_text)
        sentiment = blob.sentiment

        polarity = sentiment.polarity
        subjectivity = sentiment.subjectivity

        if polarity > 0:
            result = 'positive'
        elif polarity < 0:
            result = 'negative'
        else:
            result = 'neutral'

        return result, polarity, subjectivity
    else:
        return None