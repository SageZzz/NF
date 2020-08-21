#!/usr/bin/env python3
# coding: utf-8

import requests
import json

def tuling(msg):
    api_key = "71f0ff92dcd74f40a291ae1ac02ce816"
    url = 'http://openapi.tuling123.com/openapi/api/v2'
    data = {
        "perception": {
            "inputText": {
                "text": msg
            },
        },
        "userInfo": {
            "apiKey": api_key,
            "userId": "1"
        }
    }
    datas = json.dumps(data)
    html = requests.post(url, datas).json()
    if html['intent']['code'] == 40004:
        print("次数用完")
        return None
    return html['results']
msg = '一月二十二是什么星座'
print("原话>>", msg)
res = tuling(msg)
for resi in res:
    if('text' in resi['values']):
        print("图灵>>", resi['values']['text'])

    if('url' in resi['values']):
        print("链接>>", resi['values']['url'])
