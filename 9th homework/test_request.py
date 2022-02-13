# Module for testing service.

import json
from urllib import request


def get_prediction(url, comment_text):
    json_data = json.dumps({'comment_text': comment_text})
    json_bytes = json_data.encode('utf-8')

    request_prediction = request.Request(url)
    request_prediction.add_header('Content-Type', 'application/json; charset=utf-8')
    request_prediction.add_header('Content-Length', str(len(json_bytes)))

    response = request.urlopen(request_prediction, json_bytes)
    return json.load(response)


# model_url = r'http://127.0.0.1:5000//predict'
model_url = r'http://0.0.0.0:8180//predict'

print(get_prediction(model_url, r'test'))
