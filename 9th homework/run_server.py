# Main module for starting service.

# $env:FLASK_APP = "run_server"
# flask run
# http://127.0.0.1:5000//predict


import flask
from flask import Flask

import pandas as pd
import dill
import json
import os


PATH_MODEL = r'/app/model.dill'


def make_prediction(text):
    x_test = pd.DataFrame({'comment_text': text}, index=[0])

    with open(PATH_MODEL, mode='rb') as file:
        dict_model = dill.load(file)

    y_pred = dict_model['model'].predict(x_test)

    return y_pred[0]


flask_app = Flask(__name__)


@flask_app.route(r'/predict', methods=['POST'])
def predict():
    try:
        dict_json = flask.request.get_json()

        return flask.jsonify(
            json.dumps({
                'success': 'True',
                'prediction': str(make_prediction(dict_json.get('comment_text')))
            })
        )

    except Exception as exc:
        return flask.jsonify(
            json.dumps({
                'success': 'False',
                'error': str(exc)
            })
        )


port = int(os.environ.get('PORT', 8180))
flask_app.run(host='0.0.0.0', debug=True, port=port)
