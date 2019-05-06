from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import pandas as pd

from build_models import *

app = Flask(__name__)
api = Api(app)


def build_probs(Xi, local_path):
    preds = []
    pickle_dir = os.listdir(local_path + 'models/')
    for file in pickle_dir:
        if file == '.DS_Store':
            pass
        else:
            model = pickle.load(open(local_path + 'models/' + file, 'rb'))
            preds.append(model.predict_proba(Xi)[0][1])
    return(preds)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class build_calendar(Resource):
    def get(self):
        args = parser.parse_args()
        user_query = args['query']
        #print('User query:' + str(user_query))
        #query_json = pd.read_json(user_query, typ='series')
        #query_df = pd.DataFrame(query_json).transpose()
        #print('query df shape:' + str(query_df.shape))
        
        #model_df = pd.DataFrame({'course_number': y.columns, 'prob': build_probs(query_df, local_path)})

        #json_out = run_erics_opt(model_df)

        #return json_out
        return user_query


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(build_calendar, '/')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)