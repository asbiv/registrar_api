from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import pandas as pd
import json

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
        #user_query = args['query']
        
        #Sanitize inputs
        eric_input = json.loads(args['query'])
        new_query = {"bachelors_grad_year": 0, "age": 0, "consulting_job": 0, "finance_job": 0, "marketing_job": 0, "entrepreneurship_job": 0, "non_profit_job": 0, "tech_job": 0, "corporate_job": 0, "consulting_intern": 0, "finance_intern": 0, "marketing_intern": 0, "entrepreneurship_intern": 0, "non_profit_intern": 0, "tech_intern": 0, "corporate_intern": 0, "asset_management_sales_trading": 0, "business_analytics": 0, "business_development_growth": 0, "b2b_marketing": 0, "corporate_finance_investment_banking": 0, "corporate_innovation": 0, "entrepreneurship": 0, "global_business": 0, "innovation_for_sustainability": 0, "market_analytics": 0, "marketing": 0, "strategy_consulting": 0, "supply_chain_management": 0, "bachelors_major_category_Accounting": 0, "bachelors_major_category_Business Administration": 0, "bachelors_major_category_Chemical Engineering": 0, "bachelors_major_category_Civil Engineering": 0, "bachelors_major_category_Computer Science": 0, "bachelors_major_category_Economics": 0, "bachelors_major_category_Electrical Engineering": 0, "bachelors_major_category_Engineering": 0, "bachelors_major_category_English": 0, "bachelors_major_category_Finance": 0, "bachelors_major_category_History": 0, "bachelors_major_category_Industrial Engineering": 0, "bachelors_major_category_International Relations": 0, "bachelors_major_category_Law": 0, "bachelors_major_category_Management": 0, "bachelors_major_category_Marketing": 0, "bachelors_major_category_Mathematics": 0, "bachelors_major_category_Mechanical Engineering": 0, "bachelors_major_category_Other": 0, "bachelors_major_category_Political Science": 0, "masters_degree_0.0": 0, "masters_degree_1.0": 0, "citizenship_country_USA": 0, "citizenship_country_non-USA": 0}
        for key in eric_input:
            if key == 'masters_degree':
                if eric_input[key] == 'Yes':
                    new_query['masters_degree_1.0'] = 1
                else:
                    new_query['masters_degree_0.0'] = 1
            elif key == 'citizenship_country':
                if eric_input[key] == 'US':
                    new_query['citizenship_country_USA'] = 1
                else:
                    new_query['citizenship_country_non-USA'] = 1
            else:
                new_query[key] = int(eric_input[key])
        print('new query:' + str(new_query))
        
        query_df = pd.DataFrame.from_dict(new_query, orient='index').transpose()
        #query_df = pd.DataFrame(pd.read_json(new_query, typ='series')).transpose()
        print('query df shape:' + str(query_df.shape))
        
        model_df = pd.DataFrame({'course_number': y.columns, 'prob': build_probs(query_df, local_path)})

        json_out = run_erics_opt(model_df)

        return json_out


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(build_calendar, '/')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)