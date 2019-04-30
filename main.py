import time
start_time = time.time()
##PREP ENV
import os
import pandas as pd
import numpy as np
import pickle

#SET LOCAL PATH IF IN SPYDER
if 'SPYDER_ARGS' in os.environ:
    local_path = '/Users/Ab/Google_Drive/MBA/Y2-2019_q3/Q3Q4_DIS/registrar_model/'
    os.chdir(local_path)
else:
    local_path = ''

from build_models import *


#3. Build predictions
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


#TEST DATA
#TODO - Makes test data a JSON object
model_df = pd.DataFrame({'course_number': y.columns, 'prob': build_probs(X.iloc[0:1, :], local_path)})


#######################
##FOLD IN ERIC'S WORK##
#######################
json_out = run_erics_opt(model_df)
print('Run time of ' + str(round((time.time() - start_time)/60,2)) + ' minutes')
print(json_out)