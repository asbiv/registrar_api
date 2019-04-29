##PREP ENV
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

#Stuff for spyder
local_path = '/Users/Ab/Google_Drive/MBA/Y2-2019_q3/Q3Q4_DIS/registrar_model/'

#TODO
#1. Read in and one-hot data
courses_by_student_dat = pd.read_csv(local_path + 'data/courses_by_student7.csv')

#y: LE Course number
#le = preprocessing.LabelEncoder()
#le.fit(courses_by_student_dat.course_number.astype(str))
#y = le.transform(courses_by_student_dat.course_number.astype(str))
y = pd.get_dummies(courses_by_student_dat['course_number'])

#X:
int_list = ['bachelors_grad_year', 'age']
oh_list = ['grade', 'bachelors_major_category', 'masters_degree', 'citizenship_country']
err_list = ['consulting_job', 'finance_job', 'marketing_job', 'entrepreneurship_job',
            'non_profit_job', 'tech_job', 'corporate_job',
            'consulting_intern', 'finance_intern', 'marketing_intern', 'entrepreneurship_intern',
            'non_profit_intern', 'tech_intern', 'corporate_intern',
            'asset_management_sales_trading', 'business_analytics', 'business_development_growth',
            'b2b_marketing', 'corporate_finance_investment_banking', 'corporate_innovation', 'entrepreneurship',
            'global_business', 'innovation_for_sustainability', 'market_analytics', 'marketing',
            'strategy_consulting', 'supply_chain_management']
X_raw = courses_by_student_dat.copy()[int_list + oh_list + err_list]

#Helper function to replace values
def replace_with_one(x):
    if x != '1' and x != '0':
        return('1')
    else:
        return(x)

def replace_with_zero(x):
    try:
        int(x)
        return(x)
    except ValueError:
        return('0')

#Change column types
def preprocess_X(df, int_list, oh_list, err_list):
    for column in df:
        if column in int_list:
            #Replace missings with zero, then coerce to int, then replace zeros with mean
            df[column] = df[column].map(lambda x: replace_with_zero(x)).astype(int)
            col_mean = int(df[column].mean())
            df[column][df[column] == 0] = col_mean
        elif column in err_list:
            df[column] = df[column].map(lambda x: replace_with_one(replace_with_zero(x))).astype(int)
    df = pd.get_dummies(df, columns=oh_list)
    return(df)

X = preprocess_X(X_raw, int_list, oh_list, err_list)


#2. Build model --> Probs
    #?What is the probabilty that a given student will take a particular course?

#Iterate through columns of y
tmp = y.iloc[:,0]
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, tmp)
clf.predict(X.iloc[0:1, :])
clf.predict_proba(X.iloc[0:1, :]) #Prob 0, 1; Check with clf.classes_

#Iterate through columns
models = []
for column in y:
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, column)
    models += clf


#3. Test prediction