##PREP ENV
import os
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pulp as plp

#SET LOCAL PATH IF IN SPYDER
if 'SPYDER_ARGS' in os.environ:
    local_path = '/Users/Ab/Google_Drive/MBA/Y2-2019_q3/Q3Q4_DIS/registrar_model/'
    os.chdir(local_path)
else:
    local_path = ''

#TODO
#1. Read in and one-hot data
courses_by_student_dat = pd.read_csv(local_path + 'data/courses_by_student7.csv')

#y: LE Course number
#le = preprocessing.LabelEncoder()
#le.fit(courses_by_student_dat.course_number.astype(str))
#y = le.transform(courses_by_student_dat.course_number.astype(str))
y = pd.get_dummies(courses_by_student_dat['course_number'].astype(str))

#X:
int_list = ['bachelors_grad_year', 'age']
oh_list = ['bachelors_major_category', 'masters_degree', 'citizenship_country']
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
        elif column == 'citizenship_country':
            df.loc[df['citizenship_country'] != 'USA', 'citizenship_country'] = 'non-USA'
    df = pd.get_dummies(df, columns=oh_list)
    return(df)

X = preprocess_X(X_raw, int_list, oh_list, err_list)


#2. Build model --> Probs
    #?What is the probabilty that a given student will take a particular course?

#Iterate through columns and build pickle [jar?]
def build_pickle_jar(y, local_path):
    for column in y:
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X, y[column])
        if os.path.exists(local_path + 'models/' + str(column) + '.pkl'):
            print('File exists: ' + str(column) + '.pkl')
        model_pkl_path = open(local_path + 'models/' + str(column) + '.pkl', 'wb')
        pickle.dump(clf, model_pkl_path)
        model_pkl_path.close()

#build_pickle_jar(y, local_path)


#Roll in Erics code
def run_erics_opt(model_df):
    opt_model = plp.LpProblem(name="MIP Model")
    
    #Read in classes list
    active_classes_df = pd.read_csv(local_path + 'data/SY_2018_2019_Classes.csv')
    active_classes_model_no_dummies_df = pd.merge(active_classes_df, model_df,
                                                  left_on='CourseNumber', right_on='course_number', how='left')
    #Adjust prob based on number of credits
    active_classes_model_no_dummies_df['adj_credit_prob'] = (active_classes_model_no_dummies_df['Credit'] /1.5) * active_classes_model_no_dummies_df['prob']
    #Dummy start times
    dummy_df = pd.get_dummies(active_classes_model_no_dummies_df[['Quarter','EarlyLate','StartTime']], prefix="d_")
    #Aggregate quarter dummies
    dummy_df['d__Qtr1_affected'] = dummy_df['d__Qtr1'] + dummy_df['d__Qtr1,Qtr2']
    dummy_df['d__Qtr2_affected'] = dummy_df['d__Qtr2'] + dummy_df['d__Qtr1,Qtr2']
    dummy_df['d__Qtr3_affected'] = dummy_df['d__Qtr3'] + dummy_df['d__Qtr3,Qtr4']
    dummy_df['d__Qtr4_affected'] = dummy_df['d__Qtr4'] + dummy_df['d__Qtr3,Qtr4']
    #Aggregate early vs late week dummies
    dummy_df['d__EarlyWeek_affected'] = dummy_df['d__EarlyWeek'] + dummy_df['d__MondayOnly'] + dummy_df['d__TuesdayOnly']
    dummy_df['d__LateWeek_affected'] = dummy_df['d__LateWeek'] + dummy_df['d__WednesdayOnly'] + dummy_df['d__ThursdaysOnly']

    active_classes_model_df = pd.concat([active_classes_model_no_dummies_df, dummy_df], axis=1)
    courseid_dummy_df = pd.get_dummies(active_classes_model_no_dummies_df[['CourseNumber']], prefix="d_")
    area_dummy_df = pd.get_dummies(active_classes_model_no_dummies_df[['Area']], prefix="d_")
    prereq_dummy_df = pd.get_dummies(active_classes_model_no_dummies_df[['Prereq']], prefix="d_")
    
    # Create dummy variables for each unique timeslot (quarter, early/late wk, time), remove dummy variable columns with 'none' values
    active_classes_model_df['uniqueTimeSlot'] = dummy_df['d__Qtr1_affected'].map(str) + dummy_df['d__Qtr2_affected'].map(str) + dummy_df['d__Qtr3_affected'].map(str) + dummy_df['d__Qtr4_affected'].map(str) + dummy_df['d__EarlyWeek_affected'].map(str) + dummy_df['d__LateWeek_affected'].map(str) + active_classes_model_df['StartTime']
    uniqueTimeSlot_dummy_df = pd.get_dummies(active_classes_model_df[['uniqueTimeSlot']], prefix="d_")
    uniqueTimeSlot_dummy_df = uniqueTimeSlot_dummy_df[uniqueTimeSlot_dummy_df.columns.drop(list(uniqueTimeSlot_dummy_df.filter(regex='None')))]

    #Build optimization model
    register_binary = plp.LpVariable.dicts("register_binary",((i) for i in active_classes_model_df.index), cat='Binary')
    model = plp.LpProblem("MaxProb", plp.LpMaximize)
    model += plp.lpSum([register_binary[a] * active_classes_model_df['adj_credit_prob'][a] for a in active_classes_model_df.index])
    
    ##Add constraints
    #Max 30 credits in SY
    model += sum([register_binary[b] * active_classes_model_df['Credit'][b] for b in active_classes_model_df.index]) <= 30
    #Max 1 for EQ4
    model += sum([register_binary[x] * active_classes_model_df['d__EQtr4'][x] for x in active_classes_model_df.index]) <= 1
    #Max 1 course for J-term
    model += sum([register_binary[x] * active_classes_model_df['d__J-Term'][x] for x in active_classes_model_df.index]) <= 1
    #Min 4.5 credits, Max 9 credits each quarter
    model += sum([register_binary[x] * active_classes_model_df['d__Qtr1'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index]) + (sum([register_binary[x] * active_classes_model_df['d__Qtr1,Qtr2'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index])/2) >= 4.5
    model += sum([register_binary[x] * active_classes_model_df['d__Qtr1'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index]) + (sum([register_binary[x] * active_classes_model_df['d__Qtr1,Qtr2'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index])/2) <= 9
    model += sum([register_binary[x] * active_classes_model_df['d__Qtr2'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index]) + (sum([register_binary[x] * active_classes_model_df['d__Qtr1,Qtr2'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index])/2) >= 4.5
    model += sum([register_binary[x] * active_classes_model_df['d__Qtr2'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index]) + (sum([register_binary[x] * active_classes_model_df['d__Qtr1,Qtr2'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index])/2) <= 9
    model += sum([register_binary[x] * active_classes_model_df['d__Qtr3'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index]) + (sum([register_binary[x] * active_classes_model_df['d__Qtr3,Qtr4'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index])/2) >= 4.5
    model += sum([register_binary[x] * active_classes_model_df['d__Qtr3'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index]) + (sum([register_binary[x] * active_classes_model_df['d__Qtr3,Qtr4'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index])/2) <= 9
    model += sum([register_binary[x] * active_classes_model_df['d__Qtr4'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index]) + (sum([register_binary[x] * active_classes_model_df['d__Qtr3,Qtr4'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index])/2) >= 4.5
    model += sum([register_binary[x] * active_classes_model_df['d__Qtr4'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index]) + (sum([register_binary[x] * active_classes_model_df['d__Qtr3,Qtr4'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index])/2) <= 9
    #No duplicate courses
    for column in courseid_dummy_df: model += sum([register_binary[x] * courseid_dummy_df[column][x] for x in active_classes_model_df.index]) <= 1
    #Min 1.5 credit of LDSP
    model += sum([register_binary[x] * area_dummy_df['d__LDSP'][x] * active_classes_model_df['Credit'][x] for x in active_classes_model_df.index]) >= 1.5
    #Bid constraint: sum of min bid amount <= 3000. Did not have bid info for new classes, assumed average of min bid amounts of all courses.
    model += sum([register_binary[x] * active_classes_model_df['MinBid'][x] for x in active_classes_model_df.index]) <= 3000
    #For required pre-req courses. Not all prereq courses included as we do not have complete list
    for column in prereq_dummy_df.columns: 
            try: model += sum([register_binary[x] * prereq_dummy_df[column][x] for x in active_classes_model_df.index]) <=  sum([register_binary[y] * courseid_dummy_df[column][y] for y in active_classes_model_df.index])
            except: 0
    
    ### Constraints: can select only one course per timeslot
    for column in uniqueTimeSlot_dummy_df.columns: 
        model += sum([register_binary[x] * uniqueTimeSlot_dummy_df[column][x] for x in active_classes_model_df.index]) <= 1

    #Solve and export solution
    model.solve()
    plp.LpStatus[model.status]
    
    output = []
    for c in active_classes_model_df.index:
        var_output = {
            'CourseNumber':active_classes_model_df['CourseNumber'][c],
            'Title':active_classes_model_df['Title'][c],
            'Qtr':active_classes_model_df['Quarter'][c],
            'Week':active_classes_model_df['EarlyLate'][c],
            'Time':active_classes_model_df['StartTime'][c],
            'RegisterClassBinary': register_binary[c].varValue,
            'Credit':active_classes_model_df['Credit'][c],
            'ProbAdjForCredits':active_classes_model_df['adj_credit_prob'][c]
        }
        output.append(var_output)
    output_df = pd.DataFrame.from_records(output)
    output_df_reg_classes = output_df[output_df['RegisterClassBinary'] == 1]
    output_df_reg_classes.sort_values(by=['Qtr', 'Week','Time'])
    
    return(output)