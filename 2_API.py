from flask import Flask, request, jsonify
import json
import pandas as pd
import numpy as np
import pickle
import warnings
import re
import os

app = Flask(__name__)


def load_data():

    # Display the loading messages and update the progress bar
    print("Loading historical data... --- Progress : [..........] 0%")
    bureau_agg = pd.read_csv(r'./hist_data/bureau_agg2.csv')
    
    print("Loading historical data... --- Progress : [==........] 20%")
    prev_agg = pd.read_csv(r'./hist_data/prev_agg2.csv')
    
    print("Loading historical data... --- Progress : [====......] 40%")
    pos_agg = pd.read_csv(r'./hist_data/pos_agg2.csv')
    
    print("Loading historical data... --- Progress : [======....] 60%")
    ins_agg = pd.read_csv(r'./hist_data/ins_agg2.csv')
    
    print("Loading historical data... --- Progress : [========..] 80%")
    cc_agg = pd.read_csv(r'./hist_data/cc_agg2.csv')
    
    print("Loading historical data... --- Progress : [==========] 100% --- All done !")
    
    return bureau_agg, prev_agg, pos_agg, ins_agg, cc_agg

def load_models():
    with open('xgb_clf.pkl', 'rb') as file:
        xgb_clf = pickle.load(file)

    with open('optimal_threshold.pkl', 'rb') as file:
        optimal_threshold = pickle.load(file)
    
    return xgb_clf, optimal_threshold

# Load data and models
bureau_agg, prev_agg, pos_agg, ins_agg, cc_agg = load_data()
xgb_clf, optimal_threshold = load_models()


nan_as_category = True 

    
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def application(df):
    # Cleaning 
    df = df[df['CODE_GENDER'] != 'XNA'] # keep only relevant gender codes
    df = df[df['AMT_INCOME_TOTAL'] < 20000000] # remove a outlier 117M

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True) # set null value
    df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True) # set null value


    #Binary features encoding
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # One-hot encoding of categorical features
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # Flag_document features - count and kurtosis
    docs = [f for f in df.columns if 'FLAG_DOC' in f]
    df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)
    df['NEW_DOC_KURT'] = df[docs].kurtosis(axis=1)

    # defining age groups 
    def get_age_label(days_birth):
        """ Return the age group label (int). """
        age_years = -days_birth / 365
        if age_years < 27: return 1
        elif age_years < 40: return 2
        elif age_years < 50: return 3
        elif age_years < 65: return 4
        elif age_years < 99: return 5
        else: return 0
    # Categorical age - based on target=1 plot
    df['AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: get_age_label(x))

    # New features based on External sources
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

    df['EXT_SOURCES_WEIGHTED'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 1 + df.EXT_SOURCE_3 * 3

    for function_name in ['min', 'max', 'mean', 'nanmedian', 'var']:
        feature_name = 'EXT_SOURCES_{}'.format(function_name.upper())
        df[feature_name] = eval('np.{}'.format(function_name))(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

    # New percentages features
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # Credit ratios
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']

    # Income ratios
    df['INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
    df['INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']

    # Time ratios
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']

    # EXT_SOURCE_X FEATURE
    df['APPS_EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['APPS_EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['APPS_EXT_SOURCE_STD'] = df['APPS_EXT_SOURCE_STD'].fillna(df['APPS_EXT_SOURCE_STD'].mean())
    df['APP_SCORE1_TO_BIRTH_RATIO'] = df['EXT_SOURCE_1'] / (df['DAYS_BIRTH'] / 365.25)
    df['APP_SCORE2_TO_BIRTH_RATIO'] = df['EXT_SOURCE_2'] / (df['DAYS_BIRTH'] / 365.25)
    df['APP_SCORE3_TO_BIRTH_RATIO'] = df['EXT_SOURCE_3'] / (df['DAYS_BIRTH'] / 365.25)
    df['APP_SCORE1_TO_EMPLOY_RATIO'] = df['EXT_SOURCE_1'] / (df['DAYS_EMPLOYED'] / 365.25)
    df['APP_EXT_SOURCE_2*EXT_SOURCE_3*DAYS_BIRTH'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['DAYS_BIRTH']
    df['APP_SCORE1_TO_FAM_CNT_RATIO'] = df['EXT_SOURCE_1'] / df['CNT_FAM_MEMBERS']
    df['APP_SCORE1_TO_GOODS_RATIO'] = df['EXT_SOURCE_1'] / df['AMT_GOODS_PRICE']
    df['APP_SCORE1_TO_CREDIT_RATIO'] = df['EXT_SOURCE_1'] / df['AMT_CREDIT']
    df['APP_SCORE1_TO_SCORE2_RATIO'] = df['EXT_SOURCE_1'] / df['EXT_SOURCE_2']
    df['APP_SCORE1_TO_SCORE3_RATIO'] = df['EXT_SOURCE_1'] / df['EXT_SOURCE_3']
    df['APP_SCORE2_TO_CREDIT_RATIO'] = df['EXT_SOURCE_2'] / df['AMT_CREDIT']
    df['APP_SCORE2_TO_REGION_RATING_RATIO'] = df['EXT_SOURCE_2'] / df['REGION_RATING_CLIENT']
    df['APP_SCORE2_TO_CITY_RATING_RATIO'] = df['EXT_SOURCE_2'] / df['REGION_RATING_CLIENT_W_CITY']
    df['APP_SCORE2_TO_POP_RATIO'] = df['EXT_SOURCE_2'] / df['REGION_POPULATION_RELATIVE']
    df['APP_SCORE2_TO_PHONE_CHANGE_RATIO'] = df['EXT_SOURCE_2'] / df['DAYS_LAST_PHONE_CHANGE']
    df['APP_EXT_SOURCE_1*EXT_SOURCE_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['APP_EXT_SOURCE_1*EXT_SOURCE_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['APP_EXT_SOURCE_2*EXT_SOURCE_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['APP_EXT_SOURCE_1*DAYS_EMPLOYED'] = df['EXT_SOURCE_1'] * df['DAYS_EMPLOYED']
    df['APP_EXT_SOURCE_2*DAYS_EMPLOYED'] = df['EXT_SOURCE_2'] * df['DAYS_EMPLOYED']
    df['APP_EXT_SOURCE_3*DAYS_EMPLOYED'] = df['EXT_SOURCE_3'] * df['DAYS_EMPLOYED']

    # AMT_INCOME_TOTAL : income
    # CNT_FAM_MEMBERS  : the number of family members
    df['APPS_GOODS_INCOME_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL']
    df['APPS_CNT_FAM_INCOME_RATIO'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    # DAYS_BIRTH : Client's age in days at the time of application
    # DAYS_EMPLOYED : How many days before the application the person started current employment
    df['APPS_INCOME_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']

    # other features
    df['CREDIT_TO_GOODS_RATIO_2'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['APP_AMT_INCOME_TOTAL_12_AMT_ANNUITY_ratio'] = df['AMT_INCOME_TOTAL'] / 12. - df['AMT_ANNUITY']
    df['APP_INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
    df['APP_DAYS_LAST_PHONE_CHANGE_DAYS_EMPLOYED_ratio'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['APP_DAYS_EMPLOYED_DAYS_BIRTH_diff'] = df['DAYS_EMPLOYED'] - df['DAYS_BIRTH']

    return df

def process_new_customer(data) :
    
    row = pd.json_normalize(data)

    for col in row.columns:
        try:
            row[col] = pd.to_numeric(row[col])

        except ValueError:
            pass
    
    
    
    def get_and_select_features(df):

        df = df.merge(bureau_agg, how='left', on='SK_ID_CURR')
        df = df.merge(prev_agg, how='left', on='SK_ID_CURR')
        df = df.merge(pos_agg, how='left', on='SK_ID_CURR')
        df = df.merge(ins_agg, how='left', on='SK_ID_CURR')
        df = df.merge(cc_agg, how='left', on='SK_ID_CURR')
        
        df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))

        feats = ['SK_ID_CURR', 'TARGET', 'POS_SK_DPD_DEF_SUM', 'CC_DRAWING_LIMIT_RATIO_MAX', 'INSTAL_PAYMENT_RATIO_MEAN', 
             'POS_REMAINING_INSTALMENTS', 'CC_LAST_AMT_BALANCE_MEAN', 'CC_PAYMENT_DIV_MIN_MIN', 'CC_LATE_PAYMENT_VAR', 
             'NEW_DOC_KURT', 'PREV_SK_ID_PREV_NUNIQUE', 'EMERGENCYSTATE_MODE_nan', 'REFUSED_AMT_GOODS_PRICE_MAX', 
             'ORGANIZATION_TYPE_Industry_type_5', 'CC_AMT_PAYMENT_TOTAL_CURRENT_SUM', 'CC_CNT_DRAWINGS_POS_CURRENT_SUM', 
             'APPROVED_AMT_GOODS_PRICE_MAX', 'PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN', 'APPROVED_AMT_ANNUITY_MAX', 
             'BURO_CREDIT_ACTIVE_Active_MEAN', 'PREV_NAME_PRODUCT_TYPE_walk_in_MEAN', 'BURO_AMT_CREDIT_SUM_MAX', 
             'CC_CNT_DRAWINGS_ATM_CURRENT_SUM', 'ACTIVE_DAYS_CREDIT_VAR', 'ACTIVE_MONTHS_BALANCE_MAX_MAX', 
             'NAME_EDUCATION_TYPE_Higher_education', 'CC_CNT_DRAWINGS_POS_CURRENT_VAR', 'PREV_APP_CREDIT_PERC_MIN', 
             'REGION_RATING_CLIENT_W_CITY', 'NAME_HOUSING_TYPE_House_apartment', 'CLOSED_AMT_CREDIT_SUM_DEBT_SUM', 
             'APPROVED_DAYS_DECISION_MEAN', 'ANNUITY_INCOME_PERC', 'ORGANIZATION_TYPE_Services', 'FLAG_DOCUMENT_8', 
             'WALLSMATERIAL_MODE_Panel', 'ORGANIZATION_TYPE_Cleaning', 'ORGANIZATION_TYPE_Military', 'LIVINGAPARTMENTS_AVG', 
             'APARTMENTS_AVG', 'ELEVATORS_AVG', 'ORGANIZATION_TYPE_School', 'INSTAL_DPD_MEAN', 'FLOORSMIN_AVG', 'INSTAL_DBD_SUM', 
             'DAYS_BIRTH', 'INSTAL_DPD_MAX', 'ACTIVE_AMT_CREDIT_SUM_DEBT_MAX', 'OCCUPATION_TYPE_High_skill_tech_staff', 
             'CC_AMT_DRAWINGS_ATM_CURRENT_VAR', 'ACTIVE_DAYS_CREDIT_ENDDATE_MEAN', 'CC_CNT_DRAWINGS_ATM_CURRENT_MIN', 
             'PREV_NAME_TYPE_SUITE_Spouse_partner_MEAN', 'CC_AMT_DRAWINGS_OTHER_CURRENT_MEAN', 'POS_SK_DPD_DEF_MEAN', 
             'ACTIVE_DAYS_CREDIT_ENDDATE_MIN', 'PREV_AMT_DOWN_PAYMENT_MAX', 'BURO_AMT_CREDIT_SUM_SUM', 
             'PREV_NAME_TYPE_SUITE_Group_of_people_MEAN', 'PREV_APP_CREDIT_PERC_MEAN', 'INSTAL_AMT_PAYMENT_MEAN', 
             'ACTIVE_DAYS_CREDIT_UPDATE_MEAN', 'INSTAL_DAYS_ENTRY_PAYMENT_MAX', 'PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN', 
             'CC_CNT_DRAWINGS_POS_CURRENT_MIN', 'ACTIVE_AMT_CREDIT_SUM_MEAN', 'OCCUPATION_TYPE_Private_service_staff', 
             'CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN', 'BURO_AMT_CREDIT_SUM_MEAN', 'CC_AMT_DRAWINGS_OTHER_CURRENT_MIN', 
             'CC_AMT_DRAWINGS_OTHER_CURRENT_SUM', 'INSTAL_AMT_PAYMENT_SUM', 'PREV_NAME_PRODUCT_TYPE_nan_MEAN', 
             'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN', 'OCCUPATION_TYPE_HR_staff', 'BURO_AMT_CREDIT_SUM_OVERDUE_MEAN', 
             'PREV_CODE_REJECT_REASON_LIMIT_MEAN', 'CC_AMT_DRAWINGS_POS_CURRENT_MIN', 'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 
             'FLOORSMAX_MODE', 'ELEVATORS_MEDI', 'CODE_GENDER', 'INSTAL_DBD_MEAN', 'ORGANIZATION_TYPE_Advertising', 
             'EXT_SOURCE_3', 'FLAG_DOCUMENT_20', 'OCCUPATION_TYPE_Managers', 'FLAG_OWN_REALTY', 'EMERGENCYSTATE_MODE_Yes', 
             'POS_COUNT', 'LIVINGAREA_MODE', 'YEARS_BUILD_MEDI', 'AMT_CREDIT', 'INCOME_PER_PERSON', 'EMERGENCYSTATE_MODE_No', 
             'ORGANIZATION_TYPE_Police', 'FLAG_WORK_PHONE', 'LANDAREA_MEDI', 'COMMONAREA_AVG', 'ORGANIZATION_TYPE_University', 
             'ORGANIZATION_TYPE_Medicine', 'ORGANIZATION_TYPE_Telecom', 'NONLIVINGAPARTMENTS_AVG', 'WALLSMATERIAL_MODE_Block', 
             'ORGANIZATION_TYPE_Housing', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'WALLSMATERIAL_MODE_Monolithic', 
             'REGION_POPULATION_RELATIVE', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLOORSMAX_MEDI', 
             'ORGANIZATION_TYPE_Electricity', 'REGION_RATING_CLIENT', 'YEARS_BUILD_MODE', 'DAYS_ID_PUBLISH', 'TOTALAREA_MODE', 
             'WALLSMATERIAL_MODE_Mixed', 'EXT_SOURCE_1', 'FLAG_DOCUMENT_16', 'YEARS_BEGINEXPLUATATION_MODE', 'INSTAL_COUNT', 
             'ORGANIZATION_TYPE_Realtor', 'FLAG_DOCUMENT_6', 'COMMONAREA_MODE', 'FLAG_DOCUMENT_3', 'FLOORSMAX_AVG', 
             'OCCUPATION_TYPE_Laborers', 'APARTMENTS_MODE', 'ORGANIZATION_TYPE_Security', 'AMT_INCOME_TOTAL', 'ENTRANCES_AVG', 
             'PAYMENT_RATE', 'FLAG_DOCUMENT_17', 'FLAG_OWN_CAR', 'FLOORSMIN_MODE', 'FLOORSMIN_MEDI', 'FLAG_DOCUMENT_19', 
             'ORGANIZATION_TYPE_Mobile', 'INSTAL_DBD_MAX', 'LANDAREA_MODE', 'DAYS_EMPLOYED_PERC', 'INCOME_CREDIT_PERC', 
             'LIVINGAREA_AVG', 'ORGANIZATION_TYPE_Postal', 'BASEMENTAREA_AVG', 'ORGANIZATION_TYPE_Insurance', 
             'OCCUPATION_TYPE_Accountants', 'BURO_CREDIT_TYPE_Microloan_MEAN', 'NONLIVINGAREA_MEDI', 
             'INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE', 'FONDKAPREMONT_MODE_nan', 'INS_24M_AMT_BALANCE_MAX', 
             'ORGANIZATION_TYPE_Agriculture', 'CC_AMT_PAYMENT_TOTAL_CURRENT_MIN', 'CC_CNT_DRAWINGS_OTHER_CURRENT_SUM', 
             'PREV_NAME_TYPE_SUITE_Other_B_MEAN', 'CC_AMT_DRAWINGS_ATM_CURRENT_MIN', 
             'NONLIVINGAREA_AVG', 'FLAG_DOCUMENT_11', 'CC_CNT_DRAWINGS_CURRENT_MIN', 'EXT_SOURCE_2', 'NONLIVINGAREA_MODE', 
             'AMT_ANNUITY', 'BURO_CREDIT_TYPE_Mortgage_MEAN', 'AMT_GOODS_PRICE', 'APPROVED_CNT_PAYMENT_MEAN', 'FLAG_DOCUMENT_7', 
             'FLAG_DOCUMENT_18', 'NONLIVINGAPARTMENTS_MEDI', 'ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN', 'ORGANIZATION_TYPE_Construction', 
             'INSTAL_AMT_PAYMENT_MIN', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN']


        # Create any missing columns with a default value of 0
        for feature in feats:
            if feature not in df.columns:
                df[feature] = 0

        # Select the features
        df = df[feats]

        return df
    
    # process new row
    new_customer_id = row['SK_ID_CURR']
    new_customer_id = new_customer_id[0]
    new_customer_id = new_customer_id.item()
    
    df = application(row)
    df = get_and_select_features(df)
    
    # processed_info = df.loc[df['SK_ID_CURR'] == new_customer_id]
    processed_info = df
    
    if 'TARGET' in processed_info.columns:
        processed_info = processed_info.drop(columns=['TARGET'])
    
    processed_info.set_index('SK_ID_CURR', inplace=True)
    
    
   
    # MAKE PREDICTION
    
    y_prob = xgb_clf.predict_proba(processed_info)[:, 1]
    y_prob = y_prob[0]
    y_prob = y_prob.round(4)
    y_prob = y_prob.item()
    
    
    y_pred = (y_prob >= optimal_threshold).astype(int)
    y_pred = y_pred.item()
   
    
    return new_customer_id, y_prob, y_pred



@app.route('/process_customer', methods=['POST'])

def process_customer():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        new_customer_id, y_prob, y_pred = process_new_customer(data)

        response = {
            'new_customer_id': new_customer_id,
            'y_prob': y_prob,
            'y_pred': y_pred
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)