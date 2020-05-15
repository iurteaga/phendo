#!/usr/bin/python

# Imports
import sys, os, re, time
import argparse
import pdb
import pickle
from itertools import *
# Science
import numpy as np
import scipy.stats as stats
import pandas as pd

#####################################################################
######## Useful functions to process WERF and Phendo profile surveys
#####################################################################
### WERF related
# Merging overall health question
def health_to_category(health):
    health_category=np.nan
    if health =='Poor':
        health_category='poor'
    elif health == 'Fair':
        health_category='poor'
    elif health == 'Good':
        health_category='good'
    elif health== 'Very good':
        health_category='excellent'
    elif health=='Excellent':
        health_category='excellent'
    return health_category

# Age questions to continous
def first_period_to_age(first_period):
    first_age=np.nan

    if first_period=='8 years or younger':
        first_age=8
    elif first_period=='17 years or older':
        first_age=17
    elif first_period=='Uncertain':
        first_age=np.nan
    else:
        first_age=int(first_period)
    return first_age

# Day questions to continous
def days_to_number(day):
    day_number=np.nan
    if day=='>= 20':
        day_number=20
    elif day=='too irregular to say':
        day_number=np.nan
    else:
        day_number=int(day)
    return day_number

def load_werf_and_phendo_data(werf_file, werf_to_phendo_file):
    #### Load WERF INFO
    # Load to dataframe
    werf_data=pd.read_csv(werf_file, quotechar='"')

    # Timestamp
    timestamp_column=werf_data.columns[2]
    # Email info
    email_column=werf_data.columns[3]
    # Drop NaN emails
    werf_data.dropna(subset=[email_column], inplace=True)
    # There are duplicates, thus pick the ones with latest survey timestamp and completed
    werf_data.sort_values(by=['Complete?',timestamp_column], ascending=[False,True], inplace=True)
    werf_data.drop_duplicates(subset=[email_column],keep='last', inplace=True)
    
    #### WERF to PHENDO MAPPING INFO
    # Load to dataframe
    werf_to_phendo=pd.read_csv(werf_to_phendo_file, quotechar='"')
    # Doublecheck there are NaNs in participant_id
    werf_to_phendo.dropna(subset=['participant_id'], inplace=True)

    # Map werf_data to phendo user
    werf_data_to_phendo=werf_data.merge(werf_to_phendo, left_on=[email_column], right_on=['email'])

    # Deidentify (and drop other unnecesary fields)
    werf_data_to_phendo.drop(labels=[email_column,'Record ID', 'Survey Identifier', 'name_from_redcap','email','rid'], axis='columns', inplace=True)

    # Fix character issue
    werf_data_to_phendo['How many days of bleeding did you have usually have each period in the last 3 months?  Not counting discharge/spotting for which you needed a panty liner only) '][werf_data_to_phendo['How many days of bleeding did you have usually have each period in the last 3 months?  Not counting discharge/spotting for which you needed a panty liner only) ']=='&#8805 20']='>= 20'
    # Recast
    werf_data_to_phendo['participant_id']=werf_data_to_phendo['participant_id'].astype(int)
    werf_data_to_phendo=werf_data_to_phendo.astype(pd.api.types.CategoricalDtype())
    werf_data_to_phendo['timestamp']=pd.to_datetime(werf_data_to_phendo[timestamp_column], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    werf_data_to_phendo.drop(labels=[timestamp_column], axis='columns', inplace=True)
    
    # Return combined dataframe
    return werf_data_to_phendo

def clean_werf_data_to_phendo(werf_data_to_phendo):
    # Continuous variables
    werf_data_to_phendo=werf_data_to_phendo.astype(
        {'What is your height in feet?':float,
         'What is your height in inches?':float,
         'What is your current weight?':float,
         'Since 18 years of age, what is the most you have ever weighed (not including pregnancy and the 12 months following pregnancy)?':float,
         'How old were when you weighed that amount?':float,
         'Since 18 years of age, what is the least you have weighed ?':float,
         'How old were you when you weighed that amount?':float,
         'Age':float,
         'How old were you when you had your first menstrual period?':pd.api.types.CategoricalDtype(), #(categorical, because of less than older than)
         'At what age did you start having pain with your period?':float,
         'How many surgical procedures have you had for endometriosis or pelvic pain?':float,
         'How old were you when you first had symptoms?':float,
         'How many doctors did you see before receiving a diagnosis of endometriosis?':float,
         'How many days of bleeding did you have usually have each period in the last 3 months?  Not counting discharge/spotting for which you needed a panty liner only) ':pd.api.types.CategoricalDtype(), #(categorical, because of more than)
              })
    
    # Debugging
    '''
    for (column_idx, column) in enumerate(werf_data_to_phendo.columns):
        print('{}: {} of type {}'.format(column_idx, column, werf_data_to_phendo[column].dtype))
        if werf_data_to_phendo[column].dtype.str=='|O08':  #pd.api.types.CategoricalDtype() in string
            print('\t with values {}'.format(werf_data_to_phendo[column].cat.categories.values))
        elif werf_data_to_phendo[column].dtype.str=='<f8': #float64 in string
            print('\t with values {}'.format(np.unique(werf_data_to_phendo[column].values)))
        else:
            print('Unexpected dtype={} ({}) in {}:{}'.format(werf_data_to_phendo[column].dtype.str, werf_data_to_phendo[column].dtype, column_idx, column))
    '''
    werf_data_to_phendo['How many days of bleeding did you have usually have each period in the last 3 months?  Not counting discharge/spotting for which you needed a panty liner only) (continuous)']=werf_data_to_phendo['How many days of bleeding did you have usually have each period in the last 3 months?  Not counting discharge/spotting for which you needed a panty liner only) '].apply(days_to_number).astype(float)

    # Simplify period regularity
    # Regular = 'extremely regular (period starts 1-2 days before or after it is expected)', 'very regular (period starts 3-4 days before or after it is expected)', 'regular (period starts 5-7 days before or after it is expected)',
    # Irregular = 'somewhat irregular (period starts 8-20 days before or after it is expected)', 'irregular (period starts more than 20 days before or after it is expected)'
    werf_data_to_phendo['period_regularity_simple']=werf_data_to_phendo['Were your periods in the the last 3 months regular ?'].isin(['extremely regular (period starts 1-2 days before or after it is expected)', 'very regular (period starts 3-4 days before or after it is expected)', 'regular (period starts 5-7 days before or after it is expected)']).replace({True:'period_regular', False:'period_irregular'}).astype(pd.api.types.CategoricalDtype())

    # Merge last period pains
    last_period_pain_columns=[
        'During your last period, did your pelvic pain prevent you from going to work or school or carrying out your daily activities (even if taking pain-killers)?',
        'During your last period, did you have to lie down for any part of the day or longer because of your pelvic pain?'
        ]

    werf_data_to_phendo['last_period_pain']=werf_data_to_phendo[last_period_pain_columns].apply(lambda col_values : np.any(col_values=='Yes'), axis=1).replace({True:'last_period_pain_yes', False:'last_period_pain_no'}).astype(pd.api.types.CategoricalDtype())
    werf_data_to_phendo['How old were you when you had your first menstrual period? (continuous)']=werf_data_to_phendo['How old were you when you had your first menstrual period?'].apply(first_period_to_age).astype(float)
    
    # Merge daily living activity issues
    daily_living_columns=[
        'Vigorous activities, such as running, lifting heavy objects, participating in strenuous sports',
        'Moderate activities, such as moving a table, pushing a vaccum cleaner, bowling, or playing golf',
        'Lifting or carrying groceries',
        'Climbing several flights of stairs',
        'Climbing one flight of stairs',
        'Bending, kneeling or stooping',
        'Walking more than a mile',
        'Walking several blocks',
        'Walking one block',
        'Bathing or dressing yourself'
        ]
    werf_data_to_phendo['daily_living_columns']=werf_data_to_phendo[daily_living_columns].apply(lambda col_values : np.any(col_values.isin(['Yes, limited a little', 'Yes, limited a lot'])), axis=1).replace({True:'daily_living_affected', False:'daily_living_not_affected'}).astype(pd.api.types.CategoricalDtype())
    
    # Overal health
    werf_data_to_phendo['overall_health']=werf_data_to_phendo['In general, would you say your health is:'].apply(health_to_category).astype(pd.api.types.CategoricalDtype())
    # TODO: Merge diseases?
    # Merge laparoscopy
    laparoscopy_columns=[
        'Laparoscopy #1 (choice=Yes)',
        'Laparoscopy #2 (choice=Yes)',
        'Laparoscopy #3 (choice=Yes)',
        'Laparoscopy #4 (choice=Yes)',
        'Laparoscopy #5 (or last if more than 5) (choice=Yes)']
    werf_data_to_phendo['Laparoscopy count']=werf_data_to_phendo[laparoscopy_columns].apply(lambda col_values : 0 if np.all(np.array(col_values=='Unchecked')) else np.argwhere(np.array(col_values=='Checked'))[-1][0]+1, axis=1).astype(float)

    # Merge some abdominal surgery
    surgery_columns=[
        'Laparoscopy #1 (choice=Yes)',
        'Laparoscopy #2 (choice=Yes)',
        'Laparoscopy #3 (choice=Yes)',
        'Laparoscopy #4 (choice=Yes)',
        'Laparoscopy #5 (or last if more than 5) (choice=Yes)',
        'Any other abdominal surgery ? (choice=Yes)',
        'Any additional other abdominal surgery? (choice=Yes)']

    werf_data_to_phendo['any abdominal surgery']=werf_data_to_phendo[surgery_columns].apply(lambda col_values : np.any(col_values=='Checked'), axis=1).replace({True:'some_abdominal_surgery', False:'no_abdominal_surgery'}).astype(pd.api.types.CategoricalDtype())

    # Endometriosis in family
    #   Daugther and sister questions are doubtful, since we don't know if they have a daughter/sister?
    family_endometriosis=[
        'Mother (choice=Endometriosis)',
        'Maternal Grandmother, aunt, and/or cousin (choice=Endometriosis)',
        'Paternal Grandmother, aunt, and/or cousin (choice=Endometriosis)',
        'Sister (choice=Endometriosis)',
        'Daughter (choice=Endometriosis)']

    werf_data_to_phendo['family endometriosis']=werf_data_to_phendo[family_endometriosis].apply(lambda col_values : np.any(col_values=='Checked'), axis=1).replace({True:'Checked', False:'Unchecked'}).astype(pd.api.types.CategoricalDtype())

    # Chronic pelvic pain in family
    #   Daugther and sister questions are doubtful, since we don't know if they have a daughter/sister?
    family_chronic_pelvic_pain=[
        'Mother (choice=Chronic pelvic pain)',
        'Maternal Grandmother, aunt, and/or cousin (choice=Chronic pelvic pain)',
        'Paternal Grandmother, aunt, and/or cousin (choice=Chronic pelvic pain)',
        'Sister (choice=Chronic pelvic pain)',
        'Daughter (choice=Chronic pelvic pain)']

    werf_data_to_phendo['family chronic pelvic pain']=werf_data_to_phendo[family_chronic_pelvic_pain].apply(lambda col_values : np.any(col_values=='Checked'), axis=1).replace({True:'Checked', False:'Unchecked'}).astype(pd.api.types.CategoricalDtype())

    # Some cancer       
    werf_data_to_phendo['some cancer']=werf_data_to_phendo['Type of 1st cancer (primary location) you been diagnosed with.'].isna().replace({False:'Checked', True:'Unchecked'}).astype(pd.api.types.CategoricalDtype())

    # Other chronic medical condition
    other_chronic_condition=[
            'Please specify the first other chronic medical condition', # Categories (157, object)
            'Please specify the second other chronic medical condition', # Categories (77, object)
            'Please specify the third other chronic medical condition', # Categories (34, object)
            ]

    werf_data_to_phendo['other chronic condition']=werf_data_to_phendo[other_chronic_condition].apply(lambda col_values : not np.all(col_values.isna()), axis=1).replace({True:'Checked', False:'Unchecked'}).astype(pd.api.types.CategoricalDtype())

    # SELECT cohort and drop unnecessary columns
    # Let's try with just completed, a way to identify others would be with NaN in a previous question
    werf_data_to_phendo=werf_data_to_phendo[werf_data_to_phendo['Complete?']=='Complete']
    werf_data_to_phendo.drop(labels=['Complete?', 'timestamp'], axis='columns', inplace=True)
    # Drop shape related questions: the goal was to check for completeness via NaNs
    werf_data_to_phendo.drop(labels=['Ages 10-14', 'Ages 15-19', 'Ages 20-24', 'Ages 25-29', 'Ages 30-34',
           'Ages 35-39', 'Ages 40-44', 'Ages &#8805 45 yrs', '50 +', '40-49',
           '15-19', '20-29', '30-39'], axis='columns', inplace=True)
    return werf_data_to_phendo

### Profile related
# Weight in pounds
def weight_to_lbs(entry):
    units=entry['str_value'].drop_duplicates()
    value=entry['parent_id'].drop_duplicates()
    if units.shape[0]==1:
        units=units.values[0]
    else:
        units=units.dropna().values[0]
    if value.shape[0]==1:
        value=value.values[0]
    else:
        value=value.dropna().values[0]
    
    #print('{} {}'.format(value, units))
    return value*2.20462 if units == 'kg' else value

# BMI: remember to cast feet to inches!
def height_to_inches(height):
    sep_height=height.split('.')
    return int(sep_height[0])*12+int(sep_height[1])  

# BMI groups
def bmi_to_groups(bmi):
    group=np.nan
    if bmi < 15.:
        pass
    elif bmi < 18.5:
        group='underweight'
    elif bmi < 25.:
        group='normal'
    #elif bmi < 30.:
    #    group='pre-obese'
    elif bmi < 50.:
        group='obese'
    return group
    
def dob_to_age(dob):
    age=np.floor((pd.datetime.now()-pd.to_datetime(dob.split('T')[0]))/np.timedelta64(1,'Y'))
    return age if age>=13. else np.NaN

def create_participant_profile_dataframe(profile_file, age_file):
    # Load to dataframe
    profile=pd.read_csv(profile_file, quotechar='"')
    # Pick only those that have not been edited
    profile=profile[profile['is_edited']==0]

    # Per-participant weight in lbs
    participant_weight=profile[(profile['question_id']==280) | (profile['question_id']==750)][['participant_id','str_value','parent_id']].groupby('participant_id').apply(weight_to_lbs)
    # Per-participant height
    participant_height=profile[profile['question_id']==250][['participant_id','str_value']].drop_duplicates(subset=['participant_id'],keep=False)
    # Combine height and weight
    participant_bmi=pd.merge(participant_height, pd.DataFrame(participant_weight), on=['participant_id'])
    participant_bmi.rename(index=str, columns={'str_value': 'height', 0: 'weight'}, inplace=True)        
    participant_bmi['height_inches']=participant_bmi['height'].apply(height_to_inches).astype(float)
    participant_bmi.drop(labels=['height'], axis='columns', inplace=True)
    # Compute BMI
    participant_bmi['bmi']=703*participant_bmi['weight']/np.power(participant_bmi['height_inches'],2)
    participant_bmi['bmi_groups']=participant_bmi['bmi'].apply(bmi_to_groups)
    participant_bmi.dropna(subset=['bmi_groups'],inplace=True)

    # What is your race/ethnicity? 202
    participant_race=profile[profile['question_id']==202][['participant_id','str_value']].drop_duplicates(subset=['participant_id'],keep=False).reset_index(drop=True).rename(columns={'str_value':'race'})
    # What is your highest level of education? 203
    participant_education=profile[profile['question_id']==203][['participant_id','str_value']].drop_duplicates(subset=['participant_id'],keep=False).reset_index(drop=True).rename(columns={'str_value':'education'})
    # What is your living environment? 207
    participant_living=profile[profile['question_id']==207][['participant_id','str_value']].drop_duplicates(subset=['participant_id'],keep=False).reset_index(drop=True).rename(columns={'str_value':'living'})

    # Load Age info to dataframe
    participant_age=pd.read_csv(age_file, quotechar='"')
    participant_age['age']=participant_age['date_value'].apply(dob_to_age)

    # Merge all info from profile into a single dataframe
    participant_profile=pd.merge(
        pd.merge(
            pd.merge(participant_bmi, participant_race, how='outer', on='participant_id'),
            pd.merge(participant_education, participant_living, how='outer', on='participant_id'),
            how='outer', on='participant_id'
            ),
            participant_age[['participant_id','age']], how='outer', on='participant_id')

    # Recast and return
    return participant_profile.astype(
        {'participant_id':pd.api.types.CategoricalDtype(),
        'bmi_groups':pd.api.types.CategoricalDtype(),
        'race':pd.api.types.CategoricalDtype(),
        'education':pd.api.types.CategoricalDtype(),
        'living':pd.api.types.CategoricalDtype()
        })

#####################################################################
