import pandas as pd
import numpy as np
import pandas as pd
from pandas import DataFrame
import datetime as dt
import re
import csv
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,RidgeCV,Lasso, LassoCV
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_validate
from sklearn import  metrics as mt
from  statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from decimal import *
from collections import Counter
import category_encoders as ce
import math
from scipy import stats
from scipy.stats.mstats import kruskalwallis
from pandas import read_csv
import os
from datetime import datetime
from calendar import isleap
import datetime
import time

database = pd.read_csv("/Users/jadonzhou/Research Projects/Healthcare Predictives/ACEI ARB Cancer/Data/Database.csv", encoding='windows-1252')
comCategory = pd.read_csv("/Users/jadonzhou/Research Projects/Healthcare Predictives/Tools/Comorbidity/CharlsonCodes.csv", encoding='windows-1252')
comCategory=comCategory.astype(str)
Dx = pd.read_csv("/Users/jadonzhou/Research Projects/Healthcare Predictives/ACEI ARB Cancer/Data/Dx.csv", encoding='windows-1252')
Dx=Dx[Dx['Reference Key'].isin(database['Reference Key'].tolist())]
#Dx=Dx[Dx['Reference Date']!='2015-09-']
Dx['Reference Date'] = pd.to_datetime(Dx['Reference Date'])
Dx = Dx.sort_values(by = 'Reference Date',ascending=True)
#Dx=Dx[Dx['All Diagnosis Code (ICD9)'].isin(comCategory.iloc[:,0].tolist())]
#result_age=pd.DataFrame(np.zeros((databas ane.shape[0],comCategory.shape[1])))
#result_age.columns=comCategory.columns
result_disease=pd.DataFrame(np.zeros((database.shape[0],comCategory.shape[1])))
result_disease.columns=comCategory.columns
result_date=pd.DataFrame(np.zeros((database.shape[0],comCategory.shape[1])))
result_date.columns=comCategory.columns
for p in range(database.shape[0]):
    print(p)
    #birthdat=datetime.strptime(database.iloc[p,2],'%Y/%m/%d')
    comorbidities=Dx[Dx['Reference Key']==database.iloc[p,0]]
    #comorbidities=comorbidities.sort_values(by = 'Reference Date',ascending=True)
    baselineDate=database.iloc[p,1]
    comorbidities = comorbidities[(comorbidities['Reference Date']<=pd.to_datetime(baselineDate))]
    for i in range(comorbidities.shape[0]):
        code=comorbidities.iloc[i,2]
        for j in range(comCategory.shape[1]):
            if code in comCategory.iloc[:,j].dropna().tolist() or str(code) in comCategory.iloc[:,j].dropna().tolist():
                result_disease.iloc[p,j]=1
                #result_age.iloc[p,j]=comorbidities.iloc[i,8]
                #result_date.iloc[p,j]=comorbidities.iloc[i,2]
                #result_date.iloc[p,j]=comorbidities.iloc[i,3].strftime('%Y/%m/%d')
                #difference=datetime.strptime(comorbidities.iloc[i,2],'%Y-%m-%d')-birthdat
                #result_age.iloc[p,j]=(difference.days + difference.seconds/86400)/365.2425
#result_date.to_csv('/Users/jadonzhou/Research Projects/Healthcare Predictives/ACEI ARB Cancer/Data/Charlsonresult_date.csv')        
result_disease.to_csv('/Users/jadonzhou/Research Projects/Healthcare Predictives/ACEI ARB Cancer/Data/Charlsonresult_disease.csv')        
#result_age.to_csv('/Users/jadonzhou/Research Projects/Healthcare Predictives/0. HA Cancer Projects (5+)/Data/comsafter_age.csv')        






# extract binary prior com indiccators
database = pd.read_csv("/Users/jadonzhou/Research Projects/Healthcare Predictives/Variable variability studies/FM cohort studies/Data/Drugdata.csv", encoding='windows-1252')
comCategory = pd.read_csv("/Users/jadonzhou/Research Projects/Healthcare Predictives/Tools/Comorbidity/HA ComorbiditiesCategoryCodesOnlyCancer.csv", encoding='windows-1252')
Dx = pd.read_csv("/Users/jadonzhou/Research Projects/Healthcare Predictives/0. HA Cancer Projects (5+)/Data/Dx Cancer.csv", encoding='windows-1252')
Dx=Dx[Dx['Reference Key'].isin(database['Reference Key'].tolist())]
result=pd.DataFrame(np.zeros((database.shape[0],comCategory.shape[1])))
result.columns=comCategory.columns
for p in range(database.shape[0]):
    print(p)
    comorbidities=Dx[Dx['Reference Key']==database.iloc[p,0]]
    for i in range(comorbidities.shape[0]):
        code=comorbidities.iloc[i,1]
        for j in range(comCategory.shape[1]):
            if code in comCategory.iloc[:,j].dropna().tolist() or str(code) in comCategory.iloc[:,j].dropna().tolist():
                #result.iloc[p,j]=comorbidities.iloc[i,47]
                result.iloc[p,j]=comorbidities.iloc[i,2]
result.to_csv('/Users/jadonzhou/Research Projects/Healthcare Predictives/0. HA Cancer Projects (5+)/Data/cancer coms.csv')        

