import pandas as pd
import numpy as np
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
def median(x):
    x = sorted(x)
    length = len(x)
    mid, rem = divmod(length, 2)    # divmod函数返回商和余数
    if rem:
        return x[:mid], x[mid+1:], x[mid]
    else:
        return x[:mid], x[mid:], (x[mid-1]+x[mid])/2
    
# extract coutinuous lab test results under each outcome
database = pd.read_csv("/Users/jadonzhou/OneDrive -Jadon/Ongoing projects/Healthcare Predictives/COVID-19/3. Comorbidity:laboratory-based risk score for Covid-19 mortality/Database_baseline.csv", encoding='windows-1252')
adm = pd.read_csv("/Users/jadonzhou/OneDrive -Jadon/Ongoing projects/Healthcare Predictives/COVID-19/3. Comorbidity:laboratory-based risk score for Covid-19 mortality/Adm.csv", encoding='windows-1252')
results=[]
for patient in database['Reference Key'].tolist():
    admtemp=adm[adm['Reference Key']==patient]
    if len(admtemp):
        noofEpisodes=max(admtemp['No. of Episodes (Patient Based)'])
        LOS=max(admtemp['Length of Stay (Patient Based)'])
        noofEmergencyReadmission=len(admtemp[admtemp['With Emergency Readmission Within 28 Days After Discharge in HA (Y/N)']=='Y'])
        results.append([patient, noofEpisodes, LOS, noofEmergencyReadmission])
    else:
        results.append([patient, " ", " ", " "])
pd.DataFrame(results).to_csv("/Users/jadonzhou/OneDrive -Jadon/Ongoing projects/Healthcare Predictives/COVID-19/3. Comorbidity:laboratory-based risk score for Covid-19 mortality/hospitalization.csv", encoding='windows-1252')

