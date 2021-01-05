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
from scipy import stats
from scipy.stats.mstats import kruskalwallis
from pandas import read_csv
import os
import datetime

database = pd.read_csv("/Users/jadonzhou/OneDrive -Jadon/Ongoing projects/Healthcare Predictives/0. HA Cancer Projects (5+)/ACEI ARB Lung Commodities Analysis/Data/Database.csv", encoding='windows-1252')
filePath = '/Users/jadonzhou/OneDrive -Jadon/Ongoing projects/Healthcare Predictives/0. HA Cancer Projects (5+)/Data/ACEI duration/'
filepaths=os.listdir(filePath)
if '.DS_Store' in filepaths:
    filepaths.remove('.DS_Store')
variables=['Reference Key', 'Dispensing Date (yyyy-mm-dd)','Prescription Start Date', 'Prescription End Date', 'Drug Name (Full Drug Description)','Drug Strength','Base Unit', 'No. of Item Prescribed','Dosage','Quantity (Named Patient)', 'Dispensing Duration', 'Dispensing Case Type']
Data=pd.DataFrame(columns=variables)
for path in filepaths:
    data = pd.read_html(filePath+path)[0]
    if len(data):
        data.columns=data.iloc[0,:]
        data=data.drop([0])
        data=data[data['Reference Key'].isin(list(map(str,database['Reference Key'].tolist())))]
        if len(data):
            data=data[variables]
            Data=Data.append(data)
            print(path)
Data.to_csv(filePath+'Data1.csv')


# extract drug duration and dosage data
database = pd.read_csv("/Users/jadonzhou/Research Projects/Healthcare Predictives/Variable variability studies/FM cohort studies/Data/Database.csv")
Drugdata = pd.read_csv('/Users/jadonzhou/Research Projects/Healthcare Predictives/0. HA Cancer Projects (5+)/Data/ACEI IP OP drugs.csv')
#Drugdata=Drugdata.dropna()
#Drugdata['Dispensing Duration'].fillna(36, inplace=True) 
drugClass = pd.read_csv("/Users/jadonzhou/Research Projects/Healthcare Predictives/Tools/Drug/DrugClass.csv", encoding='windows-1252')
durationResult=pd.DataFrame(np.zeros((database.shape[0],drugClass.shape[1])))
durationResult.columns=drugClass.columns
druguse=pd.DataFrame(np.zeros((database.shape[0],2)))
druguse.columns=['Date of first ACEI prescription', 'Accumulative prescription frequency']
dosageResult=pd.DataFrame(np.zeros((database.shape[0],drugClass.shape[1])))
dosageResult.columns=drugClass.columns
meanDosResult=pd.DataFrame(np.zeros((database.shape[0],drugClass.shape[1])))
meanDosResult.columns=drugClass.columns
drugs=drugClass.columns.tolist()
for i in range(database['Reference Key'].shape[0]):
    print(i)
    drugdata=Drugdata[Drugdata['Reference Key']==database.iloc[i,0]].sort_values(by='Dispensing Date (yyyy-mm-dd)')
    if len(drugdata):
        druguse.iloc[i,0]=drugdata['Dispensing Date (yyyy-mm-dd)'].iloc[0]
        druguse.iloc[i,1]=len(drugdata)
        for j in range(len(drugdata)):
            #loc=drugs.index(drugdata.iloc[j,3])
            loc=0
            durationResult.iloc[i,loc]=durationResult.iloc[i,loc]+float(drugdata.iloc[j,9])
            dosage=0
            string=drugdata.iloc[j,7]
            if str(string)!='nan' and len(str(string)):
                unit=drugdata.iloc[j,5]
                if str(string)!='nan' and len(str(string)): 
                    strs=string.split(' ')
                    if unit in ['CAP','TAB','BOTT','ML','PCS']:
                        dosage=dosage+pd.Series([float(s) for s in re.findall(r"\d+\.?\d*",string)]).prod()*float(drugdata.iloc[j,8])
                    elif unit=='AMP' and strs[1]=='MG':
                        dosage=dosage+ float(strs[0])*drugdata.iloc[j,8]
                    elif unit=='AMP' and len(strs)>3 and len(strs[2].split('MG'))>1:
                        dosage=dosage+ pd.Series([float(s) for s in re.findall(r"\d+\.?\d*",string)]).prod()*float(drugdata.iloc[j,8])
                    elif unit=='VIAL' and strs[1]=='MG':
                        dosage=dosage+ float(strs[0])*drugdata.iloc[j,8]
                    elif unit=='VIAL' and len(strs)>3 and len(strs[2].split('MG'))>1:
                        dosage=dosage+ pd.Series([float(s) for s in re.findall(r"\d+\.?\d*",string)]).prod()*float(drugdata.iloc[j,8])
                    else:
                        s=1
            dosageResult.iloc[i,loc]=dosageResult.iloc[i,loc]+dosage
#meanDosResult = dosageResult.iloc[:,1:].div(durationResult.iloc[:,1:],axis=0)
dosageResult.to_csv('/Users/jadonzhou/Research Projects/Healthcare Predictives/0. HA Cancer Projects (5+)/Data/dosageResult.csv')
durationResult.to_csv('/Users/jadonzhou/Research Projects/Healthcare Predictives/0. HA Cancer Projects (5+)/Data/durationResult.csv')
druguse.to_csv('/Users/jadonzhou/Research Projects/Healthcare Predictives/0. HA Cancer Projects (5+)/Data/druguse.csv')














