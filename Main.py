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
import statistics 
from scipy.stats import variation
def rmsValue(arr, n): 
    square = 0
    mean = 0.0
    root = 0.0   
    #Calculate square 
    for i in range(0,n): 
        square += (arr[i]**2)   
    #Calculate Mean  
    mean = (square / (float)(n))   
    #Calculate Root 
    root = math.sqrt(mean)
    return root 

database = pd.read_csv("/Users/jadonzhou/OneDrive -Jadon/Ongoing projects/Healthcare Predictives/COVID-19/3. Comorbidity:laboratory-based risk score for Covid-19 mortality/Database.csv", encoding='windows-1252')
demographicsData = pd.read_csv("/Users/jadonzhou/OneDrive -Jadon/Ongoing projects/Healthcare Predictives/COVID-19/3. Comorbidity:laboratory-based risk score for Covid-19 mortality/DistrictRace.csv", encoding='windows-1252')# extract district and race
dists=[]
races=[]
for ref in database['Reference Key'].tolist():
    dist=demographicsData['District of Residence on Latest Selected Encounter (district)'][demographicsData['Reference Key']==ref].tolist()
    if len(dist):
        dists.append(dist[0])
    else:
        dists.append(" ")
    race=demographicsData['Race Description'][demographicsData['Reference Key']==ref].tolist()
    if len(race):
        races.append(race[0])
    else:
        races.append(" ")
dists=pd.DataFrame(dists)
races=pd.DataFrame(races)


# extract adm date
database = pd.read_csv("/Users/jadonzhou/OneDrive -Jadon/Ongoing projects/Healthcare Predictives/COVID-19/3. Comorbidity:laboratory-based risk score for Covid-19 mortality/Database_baseline.csv", encoding='windows-1252')
adm = pd.read_csv("/Users/jadonzhou/OneDrive -Jadon/Ongoing projects/Healthcare Predictives/COVID-19/3. Comorbidity:laboratory-based risk score for Covid-19 mortality/admsss.csv", encoding='windows-1252')
admresult=[]
for patient in np.unique(adm['Reference Key'].tolist()).tolist():
    date=max(adm['Admission Date (yyyy-mm-dd)'][adm['Reference Key']==patient])
    admresult.append([patient, date])
admresult=pd.DataFrame(admresult) 
admresult.columns=adm.columns  
database=pd.merge(database, admresult, how = 'left')
database.to_csv('/Users/jadonzhou/OneDrive -Jadon/Ongoing projects/Healthcare Predictives/COVID-19/3. Comorbidity:laboratory-based risk score for Covid-19 mortality/Database_baseline_new.csv')

# extract ICU date
database = pd.read_csv("/Users/jadonzhou/OneDrive -Jadon/Ongoing projects/Healthcare Predictives/COVID-19/3. Comorbidity:laboratory-based risk score for Covid-19 mortality/Database_baseline.csv", encoding='windows-1252')
ICU = pd.read_csv("/Users/jadonzhou/OneDrive -Jadon/Ongoing projects/Healthcare Predictives/COVID-19/3. Comorbidity:laboratory-based risk score for Covid-19 mortality/ICCUs.csv", encoding='windows-1252')
result=[]
for patient in np.unique(ICU['Reference Key'].tolist()).tolist():
    date=max(ICU['APACHE: 1st Date time (yyyy-mm-dd)'][ICU['Reference Key']==patient])
    result.append([patient, date])
result=pd.DataFrame(result) 
result.columns=ICU.columns  
database=pd.merge(database, result, how = 'left')
database.to_csv('/Users/jadonzhou/OneDrive -Jadon/Ongoing projects/Healthcare Predictives/COVID-19/3. Comorbidity:laboratory-based risk score for Covid-19 mortality/Database_baseline_new.csv')


# merge ICU xlsx files
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

#database1 = pd.read_csv("/Users/jadonzhou/OneDrive -Jadon/Ongoing projects/Healthcare Predictives/COVID-19/2. Short paper/Covid Negative/Covid19_negative_DoB_Sex_Race_Mortality.csv", encoding='windows-1252')
filePath = '/Users/jadonzhou/OneDrive -Jadon/Ongoing projects/Healthcare Predictives/COVID-19/3. Comorbidity:laboratory-based risk score for Covid-19 mortality/ICU/ICU parameters/'
filepaths=os.listdir(filePath)
if '.DS_Store' in filepaths:
    filepaths.remove('.DS_Store')
variables=['Reference Key', 'APACHE: 1st Date time (yyyy-mm-dd)','APACHE: 1st pH','APACHE: 1st PaCO2 (kPa)','APACHE: 1st PaCO2 mmHg','APACHE: 1st PaO2 (kPa)','APACHE: 1st PaO2 (mmHg)','APACHE: 1st HCO3-','APACHE: 1st FiO2','APACHE: 1st ETT/Trach','APACHE: 1st Mech. Vent','APACHE: 1st Arterial/Vensus','APACHE: A-aDO2 (1st set)','APACHE: urine output','APACHE: Apache IV risk of death (non-CABG)','APACHE: Apache IV risk of death (CABG)','APACHE: Apache IV length of stay (non-CABG)','APACHE: Apache IV length of stay (CABG)']
Data=pd.DataFrame(columns=variables)
for path in filepaths:
    raw = pd.read_excel(filePath+path, index_col=0)  
    if len(raw):
        data=raw.iloc[33:len(raw)-7,:]
        data.columns=raw.iloc[33,:]
        print(len(data))
        if len(data):
            #data=data[data['Reference Key'].isin(list(map(str,database['Reference Key'].tolist())))]
            if len(data):
                data=data[variables]
                Data=Data.append(data)
                print(path)
Data["Hong Kong ID"]=Data.index
#Data["Sex"]=np.nan
#Data["Date of Birth (yyyy-mm-dd)"]=np.nan
Data.to_csv(filePath+'Adm.csv')







