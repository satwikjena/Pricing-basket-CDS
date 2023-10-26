# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:10:39 2021

@author: Beta
"""

import numpy as np # arrays and basic maths
import pandas as pd # dataframe framework
from pylab import plt # plotting graphs
plt.style.use('seaborn')
from IPython.display import Markdown, display,HTML #formatting text
import heapq #returns smallest number in a list
import random # useful for geberating random samples
import scipy.stats #usefulf or obtaing statistic measures
import sobol_seq
from math import exp, sqrt, log
import random
from random import gauss
import scipy
from scipy import stats

#Data Import
# Credit_Spread_Hist.xls: 
#    Ist Tab: Historical Data of 5 yr cds spreads to calculate correlation
#    2nd Tab: Spot 1Y, 2Y, 3Y, 4Y, 5Y credit spreads for all issuers

cs=pd.ExcelFile("C:/Users/Beta/Google Drive/CoursesCQF/CQF Basket CDS/Credit_Spread_Hist.xls")
hist_data=cs.parse(0)
master_spot_cs=cs.parse(1)
master_hist_cs=hist_data.drop(hist_data.index[[0,1,2,3,4,5,6]])
master_DF=cs.parse(1)
#Discount factors
D=master_spot_cs["DF"].tolist()[1:]

# Base Scenario
spot_cs=master_spot_cs
typ_corr="Kendaltau"
hist_cs=master_hist_cs
copulamethod="gauss"
sims=1000
RR=0.4

#-----------------------------------------------------------------------------------------------------------

#Calculation of survival probability
#default leg = premium leg
#(1-Recovery rate)*(probprevioustenor-sp)*df*dt=marketspread/10000*df*dt*sp  
#The below function returns the survival probability for each issuer when suplloed with spot credit spreads and issuer name
def ISurvProb(spot_cs,issuer):
    RR=0.40
    issuer_sn=pd.DataFrame()
    issuer_sn=spot_cs[[issuer,"Maturity","DF"]].copy()
    issuer_sn["dt"]=1
    issuer_sn["SurvPro"]=1.0
    for i in range(1,6):
        issuer_sn["SurvPro"][i]=issuer_sn["SurvPro"][i-1]-(issuer_sn[issuer][i]/(1-RR)/issuer_sn["DF"][i]/10000)
    return issuer_sn['SurvPro'],issuer_sn

#-----------------------------------------------------------------------------------------------------------

surv_prob=pd.DataFrame()

#calculate cumulative survival probabilities using the function in previous sections. By using relationship between cumulative survival probabilities and non cumulative hazrad rates, wecalculate non cumulative hazard rates for the 5 issuers
#Lambda["lambda1"]=-np.log(surv_prob[1])
#Lambda["lambda2"]=-np.log(surv_prob[2])-Lambda["lambda1"]
#Lambda["lambda3"]=-np.log(surv_prob[3])-(Lambda["lambda1"]+Lambda["lambda2"])
#Lambda["lambda4"]=-np.log(surv_prob[4])-(Lambda["lambda1"]+Lambda["lambda2"]+Lambda["lambda3"])
#Lambda["lambda5"]=-np.log(surv_prob[5])-(Lambda["lambda1"]+Lambda["lambda2"]+Lambda["lambda3"]+Lambda["lambda4"])

IssuerList=["Allianz","UniCredit","SocGen","Intesa","Hannover"]
for i in range(5):
    p,q=ISurvProb(spot_cs,IssuerList[i])
    surv_prob[IssuerList[i]]=p
    
surv_prob=surv_prob.drop(surv_prob.index[0]).transpose()
Lambda=surv_prob.copy()
Lambda[0]=0
sum=0

for r in range(0,5):
    for c in range(0,5):
        sum=sum+Lambda[c+1][r]
        Lambda[c+1][r]=-np.log(surv_prob[c+1][r])

print(Lambda)
#-----------------------------------------------------------------------------------------------------------

# calculate take spread data for every Thursday and calculate correlation matrix

import datetime

hist_cs["dayofweek"]=hist_cs["Column"].dt.day_name()
wkly_hist_cs=hist_cs[hist_cs["dayofweek"]=="Thursday"].drop(columns=["dayofweek","Column"])
wkly_hist_cs=wkly_hist_cs.reset_index(drop=True)
wkly_hist_cs=wkly_hist_cs.astype(float)

kendalltau=wkly_hist_cs.corr(method='kendall')
#Cholesky decompoistion    
print(kendalltau)    

cholesky=scipy.linalg.cholesky(kendalltau,lower=True)
print(cholesky)
#-----------------------------------------------------------------------------------------------------------

random_numbers=[[]]*5
corr_random_numbers=[[]]*5
uniform_vector=[[]]*5

z=np.empty((5,sims))
for i in range(5):
    uni = stats.uniform(0, 1).rvs(sims) 
    norm = stats.distributions.norm.ppf(uni)
    z[i]=norm
z=z.transpose()

x=np.empty((sims,5))
u=np.empty((sims,5))
for i in range(sims):

    x[i][0]=(cholesky[0][0]*z[i][0])
    x[i][1]=(cholesky[1][0]*z[i][0]  + cholesky[1][1]*z[i][1])
    x[i][2]=(cholesky[2][0]*z[i][0]  + cholesky[2][1]*z[i][1]   + cholesky[2][2]*z[i][2])
    x[i][3]=(cholesky[3][0]*z[i][0]  + cholesky[3][1]*z[i][1]   + cholesky[3][2]* z[i][2]  + cholesky[3][3] * z[i][3])
    x[i][4]=(cholesky[4][0]*z[i][0]  + cholesky[4][1]* z[i][1]  + cholesky[4][2] * z[i][2] + cholesky[4][3] * z[i][3] + cholesky[4][4] *z[i][4])    
    u[i][0]=scipy.stats.norm.cdf(x[i][0],0,1)
    u[i][1]=scipy.stats.norm.cdf(x[i][1],0,1)
    u[i][2]=scipy.stats.norm.cdf(x[i][2],0,1)
    u[i][3]=scipy.stats.norm.cdf(x[i][3],0,1) 
    u[i][4]=scipy.stats.norm.cdf(x[i][4],0,1)


Hazard=np.empty((sims,5))
tao=np.empty((sims,5))
timetodefault=np.empty((sims,5))

ucdf=np.log(1-u)
for i in range(5):
    for j in range(sims):
        sum = 0
        for tenor in range(5):
            if -(sum + Lambda[i+1][tenor]) < ucdf[j][i]:
                if updcdf[j][i] < -sum:
                    timetodefault[j][i] = tenor
                    Hazard[j][i] = Lambda[i+1][tenor]
                    break
            else:
                sum = sum + Lambda[i][tenor]

            timetodefault[j][i] = "5"
            Hazard[j][i] = Lambda[i+1][4]

for i in range(5):
    for j in range(sims):
        tao[j][i] = -ucdf[j][ i] / Hazard[j][i]
        if (tao[j][i]) <= 0 :
            tao[j][i] = np.abs(tao[i][j])
taotranspose=pd.DataFrame(tao).T
taosorted = pd.DataFrame(np.sort(taotranspose, axis=0), index=taotranspose.index, columns=taotranspose.columns).T


premium_leg=interpolatedDF(taosorted[0])*taosorted[0]*(taosorted[0]<5)
default_leg=(1-RR)*interpolatedDF(taosorted[0])*1/5*(taosorted[0]<5)
ftd=default_leg.sum()/premium_leg.sum()*10000

premium_leg=interpolatedDF(taosorted[0])*taosorted[0]*(taosorted[0]<5) + interpolatedDF(taosorted[1])*(taosorted[1]-taosorted[0])*(taosorted[1]<5)*4/5
default_leg=(1-RR)*interpolatedDF(taosorted[1])*1/5*(taosorted[1]<5)
default_leg.sum()/premium_leg.sum()*10000
secondtd=default_leg.sum()/premium_leg.sum()*10000

premium_leg=interpolatedDF(taosorted[0])*taosorted[0]*(taosorted[0]<5) + interpolatedDF(taosorted[1])*(taosorted[1]-taosorted[0])*(taosorted[1]<5)*4/5 + interpolatedDF(taosorted[2])*(taosorted[2]-taosorted[1])*(taosorted[2]<5)*4/5
default_leg=(1-RR)*interpolatedDF(taosorted[2])*1/5*(taosorted[1]<5)
default_leg.sum()/premium_leg.sum()*10000
thirdtd=default_leg.sum()/premium_leg.sum()*10000

print("First to default : ",ftd)
print("First to default : ",secondtd)
print("First to default : ",thirdtd)