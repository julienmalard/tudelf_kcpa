# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:59:23 2021

@author: denni
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#%%

dat=pd.read_csv("Final_Analysis_345_nrh_TijmenData_v4_villageEdit.csv", header=0)
dat2=pd.read_excel("../input_data/KPCA/KPCA_dat_adj.xlsx", header=0)

#%%
Village=dat["General/village_name"]
Income=dat["addtl_income_total"]
CotYield=dat["total_cotton_yield"]
CotPrice=dat["price_cotton_factors/crop_selling_price"]
lat=dat['General/GPS_lat']
long=dat['General/GPS_long']
toterr=dat2["YieldDiff"]
IncomeTot=Income+(CotYield*CotPrice)

Village_un = Village.unique()

col=['Lat','Long','Income','TotalError','CotYield','CotPrice','IncomeYield','IncomeTot']
results=pd.DataFrame(index=range(0,len(Village_un)),columns=col)

for i in range(len(Village_un)):
    tmp=Village==Village_un[i]
    results.Income[i]=np.round(Income[tmp].mean())
    results.CotYield[i]=np.round(CotYield[tmp].mean())*100   #from quintal to kg
    results.CotPrice[i]=np.round(CotPrice[tmp].mean())/100   #from INR/quintal to INR/kg
    results.TotalError[i]=toterr[tmp].mean()
    results.Lat[i]=lat[tmp].unique()[0]
    results.Long[i]=long[tmp].unique()[0]
    results.IncomeYield[i]=results.CotYield[i]*results.CotPrice[i]
    results.IncomeTot[i]=results.Income[i]+results.IncomeYield[i]


results.to_csv('IncomeErrorMap.csv', index=False)

#%%
plt.figure(figsize=(10,8)) 
Xplot=np.array(results['IncomeTot'], dtype=float)
Yplot=np.array(results['TotalError'], dtype=float)
Xplot2 = sm.add_constant(Xplot)
estPlot = sm.OLS(Yplot, Xplot2)
estPlot2=estPlot.fit()
r2=round(estPlot2.rsquared,3)
plt.scatter(results.IncomeTot,results.TotalError,c="red",s=20, edgecolor='k')
plt.plot(np.sort(Xplot), np.sort(estPlot2.predict(Xplot2)), color="black",linestyle='--')
plt.text(100000,1600,'$r^2$ = {}'.format(r2), ha='left', va='center',fontsize=22)
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.title("Total Income vs. Total Error",fontsize=18)
plt.xlabel("Total Income [INR/y]",fontsize=18)
plt.ylabel("Total Error [kg/ha]",fontsize=18)
plt.show

plt.savefig('IncomeVsTotalError.png',dpi=300, bbox_inches = "tight")

# plt.figure(figsize=(10,8)) 
# plotfit=np.arange(-5000,120000,5000)
# Xplot=np.array(IncomeTot, dtype=float)
# Yplot=np.array(toterr, dtype=float)
# Xplot2 = sm.add_constant(Xplot)
# plotfit2 = sm.add_constant(plotfit)
# estPlot = sm.OLS(Yplot, Xplot2)
# estPlot2=estPlot.fit()
# r2=round(estPlot2.rsquared,3)
# plt.scatter(Xplot,Yplot,c="red",s=20, edgecolor='k')
# plt.plot(Xplot, estPlot2.predict(Xplot2), color="black",linestyle='--')
# plt.text(1000,1400,'$r^2$ = {}'.format(r2), ha='left', va='center',fontsize=22)
# plt.grid(color='k', linestyle='-', linewidth=0.1)
# plt.title("Income vs. Total Error",fontsize=18)
# plt.xlabel("Income [INR/y]",fontsize=18)
# plt.ylabel("Total Error [kg/ha]",fontsize=18)
# plt.show
