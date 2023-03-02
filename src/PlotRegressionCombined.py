# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 20:00:43 2023

@author: denni
"""
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})


from sklearn.linear_model import LinearRegression

SPSS_dat=pd.read_excel("PCs_TotalError_WithModelYieldNoPrice_EV1.xlsx", header=0)

plotfit=np.arange(-3,5.5,0.5)

plt.figure(figsize=(15,17))
plt.title('Total Error by Prediction by Each PC')
plt.xlabel('Month')
plt.ylabel('Total Error [kg/ha]')
namePC=['PC1','PC2','PC3','PC4','PC5']
plt.rcParams.update({'font.size': 18})
Yplot=np.array(SPSS_dat.iloc[:,-1])
for i in range(len(namePC)):
    Xplot=np.array(SPSS_dat[namePC[i]])
    plt.subplot(3,2,i+1)
    plt.scatter(Xplot, Yplot, c="red",s=20, edgecolor='k') #,label='PC {}'.format(i+1))
    
    fit=np.poly1d(np.polyfit(Xplot, Yplot, 1))
    model=LinearRegression().fit(Xplot.reshape(-1,1),Yplot.reshape(-1,1))
    adj_r2Plot=round(model.score(Xplot.reshape(-1,1), Yplot.reshape(-1,1)),3)

    plt.text(1.2,-1900,'PC {}'.format(i+1), ha='center', va='center',fontsize=22)
    plt.text(5.8,3300,'$r^2$ = {}'.format(adj_r2Plot), ha='right', va='center',fontsize=22)
    plt.plot(plotfit, fit(plotfit), color="black",linestyle='--',linewidth=2)
    plt.xlim(-3.5,6)
    plt.grid(color='k', linestyle='-', linewidth=0.1)
    plt.tight_layout(pad=1.0)
  
plt.text(7, 16000, 'Total Error by Principal Components', ha='center', fontsize=30)
plt.text(7, -2700, 'Principal components ', ha='center', fontsize=30)
plt.text(-6, 7000, 'Total error [kg/ha]', va='center', rotation='vertical', fontsize=30)

plt.savefig('RegressionCombined_WithModelYieldNoPrice_test.png'.format(namePC),dpi=300, bbox_inches = "tight")
