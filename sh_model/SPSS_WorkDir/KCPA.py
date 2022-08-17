# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:35:17 2021

@author: denni
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PLOT_DIR = "plots"
plt.rcParams.update({'font.size': 12})
if not os.path.isdir(PLOT_DIR):
    os.makedirs(PLOT_DIR)
# %%get dat
# dat=pd.read_excel("KPCA_dat.xlsx", header=0)
dat = pd.read_excel("../../input_data/Data_Dennis/Data/KPCA/KPCA_dat_adj.xlsx", header=0)
dat_irr = pd.read_excel("../../input_data/Data_Dennis/Data/KPCA/KPCA_dat_adj_irr.xlsx", header=0)

X = StandardScaler().fit_transform(dat.iloc[:, 0:-3])
X_irr = StandardScaler().fit_transform(dat_irr.iloc[:, 0:-3])
# X = StandardScaler().fit_transform(dat.iloc[:,[0,1,3,4,5,6,7,8,9,10,12]])
# X = StandardScaler().fit_transform(dat.iloc[:,[0,1,2]])
# X = StandardScaler().fit_transform(np.array(dat.CotYield).reshape(-1,1))
y = dat.iloc[:, -3]

# split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Running KPCA
kernelType = "sigmoid"
deg = 5
kpca = KernelPCA(kernel=kernelType, fit_inverse_transform=True, n_components=None, degree=deg)
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

# %%
V = kpca.alphas_
D = kpca.lambdas_
D_cumsum = np.cumsum(D)
var_explained = D_cumsum / np.sum(D)
nCount = sum(var_explained <= 0.90) + 1

Xplot = np.arange(1, len(var_explained) + 1)
Yplot = var_explained
print(len(D))

plt.figure(figsize=(10, 7))
plt.plot(Xplot, Yplot, c="blue")
plt.axhline(y=0.90, color='r', linestyle='--')
plt.axvline(x=nCount, color='k', linestyle='--')
plt.xlim(0, 50)
plt.ylim(0, 1)
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.text(nCount + 1, 0.5, 'Number of PCs to explain >90% variance: {}'.format(nCount), ha='left', va='center',
         fontsize=18)
plt.text(-5, 0.9, 'var=0.9', ha='left', va='center')
plt.title("Variance Explained, Kernel: {}".format(kernelType), fontsize=18)
# plt.title("Variance Explained, Kernel: {}, Degree: {}" .format(kernelType,deg),fontsize=18)
plt.xlabel("Principal Components in Feature Space", fontsize=18)
plt.ylabel("Variance Explained [-]", fontsize=18)
plt.savefig(f'{PLOT_DIR}/varExplained_{kernelType}_deg{deg}.png',dpi=300, bbox_inches = "tight")

# %% fitting model with significant pval
X_model = X_train[:, 1:nCount + 1]
X2 = sm.add_constant(X_model)
est = sm.OLS(y_train, X2)
est2 = est.fit()
# idx_est=est2.pvalues<0.05
print(est2.summary())

# %% Get rid of the model with low significance
# X_model=X_train[:,1]
# X_model=X_train[:,(2,3,4,5,7)]
X_model = X_train[:, (1, 2, 3, 4, 5, 6)]  # sigmoid
# X_model=X_train[:,(1,2,3,4,5,6,8,9,10,12,15,16,17,21,23,25)] #for rbf
X2 = sm.add_constant(X_model)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())

r2 = round(est2.rsquared, 3)
adj_r2 = round(est2.rsquared_adj, 3)
f = est2.fvalue
beta = est2.params
pval = est2.pvalues
y_pred_train = est2.predict(X2)

plt.rc('figure', figsize=(8, 5))
plt.text(0.01, 0.05, str(est2.summary()), {'fontsize': 10},
         fontproperties='monospace')  # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/KPCA_Summary_{kernelType}_deg{deg}.png',dpi=300, bbox_inches = "tight")

MAE = np.sum(abs(y_pred_train - y_train)) / len(y_pred_train)
NS = 1 - (np.sum((y_pred_train - y_train) ** 2) / np.sum((y_train.mean() - y_train) ** 2))
NS_log = 1 - (np.sum((np.log10(y_pred_train) - np.log10(y_train)) ** 2) / np.sum(
    (np.log10(y_train.mean()) - np.log10(y_train)) ** 2))

print('MAE of {} Kernel PCA Train: {} [kg/ha]'.format(kernelType, MAE))
print('NS of {} Kernel PCA Train: {} [kg/ha]'.format(kernelType, NS))
print('NS log of {} Kernel PCA Train: {} [kg/ha]'.format(kernelType, NS_log))

# predict
# X2_test=sm.add_constant(X_test[:,1])
# X2_test=sm.add_constant(X_test[:,(2,3,4,5,7)])
X2_test = sm.add_constant(X_test[:, (1, 2, 3, 4, 5, 6)])  # sigmoid
# X2_test=sm.add_constant(X_test[:,(1,2,3,4,5,6,8,9,10,12,15,16,17,21,23,25)]) #for rbf
y_pred = est2.predict(X2_test)

MAE = np.sum(abs(y_pred - y_test)) / len(y_pred)
NS = 1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test.mean() - y_test) ** 2))
NS_log = 1 - (np.sum((np.log10(y_pred) - np.log10(y_test)) ** 2) / np.sum(
    (np.log10(y_test.mean()) - np.log10(y_test)) ** 2))

print('MAE of {} Kernel PCA Test: {} [kg/ha]'.format(kernelType, MAE))
print('NS of {} Kernel PCA Test: {} [kg/ha]'.format(kernelType, NS))
print('NS log of {} Kernel PCA Test: {} [kg/ha]'.format(kernelType, NS_log))

# %% plots test data KPCA
plt.figure(figsize=(10, 8))
Xplot = y_pred
Yplot = y_test

plt.scatter(Xplot, Yplot, c="red", s=20, edgecolor='k')
Xplot2 = sm.add_constant(Xplot)
estPlot = sm.OLS(Yplot, Xplot2)
estPlot2 = estPlot.fit()
# betaPlot=estPlot2.params
adj_r2Plot = round(estPlot2.rsquared_adj, 3)
plt.text(-1800, 3600, '$r^2$ = {}'.format(adj_r2Plot), ha='left', va='center', fontsize=18)
plt.plot(np.arange(-2000, 4000), np.arange(-2000, 4000), color="black")
plt.xlim(-2000, 4000)
plt.ylim(-2000, 4000)
# plt.title("Predicted vs. Observed Total Error, Kernel: {}, Degree: {}".format(kernelType,deg),fontsize=18)
plt.title("Predicted vs. Observed Total Error, Kernel: {} (test data)".format(kernelType), fontsize=18)
plt.xlabel("Predicted Total Error [kg/ha]", fontsize=18)
plt.ylabel("Observed Total Error [kg/ha]", fontsize=18)
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.savefig(f'{PLOT_DIR}/KPCA_{kernelType}_deg{deg}.png',dpi=300, bbox_inches = "tight")

# %% ################################## KPCA ALL DATA ##############################################
# transform all data and estimate them
X_all = kpca.transform(X)
X2_all = sm.add_constant(
    X_all[:, (1, 2, 3, 4, 5, 6)])  # components cumulative to >90% variance and significant in regression model
y_pred = est2.predict(X2_all)

# plots
plt.figure(figsize=(10, 8))
Xplot = y_pred
Yplot = y

plt.scatter(Xplot, Yplot, c="red", s=20, edgecolor='k')
Xplot2 = sm.add_constant(Xplot)
estPlot = sm.OLS(Yplot, Xplot2)
estPlot2 = estPlot.fit()
betaPlot = estPlot2.params
adj_r2Plot = round(estPlot2.rsquared_adj, 3)
plt.text(-1800, 3600, '$r^2$ = {}'.format(adj_r2Plot), ha='left', va='center', fontsize=18)
plt.plot(np.arange(-2000, 4000), np.arange(-2000, 4000), color="black")
plt.plot(np.unique(Xplot), np.unique(estPlot2.predict(sm.add_constant(Xplot))), color="black", linestyle='--')
plt.xlim(-2000, 4000)
plt.ylim(-2000, 4000)
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.title("Predicted vs. Observed Total Error, Kernel: {}".format(kernelType), fontsize=18)
plt.xlabel("Predicted Total Error [kg/ha]", fontsize=18)
plt.ylabel("Observed Total Error [kg/ha]", fontsize=18)

plt.savefig(f'{PLOT_DIR}/KPCA_{kernelType}_ALLDATA.png',dpi=300, bbox_inches = "tight")

MAE = np.sum(abs(y_pred - y)) / len(y_pred)
NS = 1 - (np.sum((y_pred - y) ** 2) / np.sum((y.mean() - y) ** 2))
NS_log = 1 - (np.sum((np.log10(y_pred) - np.log10(y)) ** 2) / np.sum((np.log10(y.mean()) - np.log10(y)) ** 2))

print('MAE of {} Kernel PCA Test: {} [kg/ha]'.format(kernelType, MAE))
print('NS of {} Kernel PCA Test: {} [-]'.format(kernelType, NS))
print('NS log of {} Kernel PCA Test: {} [-]'.format(kernelType, NS_log))

# %% plot adjusted yield
Yield_adj = dat.iloc[:, -2] + y_pred

Xplot = Yield_adj
Yplot = dat.iloc[:, -1]

plt.figure(figsize=(10, 8))
plt.fill_between(np.unique(Xplot), np.unique(estPlot2.predict(sm.add_constant(Xplot))) + 450,
                 np.unique(estPlot2.predict(sm.add_constant(Xplot))) - 450, color='yellow', alpha='0.5')
plt.scatter(Xplot, Yplot, c="red", s=20, edgecolor='k')
Xplot2 = sm.add_constant(Xplot)
estPlot = sm.OLS(Yplot, Xplot2)
estPlot2 = estPlot.fit()
betaPlot = estPlot2.params
adj_r2Plot = round(estPlot2.rsquared_adj, 3)
plt.text(300, 4500, '$r^2$ = {}'.format(adj_r2Plot), ha='left', va='center', fontsize=18)
# plt.errorbar(Xplot, Yplot,  0, 450, fmt='r^',label='Annual yield per farmer', elinewidth =0.5,
#               marker='x', markersize='5',markeredgecolor='blue', ecolor=['red'],barsabove=False)
plt.plot(np.arange(0, 5000), np.arange(0, 5000), color="black")
plt.plot(np.unique(Xplot), np.unique(estPlot2.predict(sm.add_constant(Xplot))), color="black", linestyle='--')
plt.xlim(0, 5000)
plt.ylim(0, 5000)
plt.title("Predicted Yield vs. Observed Yield", fontsize=18)
plt.xlabel("Predicted Yield [kg/ha]", fontsize=18)
plt.ylabel("Observed Yield [kg/ha]", fontsize=18)
plt.savefig(f'{PLOT_DIR}/AdjustedYieldUncer.png',dpi=300, bbox_inches = "tight")

MAE = np.sum(abs(Xplot - Yplot)) / len(Xplot)
NS = 1 - (np.sum((Xplot - Yplot) ** 2) / np.sum((Yplot.mean() - Yplot) ** 2))
NS_log = 1 - (
        np.sum((np.log10(Xplot) - np.log10(Yplot)) ** 2) / np.sum((np.log10(Yplot.mean()) - np.log10(Yplot)) ** 2))

print('MAE of {} Predicted Yield: {} [kg/ha]'.format(kernelType, MAE))
print('NS of {} Predicted Yield: {} [-]'.format(kernelType, NS))
print('NS log of {} Predicted Yield: {} [-]'.format(kernelType, NS_log))
# %% histogram of er
a = dat.iloc[:, -1] - (dat.iloc[:, -2] + y_pred)

mean = 0
std = 150
# mean,std=norm.fit(a)
x = np.linspace(mean - 3 * std, mean + 3 * std, 100)

bins = np.linspace(-1000, 1000, 100)
labels = ['$\epsilon_r$']

plt.figure(figsize=(12, 7))
plt.hist(a, bins, rwidth=0.8, label=labels, normed=True)
plt.plot(x, stats.norm.pdf(x, mean, std), label='Assumed distribution of $\epsilon_r$')
plt.title('Histogram of residual error ($\epsilon_r$)', fontsize=18)
plt.xlabel('Residual error ($\epsilon_r$) [kg/ha]', fontsize=18)
plt.ylabel('Probability', fontsize=18)
plt.legend()
plt.savefig(f'{PLOT_DIR}/HistEr.png',dpi=300, bbox_inches = "tight")

# %% predict error using  irrigation data

X_all_irr = kpca.transform(X_irr)
X2_irr = sm.add_constant(
    X_all_irr[:, (1, 2, 3, 4, 5, 6)])  # components cumulative to >90% variance and significant in regression model
y_pred_irr = est2.predict(X2_irr)

Yield_adj_irr = dat_irr.iloc[:, -2] + y_pred_irr

# %% evaluating benefit
survey_dat = pd.read_csv("../input_data/Data_Dennis/Data/Baseline/Final_Analysis_345_nrh_TijmenData_v4_villageEdit.csv", header=0)

Irr_exist = survey_dat["water/area_irrig"] > 0
CottonArea = survey_dat['financial_information/area_cotton']
# CropIncome=Yield_adj*CottonArea*67.*3.5
# CropIncome_irr=Yield_adj_irr*CottonArea*67.*3.5

Benefit = (Yield_adj_irr - Yield_adj) * 67. * 3.5
# Benefit=(Yield_adj_irr-Yield_adj)*CottonArea*67.*3.5
# Benefit=(dat_irr.iloc[:,-2]-dat.iloc[:,-2])*67.*3.5
# Benefit=(dat_irr.iloc[:,-2]-dat.iloc[:,-2])*CottonArea*67.*3.5

Ben_extIrr = round(Benefit[Irr_exist].mean())
Ben_noIrr = round(Benefit[~Irr_exist].mean())
IrrBenefit = np.array((Ben_extIrr, Ben_noIrr))

colors_ben = np.array(['g'] * len(IrrBenefit))
colors_ben[IrrBenefit < 0] = 'r'
x = np.array(['had irrigation', 'no irrigation'])
plt.figure(figsize=(3, 7))
bars2 = plt.bar(x, IrrBenefit, color=colors_ben)
plt.ylabel("Mean Benefit per farmer [INR/y/ha]", fontsize=16)
plt.title("Benefit", fontsize=16)
plt.text(bars2[0].get_x() + 0.2, bars2[0].get_height() + 50, bars2[0].get_height())
plt.text(bars2[1].get_x() + 0.2, bars2[1].get_height() - 300, bars2[1].get_height())

# xlocs, xlabs = plt.xticks()
# xlocs=[i for i in x]
# xlabs=[i for i in x]
# plt.xticks(xlocs, xlabs)

# for bar in bars2:
#     yval = bar.get_height()
#     plt.text(bar.get_x()+ 0.2, yval + 50, yval)

plt.savefig(f'{PLOT_DIR}/BenefitBarIrr.png',dpi=300, bbox_inches = "tight")

# %% bar plot
Village = survey_dat["General/village_name"]
Villages = Village.unique()
IrrBenefit_vil = np.zeros(len(Villages))
for i in range(len(Villages)):
    tmp = Village == Villages[i]
    IrrBenefit_vil[i] = np.round(Benefit[tmp].mean())

# for i in range(len(Villages)):
#     print('{} : {} [INR/ha/y]' .format(Villages[i],IrrBenefit_vil[i]))

colors = np.array(['g'] * len(Villages))
colors[IrrBenefit_vil < 0] = 'r'
plt.figure(figsize=(10, 20))
bars = plt.barh(Villages, IrrBenefit_vil, color=colors)
plt.ylabel("Mean Benefit per farmer [Rs/y]")
plt.title("Benefit per village")

# xlocs, xlabs = plt.xticks()
# xlocs=[i for i in Villages]
# xlabs=[i for i in Villages]
# plt.yticks(xlocs, xlabs)

# for bar in bars:
#     yval = bar.get_width()
#     plt.text(bar.get_x()+ 0.1, yval + 100, yval)

# for i, v in enumerate(np.round(IrrBenefit_vil)):
#     plt.text(xlocs[i] - 0.3, v + 1000, str(v))

plt.savefig(f'{PLOT_DIR}/BenefitBarVillage.png',dpi=300, bbox_inches = "tight")

# %% hist of yield increase
a = Yield_adj_irr - Yield_adj
# a=Yield_adj_irr[Irr_exist]-Yield_adj[Irr_exist]
# b=Yield_adj_irr[~Irr_exist]-Yield_adj[~Irr_exist]

# mean = 0
# std = 25
# # mean,std=norm.fit(a)
# x = np.linspace(mean - 3*std, mean + 3*std, 100)

bins = np.linspace(-200, 200, 100)
labels = ['Yield Increase']
labels = ['Farmers with existing irrigation', 'Farmers without prior irrigation']

plt.figure(figsize=(12, 7))
plt.hist(a, bins, rwidth=0.8, label=labels, normed=True)
# plt.plot(x, stats.norm.pdf(x, mean, std), label='Assumed distribution of yield increase')
plt.title('Histogram of yield increase', fontsize=18)
# plt.title('Histogram of yield increase (farmers with existing irrigation)',fontsize=18)
# plt.title('Histogram of yield increase (farmers with no prior irrigation)',fontsize=18)
plt.xlabel('Yield increase [kg/ha]', fontsize=18)
plt.ylabel('Probability', fontsize=18)
plt.legend()

plt.savefig(f'{PLOT_DIR}/histYieldIncrease_existingIrr_noIrr.png',dpi=300, bbox_inches = "tight")

# %% hist of yield
a = Yield_adj  # [Irr_exist]
b = Yield_adj_irr  # [Irr_exist]

bins = np.linspace(0, 5000, 50)
labels = ['Old yield', 'Yield with new pond']

plt.figure(figsize=(12, 7))
plt.hist([a, b], bins, rwidth=0.8, label=labels, normed=True)
plt.title('Histogram of yield', fontsize=18)
# plt.title('Histogram of yield (farmers with existing irrigation)',fontsize=18)
plt.xlabel('Yield [kg/ha]', fontsize=18)
plt.ylabel('Probability', fontsize=18)
plt.legend()

plt.savefig(f'{PLOT_DIR}/histYield_existingIrr.png',dpi=300, bbox_inches = "tight")
# %%
Xplot = Yield_adj_irr
Xplot2 = Yield_adj
Yplot = dat_irr.iloc[:, -1]

plt.figure(figsize=(10, 8))
plt.fill_between(np.unique(Xplot), np.unique(estPlot2.predict(sm.add_constant(Xplot))) + 450,
                 np.unique(estPlot2.predict(sm.add_constant(Xplot))) - 450, color='yellow', alpha='0.5')
plt.scatter(Xplot, Yplot, c="red", s=20, edgecolor='k', label='Predicted yield with new ponds')
plt.scatter(Xplot2, Yplot, c="blue", s=20, edgecolor='k', label='Predicted yield')
Xplot2 = sm.add_constant(Xplot)
estPlot = sm.OLS(Yplot, Xplot2)
estPlot2 = estPlot.fit()
betaPlot = estPlot2.params
adj_r2Plot = round(estPlot2.rsquared_adj, 3)
plt.text(300, 4000, '$r^2$ = {}'.format(adj_r2Plot), ha='left', va='center', fontsize=18)
# plt.errorbar(Xplot, Yplot,  0, 450, fmt='r^',label='Annual yield per farmer', elinewidth =0.5,
#               marker='x', markersize='5',markeredgecolor='blue', ecolor=['red'],barsabove=False)
plt.plot(np.arange(0, 5000), np.arange(0, 5000), color="black")
plt.plot(np.unique(Xplot), np.unique(estPlot2.predict(sm.add_constant(Xplot))), color="black", linestyle='--')
plt.xlim(0, 5000)
plt.ylim(0, 5000)
plt.title("Predicted Yield vs. Observed Yield", fontsize=18)
plt.xlabel("Predicted Yield [kg/ha]", fontsize=18)
plt.ylabel("Observed Yield [kg/ha]", fontsize=18)
plt.legend()

# %% linear regressions of pc results from spss
SPSS_dat = pd.read_excel("PCs_TotalError.xlsx", header=0)

plotfit = np.arange(-2.2, 3.5, 0.5)

namePC = 'PC3'
Xplot = np.array(SPSS_dat[namePC])
Yplot = np.array(SPSS_dat.iloc[:, -1])

plt.figure(figsize=(10, 8))
plt.scatter(Xplot, Yplot, c="red", s=20, edgecolor='k')
# Xplot2 = sm.add_constant(Xplot)
# estPlot = sm.OLS(Yplot, Xplot2)
# estPlot2=estPlot.fit()
# adj_r2Plot=round(estPlot2.rsquared_adj,3)

fit = np.poly1d(np.polyfit(Xplot, Yplot, 1))
model = LinearRegression().fit(Xplot.reshape(-1, 1), Yplot.reshape(-1, 1))
adj_r2Plot = round(model.score(Xplot.reshape(-1, 1), Yplot.reshape(-1, 1)), 3)

plt.text(min(Xplot) + 6, max(Yplot), '$r^2$ = {}'.format(adj_r2Plot), ha='left', va='center', fontsize=18)
# plt.plot(np.unique(Xplot), np.unique(estPlot2.predict(sm.add_constant(Xplot))), color="black",linestyle='--')
plt.plot(plotfit, fit(plotfit), color="black", linestyle='--', linewidth=2)
plt.xlim(-3.5, 6)
# plt.ylim(0,5000)
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.title("Total Error by Perceived Fertilizer Cost, Pesticide Cost, and Crop Price ({})".format(namePC), fontsize=18)
plt.xlabel("Perceived Fertilizer Cost, Pesticide Cost, and Crop Price ({})".format(namePC), fontsize=18)
plt.ylabel("Total Error [kg/ha]", fontsize=18)

plt.savefig(f'{PLOT_DIR}/Regression{namePC}.png',dpi=300, bbox_inches = "tight")
