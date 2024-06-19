# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:40:24 2020

@author: denni
"""

import os
# from scipy import stats
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from netCDF4 import Dataset
from numpy import genfromtxt
# import statsmodels.discrete.discrete_model as sm
from sklearn.linear_model import LinearRegression

from src.sh_model.globalparameters import Original, Adjusted

# from sklearn.linear_model import LogisticRegression

# os.environ["PROJ_LIB"] = "C:\\Users\\denni\\Anaconda3\\Library\\share";

pd.set_option('mode.chained_assignment', None)

float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})
plt.rcParams.update({'font.size': 12})

# to do sensitivity analysis change to 1, otherwise 0. same with bootstrap for prec and evap
# cant have both at 1 at the same time or it wont run any calculation
Sens_bool = 0
Bootstrap_bool = 0
Range_bool = 0

# %% Updated Precipitation and Temperature data
print('Importing Temperature Data')
T_max = pd.read_excel("../../input_data/Temperature/Max_Maharastra_1975-2019IMDData.xls", header=0, parse_dates=True,
                      skiprows=[1, 2])
T_min = pd.read_excel("../../input_data/Temperature/MINTemp1975_2019IMDData.xls", header=0, parse_dates=True, skiprows=[1, 2])
T_mean = pd.read_excel("../../input_data/Temperature/MEAN_Maharastra_1975_2013IMDData.xls", header=0, parse_dates=True,
                       skiprows=[1, 2])
T_coor = pd.read_excel("../../input_data/Temperature/T_coor.xlsx", header=0, index_col=0)

T_max.loc[:, 'datetime'] = pd.to_datetime(T_max[['Year', 'Month', 'Day']])
T_max.index = T_max['datetime']
T_min.loc[:, 'datetime'] = pd.to_datetime(T_min[['Year', 'Month', 'Day']])
T_min.index = T_min['datetime']
T_mean.loc[:, 'datetime'] = pd.to_datetime(T_mean[['Year', 'Month', 'Day']])
T_mean.index = T_mean['datetime']

T_max = T_max.drop(columns=['Year', 'Month', 'Day', 'datetime'])
T_min = T_min.drop(columns=['Year', 'Month', 'Day', 'datetime'])
T_mean = T_mean.drop(columns=['Year', 'Month', 'Day', 'datetime'])

# Dennis: T_min on this timestamp is weird so i just replaced it with the average between the 2 dats before and after
T_min.loc[pd.Timestamp('2019-01-11 00:00:00')] = (T_min.loc[pd.Timestamp('2019-01-10 00:00:00')] + T_min.loc[
    pd.Timestamp('2019-01-12 00:00:00')]) / 2

# Dennis: Ra values taken from:
# http://www.zohrabsamani.com/research_material/files/Hargreaves-samani.pdf
Ra = pd.read_excel("../../input_data/Temperature/Ra.xlsx", header=0, index_col=0, parse_dates=True)

Ra_calc = np.zeros(len(T_max))
Ra_month = T_max.index.month
for i in range(len(T_max)):
    Ra_calc[i] = Ra.loc[Ra_month[i]]

Ra_calc = np.reshape(Ra_calc, (-1, 1))

for i in range(np.shape(T_max)[1] - 1):
    Ra_calc = np.insert(Ra_calc, 1, Ra_calc[:, 0], axis=1)

# Dennis: tmean is only up to 2013 right now i just used the mid point, values are close enough though not accurate
T_mean_est = (T_max + T_min) / 2
# Dennis: Hargreaves eq to estimate potential evaporation taken from:
# https://www.researchgate.net/publication/247373660_Reference_Crop_Evapotranspiration_From_Temperature
ET0 = 0.0023 * Ra_calc * (T_mean_est + 17.8) * np.sqrt(T_max - T_min)
ET_coor = T_coor

# Precipitation
prec = pd.read_excel('../../input_data/Precipitation/Maharashtra_1975_2019.xls', sheet_name='Prec', header=0, parse_dates=True,
                     skiprows=[1, 2])
prec_coor = pd.read_excel('../../input_data/Precipitation/Maharashtra_1975_2019.xls', sheet_name='Coor', header=0, index_col=0)

prec.loc[:, 'datetime'] = pd.to_datetime(prec[['Year', 'Month', 'Day']])
prec.index = prec['datetime']
prec = prec.drop(columns=['Year', 'Month', 'Day', 'datetime'])

start_year = 2010
Tsimul = 9
years = np.linspace(start_year, start_year + Tsimul - 1, Tsimul)

# %%
################################################### PLOTS FOR Nagpur ##################################################
print('Importing Parameters and Survey Data')

# cotton
from src.sh_model.Householdmodel import Householdmodel

# Importing Constants and Parameters
Constants1 = Original.copy()
Parameters1 = Adjusted.copy()

# Importing Soil depth and coordinates
extra_data = pd.read_csv('../../input_data/Other/rainfall_soiltype.csv', usecols=[0, 1, 2, 3, 4, 5, 1473])

# soil depth, note to self change the var name
soil_depth = extra_data['storagedepth_mm_4326'].values
Smax = extra_data['storagedepth_mm_4326'].values * 0.5

WD = pd.read_csv('../../input_data/Other/Water_demand.csv', sep=';', header=0, names=None)
Kc_grass = 0.75 * np.zeros(366)  # Grass Kc = 0.75
Kc_cotton = Kc_grass
Kc_cotton[157:187] = .35
Kc_cotton[187:237] = np.linspace(.35, 1.2, 50)
Kc_cotton[237:292] = 1.2
Kc_cotton[292:337] = np.linspace(1.2, .7, 45)

# Import prices
prices = pd.read_excel('../../input_data/PRICES/Worldbank_IntlCommodityPrices.xlsx', sheet_name='Annual Prices (Real)',
                       names=['Year', 'Cotton (USD/kg)', 'Phosphate (USD/mt)', 'DAP (USD/mt)', 'TSP (USD/mt)',
                              'Urea (USD/mt)',
                              'KCl (USD/mt)', 'Exchange Rate (USD/INR)'])
prices['Cotton (INR/kg)'] = prices['Cotton (USD/kg)'] * prices['Exchange Rate (USD/INR)']
prices['Phosphate (INR/kg)'] = prices['Phosphate (USD/mt)'] * prices['Exchange Rate (USD/INR)'] / 1000
prices['DAP (INR/kg)'] = prices['DAP (USD/mt)'] * prices['Exchange Rate (USD/INR)'] / 1000
prices['TSP (INR/kg)'] = prices['TSP (USD/mt)'] * prices['Exchange Rate (USD/INR)'] / 1000
prices['Urea (INR/kg)'] = prices['Urea (USD/mt)'] * prices['Exchange Rate (USD/INR)'] / 1000
prices['KCl (INR/kg)'] = prices['KCl (USD/mt)'] * prices['Exchange Rate (USD/INR)'] / 1000
price_cotton = prices.loc[2003:2016, 'Cotton (INR/kg)'].values
price_urea = prices.loc[2003:2016, 'Urea (INR/kg)'].values

# dennis: fix this price thing ^ the price urea used in hm model is just from the survey not the above data

# %%  ============================ Irrigation Types ===================================

# Drip irrigation
Drip = {'price': 39406,  # Rs # 55813 drip + 18000 pump + 5000 pipes /2ha.
        'fp_price': 0,  # Rs
        'upkeep': 0,  # Rs/yr
        'op_cost': 0.60,  # Rs/mm
        'interest_rate': 0.03,
        'life_time': 10.,  # years
        'irr_eff': .9}

# Sprinkler irrigation
Sprinkler = {'price': 27125,  # Rs # 31250 sprinkler + 18000 pump + 5000 pipes /2ha.
             'fp_price': 0,  # Rs
             'upkeep': 0,  # Rs/yr
             'op_cost': 0.60,  # Rs/mm
             'interest_rate': 0.03,
             'life_time': 10.,  # years
             'irr_eff': .7}

# Flood irrigation + Farm pond
Furrow = {'price': 11500,  # Rs # 18000 pump + 5000 pipes /2ha.
          'fp_price': 0,  # Rs
          'upkeep': 0,  # Rs/yr
          'op_cost': 0.60,  # Rs/mm
          'interest_rate': 0.03,
          'life_time': 10.,  # years
          'irr_eff': .6}

# Drip irrigation + Farm pond
Drip_fp100 = {'price': 39406,  # Rs # 55813 drip + 18000 pump + 5000 pipes /2ha.
              'fp_price': 105000,  # Rs
              'upkeep': 0,  # Rs/yr
              'op_cost': 0.60,  # Rs/mm
              'interest_rate': 0.03,
              'life_time': 10.,  # years
              'irr_eff': .9}

# Sprinkler irrigation + Farm pond
Sprinkler_fp100 = {'price': 27125,  # Rs # 31250 sprinkler + 18000 pump + 5000 pipes /2ha.
                   'fp_price': 105000,  # Rs
                   'upkeep': 0,  # Rs/yr
                   'op_cost': 0.60,  # Rs/mm
                   'interest_rate': 0.03,
                   'life_time': 10.,  # years
                   'irr_eff': .7}

# Flood irrigation
Furrow_fp100 = {'price': 11500,  # Rs # 18000 pump + 5000 pipes /2ha.
                'fp_price': 105000,  # Rs
                'upkeep': 0,  # Rs/yr
                'op_cost': 0.60,  # Rs/mm
                'interest_rate': 0.03,
                'life_time': 10.,  # years
                'irr_eff': .6}
# Drip irrigation + Farm pond
Drip_fp200 = {'price': 39406,  # Rs # 55813 drip + 18000 pump + 5000 pipes /2ha.
              'fp_price': 210000,  # Rs
              'upkeep': 0,  # Rs/yr
              'op_cost': 0.60,  # Rs/mm
              'interest_rate': 0.03,
              'life_time': 10.,  # years
              'irr_eff': .9}

# Sprinkler irrigation + Farm pond
Sprinkler_fp200 = {'price': 27125,  # Rs # 31250 sprinkler + 18000 pump + 5000 pipes /2ha.
                   'fp_price': 210000,  # Rs
                   'upkeep': 0,  # Rs/yr
                   'op_cost': 0.60,  # Rs/mm
                   'interest_rate': 0.03,
                   'life_time': 10.,  # years
                   'irr_eff': .7}

# Flood irrigation
Furrow_fp200 = {'price': 11500,  # Rs # 18000 pump + 5000 pipes /2ha.
                'fp_price': 210000,  # Rs
                'upkeep': 0,  # Rs/yr
                'op_cost': 0.60,  # Rs/mm
                'interest_rate': 0.03,
                'life_time': 10.,  # years
                'irr_eff': .6}
# Drip irrigation + Farm pond
Drip_fp500 = {'price': 39406,  # Rs # 55813 drip + 18000 pump + 5000 pipes /2ha.
              'fp_price': 525000,  # Rs
              'upkeep': 0,  # Rs/yr
              'op_cost': 0.60,  # Rs/mm
              'interest_rate': 0.03,
              'life_time': 10.,  # years
              'irr_eff': .9}

# Sprinkler irrigation + Farm pond
Sprinkler_fp500 = {'price': 27125,  # Rs # 31250 sprinkler + 18000 pump + 5000 pipes /2ha.
                   'fp_price': 525000,  # Rs
                   'upkeep': 0,  # Rs/yr
                   'op_cost': 0.60,  # Rs/mm
                   'interest_rate': 0.03,
                   'life_time': 10.,  # years
                   'irr_eff': .7}

# Flood irrigation
Furrow_fp500 = {'price': 11500,  # Rs # 18000 pump + 5000 pipes /2ha.
                'fp_price': 525000,  # Rs
                'upkeep': 0,  # Rs/yr
                'op_cost': 0.60,  # Rs/mm
                'interest_rate': 0.03,
                'life_time': 10.,  # years
                'irr_eff': .6}

# %% ============================ Getting Survey Data ===========================
# old_survey=pd.read_csv("indat/Final_Analysis_345_nrh_TijmenData_v4.csv")
raw_survey = pd.read_csv("../../input_data/Baseline/Final_Analysis_345_nrh_TijmenData_v4.csv")
raw_survey_original = pd.read_csv("../../input_data/Baseline/Final_Analysis_345_nrh_TijmenData_v4.csv")

mod_survey = pd.DataFrame()
mod_survey['Res No'] = raw_survey['Reservoir No']
mod_survey['Res Vol'] = raw_survey['Res Vol']
mod_survey['Res Area'] = raw_survey['Res Area']
mod_survey["Village"] = raw_survey["General/village_name"]
mod_survey["Lat"] = raw_survey["General/GPS_lat"]
mod_survey["Long"] = raw_survey["General/GPS_long"]
mod_survey["Family size"] = raw_survey["General/dependents"] + 1
mod_survey["Laborers"] = raw_survey["General/children_help"] + 1
mod_survey["Total area"] = raw_survey["financial_information/area_total"]  # /2.4711
mod_survey["Cott area"] = raw_survey["financial_information/area_total"] * 0.8
mod_survey["Other area"] = raw_survey["financial_information/area_total"] - mod_survey["Cott area"]
mod_survey["Livestock"] = raw_survey["financial_information/cows"] + raw_survey["financial_information/bulls"] + 0.100 * \
                          raw_survey["financial_information/goats"]
mod_survey["Debt"] = raw_survey["financial_information/loan_debt"]
mod_survey["Interest"] = raw_survey["financial_information/interest_rate"]
mod_survey["Other income"] = raw_survey["addtl_income_total"]
mod_survey["Seed price"] = raw_survey["price_cotton_factors/seeds_package_cost"] * raw_survey[
    'price_cotton_factors/seeds_use'] * 2.4711
mod_survey["Cott price"] = raw_survey["price_cotton_factors/crop_selling_price"] / 100.
mod_survey["Yield"] = raw_survey["price_cotton_factors/cotton_yield"] * 2.4711 * 100
mod_survey["Open well"] = raw_survey["Open_Well"] == 1.0
mod_survey["Irr frac"] = np.minimum(1, raw_survey["water/area_irrig"] / mod_survey["Cott area"])
mod_survey["Irr frac"] = mod_survey["Irr frac"].fillna(0)
mod_survey["Tech"] = None
mod_survey["Tech"][raw_survey["water/irrigation_technology/furrow"] == True] = 'furrow'
mod_survey["Tech"][raw_survey["water/irrigation_technology/sprinklers"] == True] = 'sprinkler'
mod_survey["Tech"][raw_survey["water/irrigation_technology/micro"] == True] = 'drip'
mod_survey["Fert kg"] = raw_survey["fert_pest/fertilizer_amount"] * 2.4711
mod_survey["Fert extra"] = raw_survey["fert_pest/fertilizers_cost"] * 2.4711
mod_survey["Fert price"] = raw_survey["fert_cost_per_kg"]
mod_survey["Fert price"].fillna(0)
mod_survey["Pesticides"] = raw_survey["fert_pest/pesticide_cost"] * 2.4711
mod_survey["Pesticides"] = mod_survey["Pesticides"].fillna(0)
mod_survey['a_c'] = raw_survey['Basin/Farmer Area ']
mod_survey["PondID"] = np.zeros(len(raw_survey))
mod_survey["DistPond"] = np.zeros(len(raw_survey))
mod_survey['radius'] = np.zeros(len(raw_survey))
mod_survey['Pond Vol Initial'] = np.zeros(len(raw_survey))

# %% Finding closest res intake
daily_bool = 0
if daily_bool == 1:
    print('Importing Reservoirs Data: Input data is daily')
elif daily_bool == 0:
    print('Importing Reservoirs Data: Input data is monthly')

if daily_bool == 1:
    res = pd.read_excel('../../input_data/Reservoirs/rsv_D_1979-2014.xlsx', header=0, sep=",")
    res_coor = pd.read_csv('../../input_data/Reservoirs/reservoir locations solidaridad.csv', header=0, sep=",")

    ResID = np.arange(res['RES'].nunique())

    mod_survey['Res Intake'] = np.zeros(len(mod_survey))
    mod_survey['Dist'] = np.zeros(len(mod_survey))
    for i in range(len(mod_survey)):
        # Finding closest coordinate for irrigation reservoirs
        DisMat = np.sqrt((mod_survey['Lat'][i] - res_coor["Y"]) ** 2 + (mod_survey['Long'][i] - res_coor["X"]) ** 2)
        # finding index of min distance
        res_i = DisMat.idxmin()
        # finding coord of min distance
        DistPond = DisMat[res_i] * 105  # 105 is conversion from degrees to km around lat ~20 degrees north

        mod_survey['Res Intake'][i] = res_i + 1
        mod_survey['Dist'][i] = DistPond  # in km

    res_intake = mod_survey['Res Intake'].value_counts()

    mod_survey['Res Count'] = np.zeros(len(mod_survey))
    for i in range(len(mod_survey)):
        mod_survey['Res Count'].iloc[i] = res_intake[mod_survey['Res Intake'][i]]
else:
    res = pd.read_excel('../../input_data/Reservoirs/Ghatanji_350_M_1981_2019.xlsx', header=0)
    res_coor = pd.read_csv('../../input_data/Reservoirs/Ghatanji_350_M_Coord.csv', header=0, sep=",")

    ResID = np.arange(res['RES'].nunique())

    mod_survey['Res Intake'] = np.zeros(len(mod_survey))
    mod_survey['Dist'] = np.zeros(len(mod_survey))
    for i in range(len(mod_survey)):
        # Finding closest coordinate for irrigation reservoirs
        DisMat = np.sqrt(
            (mod_survey['Lat'][i] - res_coor["ycoord"]) ** 2 + (mod_survey['Long'][i] - res_coor["xcoord"]) ** 2)
        # finding index of min distance
        res_i = DisMat.idxmin()
        # finding coord of min distance
        DistPond = DisMat[res_i] * 105  # 105 is conversion from degrees to km around lat ~20 degrees north

        mod_survey['Res Intake'][i] = res_i + 1
        mod_survey['Dist'][i] = DistPond  # in km

    res_intake = mod_survey['Res Intake'].value_counts()

    mod_survey['Res Count'] = np.zeros(len(mod_survey))
    for i in range(len(mod_survey)):
        mod_survey['Res Count'].iloc[i] = res_intake[mod_survey['Res Intake'][i]]

    res.index = pd.to_datetime(res['YYYYMM'], format='%Y%m')

# %% ============================== Household ==================================
if Sens_bool == 0 and Bootstrap_bool == 0 and Range_bool == 0:
    print('Running Household Model: %d households' % len(mod_survey))
    Yield1 = np.zeros((len(mod_survey), Tsimul))
    Yield14 = np.zeros((len(mod_survey), Tsimul))
    VarMat1, WbMat1 = np.zeros((len(mod_survey), Tsimul + 1, 5)), np.zeros((len(mod_survey), Tsimul, 3))
    VarMat14, WbMat14 = np.zeros((len(mod_survey), Tsimul + 1, 5)), np.zeros((len(mod_survey), Tsimul, 3))
    Coords = np.zeros((len(mod_survey), 2))

    # Dennis: preparing matrix to get daily parameters from WB_yield
    Yieldmatlen = len(prec[(prec.index.year >= start_year) & (prec.index.year <= start_year + Tsimul - 1)])
    YieldMat = np.zeros((len(mod_survey), Yieldmatlen, 27))
    YieldMat14 = np.zeros((len(mod_survey), Yieldmatlen, 27))
    prec_id = []
    evap_id = []
    T_id = []

    for i in range(len(mod_survey.index)):
        # Finding closest coordinate for precipication and evaporation
        DisMat = np.sqrt(
            (prec_coor.loc['Lat'] - mod_survey["Lat"][i]) ** 2 + (prec_coor.loc['Long'] - mod_survey["Long"][i]) ** 2)

        # finding index of min prec
        prec_i = DisMat.idxmin()
        prec_id.append(prec_i)
        # finding coord of min prec
        Coords[i] = (extra_data['Lat'][prec_i], extra_data['Long'][prec_i])
        DisMat1 = np.sqrt(
            (ET_coor.loc['Lat'] - mod_survey["Lat"][i]) ** 2 + (ET_coor.loc['Long'] - mod_survey["Long"][i]) ** 2)
        # finding index of min evap
        evap_i = int(DisMat1.idxmin())
        evap_id.append(evap_i)

        DisMat2 = np.sqrt(
            (T_coor.loc['Lat'] - mod_survey["Lat"][i]) ** 2 + (T_coor.loc['Long'] - mod_survey["Long"][i]) ** 2)
        # finding index of min evap
        T_i = int(DisMat2.idxmin())
        T_id.append(T_i)

    # soil_depth_spss=[]
    for i in range(len(mod_survey.index)):

        Price_fert = np.full(Tsimul, mod_survey["Fert price"][i])
        Price_crop = np.full(Tsimul, mod_survey["Cott price"][i])

        if mod_survey["Tech"][i] == 'drip':
            irr_system = Drip
        elif mod_survey["Tech"][i] == 'sprinkler':
            irr_system = Sprinkler
        elif mod_survey["Tech"][i] == 'furrow':
            irr_system = Furrow
        else:
            irr_system = None
        Constants1['Price_for_chemicals'] = mod_survey['Pesticides'][i]
        Constants1['Price_of_seeds'] = mod_survey['Seed price'][i]
        # add_water = sum(prec.iloc[2:,prec_i])/(max(prec.iloc[2:].index.year) - min(prec.iloc[2:].index.year)) * mod_survey['a_c'][i]

        res_tmp = res[res['RES'] == mod_survey['Res Intake'][i]]
        res_max = mod_survey['Res Vol'].iloc[i]
        res_area = mod_survey['Res Area'].iloc[i]
        res_basin = mod_survey['a_c'][i]
        res_count = mod_survey['Res Count'][i]

        # Dennis: resampling data from monthly to daily if daily_bool==0
        if daily_bool == 0:
            idx = res_tmp.index[-1] + pd.offsets.MonthBegin(1)
            res_tmp = res_tmp.append(res.iloc[[-1]].rename({res_tmp.index[-1]: idx}))

            # resample with forward filling values, remove last helper row
            res_tmp = res_tmp.resample('D').ffill().iloc[:-1]

            # divide by size of months
            res_tmp['FLOW_INcms'] /= res_tmp.resample('MS')['FLOW_INcms'].transform('size')
        res_parameter = np.array([res_tmp, res_max, res_area, res_basin, res_count])

        total_area = mod_survey['Total area'][i]
        Constants1['soil_depth'] = soil_depth[prec_id[i]]
        # soil_depth_spss.append(soil_depth[prec_i])
        # add water: None for no reservoir, 1 for including reservoirs
        add_water = 1
        if i % 20 == 0 or i == 10:
            print("{} out of {} households".format(i, len(mod_survey)))
        iv_max = mod_survey['Res Vol'][i] * 1000 / (mod_survey["Cott area"][i] * 10000)
        VarMat1[i], WbMat1[i], Yield1[i], YieldMat[i] = Householdmodel(prec.iloc[:, prec_id[i]],
                                                                       ET0.iloc[:, evap_id[i]],
                                                                       T_mean_est.iloc[:, T_id[i]], Constants1,
                                                                       Parameters1, Kc_cotton,
                                                                       Kc_grass, Price_crop, Price_fert, start_year,
                                                                       Tsimul, res_parameter, total_area, wage=0,
                                                                       Smx=Smax[prec_id[i]], IC=100000,
                                                                       fert=mod_survey["Fert kg"][i],
                                                                       crop_area=mod_survey["Cott area"][i],
                                                                       other_area=mod_survey["Other area"][i],
                                                                       livestock=mod_survey["Livestock"][i],
                                                                       Family_size=mod_survey["Laborers"][i],
                                                                       irr_system=irr_system,
                                                                       loan_debt=mod_survey["Debt"][i],
                                                                       interest_rate=mod_survey["Interest"][i] / 100.)
        # Dennis: I think the difference between the different simulation is the iv_max
        # it is not used if using the wb_yield
        if mod_survey["Tech"][i] == 'drip':
            irr_system = Drip_fp100
        elif mod_survey["Tech"][i] == 'sprinkler':
            irr_system = Sprinkler_fp100
        else:
            irr_system = Furrow_fp100
        VarMat14[i], WbMat14[i], Yield14[i], YieldMat14[i] = Householdmodel(prec.iloc[:, prec_id[i]],
                                                                            ET0.iloc[:, evap_id[i]],
                                                                            T_mean_est.iloc[:, T_id[i]], Constants1,
                                                                            Parameters1, Kc_cotton,
                                                                            Kc_grass, Price_crop, Price_fert,
                                                                            start_year, Tsimul, res_parameter,
                                                                            total_area, wage=0, Smx=Smax[prec_id[i]],
                                                                            IC=100000,
                                                                            fert=mod_survey["Fert kg"][i],
                                                                            crop_area=mod_survey["Cott area"][i],
                                                                            other_area=mod_survey["Other area"][i],
                                                                            livestock=mod_survey["Livestock"][i],
                                                                            Family_size=mod_survey["Laborers"][i],
                                                                            irr_system=irr_system,
                                                                            loan_debt=mod_survey["Debt"][i],
                                                                            interest_rate=mod_survey["Interest"][
                                                                                              i] / 100.,
                                                                            add_water=add_water, iv_max=iv_max,
                                                                            subsidies=1.00)
    # np.savetxt('Results/'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'_precid.csv',prec_id, delimiter =',')
    # np.savetxt('Results/'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'_evapid.csv',evap_id, delimiter =',')
    # np.savetxt('Results/'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'_Tid.csv',T_id, delimiter =',')

    benefit14 = VarMat14[:, -1, 0] - VarMat1[:, -1, 0]

# %% Bootstrapping Rainfall, Evaporation, and adjusting Irrigation data
if Sens_bool == 0 and Bootstrap_bool == 1 and Range_bool == 0:
    # running bootstrap
    nbool = 1000
    Boot_Mat = np.zeros((nbool, 3))
    print('Running Bootstrap Analysis: %d households' % len(mod_survey))

    bootPrecMat = []
    bootETMat = []
    bootIrrMat = []
    Yield_Boot = []
    Yield_Boot_Res = []
    Benefit_Boot = []
    BootResult_Mat = np.zeros((nbool, 3))

    for n in range(nbool):
        if n % 20 == 0 or n == 5:
            print("{} out of {} iterations".format(n, nbool))
        bootPrec = []
        bootET = []
        # prec_years=prec.index.year.unique()
        for i in range(len(years)):
            dayCount = len(prec[prec.index.year == years[i]])
            for j in range(dayCount):
                bootTempPrec = prec[prec.index.dayofyear == j + 1].sample()
                bootTempET = ET0[ET0.index.dayofyear == j + 1].sample()
                bootPrec.append(np.squeeze(bootTempPrec.values))
                bootET.append(np.squeeze(bootTempET.values))

        bootPrecMat.append(bootPrec)
        bootETMat.append(bootET)

        precTemp = pd.DataFrame(bootPrec,
                                index=prec.index[(prec.index.year >= years[0]) & (prec.index.year <= years[-1])],
                                columns=prec.columns)
        ET0Temp = pd.DataFrame(bootET, index=ET0.index[(ET0.index.year >= years[0]) & (ET0.index.year <= years[-1])],
                               columns=ET0.columns)

        # Calculating multiplier from new prec and evap data to adjust the irrigation inflow
        MultPrec = []
        MultET = []
        for i in range(len(years)):
            _precSum = prec[prec.index.year == years[i]].sum()
            _precSum = _precSum.sum()
            _precTempSum = precTemp[precTemp.index.year == years[i]].sum()
            _precTempSum = _precTempSum.sum()
            MultPrec.append(_precTempSum / _precSum)

            _ET0Sum = ET0[ET0.index.year == years[i]].sum()
            _ET0Sum = _ET0Sum.sum()
            _ET0TempSum = ET0Temp[ET0Temp.index.year == years[i]].sum()
            _ET0TempSum = _ET0TempSum.sum()
            MultET.append(_ET0Sum / _ET0TempSum)  # the ratio is inverse of prec because more evap means less irrigation

        MultAvg = np.add(MultPrec, MultET) / 2
        Boot_Mat[n, 0] = MultPrec[-1]
        Boot_Mat[n, 1] = MultET[-1]
        Boot_Mat[n, 2] = MultAvg[-1]

        resflow_tmp = res
        for i in range(len(years)):
            resflow_tmp.FLOW_INcms[resflow_tmp.index.year == years[i]] = res.FLOW_INcms[res.index.year == years[i]] * \
                                                                         MultAvg[i]
        _bootIrrMat = resflow_tmp[['RES', 'FLOW_INcms']]
        _bootIrrMat = _bootIrrMat[_bootIrrMat.index.year >= years[0]]
        _bootIrrMat = pd.DataFrame.from_dict(_bootIrrMat.groupby('RES')['FLOW_INcms'].apply(np.unique).to_dict(),
                                             'index').T.fillna(0)

        bootIrrMat.append(_bootIrrMat.values.tolist())

        Yield1 = np.zeros((len(mod_survey), Tsimul))
        Yield14 = np.zeros((len(mod_survey), Tsimul))
        VarMat1, WbMat1 = np.zeros((len(mod_survey), Tsimul + 1, 5)), np.zeros((len(mod_survey), Tsimul, 3))
        VarMat14, WbMat14 = np.zeros((len(mod_survey), Tsimul + 1, 5)), np.zeros((len(mod_survey), Tsimul, 3))
        Coords = np.zeros((len(mod_survey), 2))

        # Dennis: preparing matrix to get daily parameters from WB_yield
        Yieldmatlen = len(
            precTemp[(precTemp.index.year >= start_year) & (precTemp.index.year <= start_year + Tsimul - 1)])
        YieldMat = np.zeros((len(mod_survey), Yieldmatlen, 27))
        YieldMat14 = np.zeros((len(mod_survey), Yieldmatlen, 27))
        prec_id = []
        evap_id = []
        T_id = []
        # soil_depth_spss=[]
        for i in range(len(mod_survey.index)):
            # for i in range(len(ResID)):
            # Finding closest coordinate for precipication and evaporation
            DisMat = np.sqrt((prec_coor.loc['Lat'] - mod_survey["Lat"][i]) ** 2 + (
                        prec_coor.loc['Long'] - mod_survey["Long"][i]) ** 2)

            # finding index of min prec
            prec_i = DisMat.idxmin()
            prec_id.append(prec_i)
            # finding coord of min prec
            Coords[i] = (extra_data['Lat'][prec_i], extra_data['Long'][prec_i])
            DisMat1 = np.sqrt(
                (ET_coor.loc['Lat'] - mod_survey["Lat"][i]) ** 2 + (ET_coor.loc['Long'] - mod_survey["Long"][i]) ** 2)
            # finding index of min evap
            evap_i = int(DisMat1.idxmin())
            evap_id.append(evap_i)

            DisMat2 = np.sqrt(
                (T_coor.loc['Lat'] - mod_survey["Lat"][i]) ** 2 + (T_coor.loc['Long'] - mod_survey["Long"][i]) ** 2)
            # finding index of min evap
            T_i = int(DisMat2.idxmin())
            T_id.append(T_i)

        for i in range(len(mod_survey.index)):
            Price_fert = np.full(Tsimul, mod_survey["Fert price"][i])
            Price_crop = np.full(Tsimul, mod_survey["Cott price"][i])

            if mod_survey["Tech"][i] == 'drip':
                irr_system = Drip
            elif mod_survey["Tech"][i] == 'sprinkler':
                irr_system = Sprinkler
            elif mod_survey["Tech"][i] == 'furrow':
                irr_system = Furrow
            else:
                irr_system = None
            Constants1['Price_for_chemicals'] = mod_survey['Pesticides'][i]
            Constants1['Price_of_seeds'] = mod_survey['Seed price'][i]

            res_tmp = resflow_tmp[resflow_tmp['RES'] == mod_survey['Res Intake'][i]]
            res_max = mod_survey['Res Vol'].iloc[i]
            res_area = mod_survey['Res Area'].iloc[i]
            res_basin = mod_survey['a_c'][i]
            res_count = mod_survey['Res Count'][i]

            # Dennis: resampling data from monthly to daily if daily_bool==0
            if daily_bool == 0:
                idx = res_tmp.index[-1] + pd.offsets.MonthBegin(1)
                res_tmp = res_tmp.append(res.iloc[[-1]].rename({res_tmp.index[-1]: idx}))

                # resample with forward filling values, remove last helper row
                res_tmp = res_tmp.resample('D').ffill().iloc[:-1]

                # divide by size of months
                res_tmp['FLOW_INcms'] /= res_tmp.resample('MS')['FLOW_INcms'].transform('size')
            res_parameter = np.array([res_tmp, res_max, res_area, res_basin, res_count])

            total_area = mod_survey['Total area'][i]
            Constants1['soil_depth'] = soil_depth[prec_id[i]]
            # soil_depth_spss.append(soil_depth[prec_i])
            # add water: None for no reservoir, 1 for including reservoirs
            add_water = 1
            iv_max = mod_survey['Res Vol'][i] * 1000 / (mod_survey["Cott area"][i] * 10000)
            VarMat1[i], WbMat1[i], Yield1[i], YieldMat[i] = Householdmodel(precTemp.iloc[:, prec_id[i]],
                                                                           ET0Temp.iloc[:, evap_id[i]],
                                                                           T_mean_est.iloc[:, T_id[i]], Constants1,
                                                                           Parameters1, Kc_cotton,
                                                                           Kc_grass, Price_crop, Price_fert, start_year,
                                                                           Tsimul, res_parameter, total_area, wage=0,
                                                                           Smx=Smax[prec_id[i]], IC=100000,
                                                                           fert=mod_survey["Fert kg"][i],
                                                                           crop_area=mod_survey["Cott area"][i],
                                                                           other_area=mod_survey["Other area"][i],
                                                                           livestock=mod_survey["Livestock"][i],
                                                                           Family_size=mod_survey["Laborers"][i],
                                                                           irr_system=irr_system,
                                                                           loan_debt=mod_survey["Debt"][i],
                                                                           interest_rate=mod_survey["Interest"][
                                                                                             i] / 100.)

            if mod_survey["Tech"][i] == 'drip':
                irr_system = Drip_fp100
            elif mod_survey["Tech"][i] == 'sprinkler':
                irr_system = Sprinkler_fp100
            else:
                irr_system = Furrow_fp100
            VarMat14[i], WbMat14[i], Yield14[i], YieldMat14[i] = Householdmodel(precTemp.iloc[:, prec_id[i]],
                                                                                ET0Temp.iloc[:, evap_id[i]],
                                                                                T_mean_est.iloc[:, T_id[i]], Constants1,
                                                                                Parameters1, Kc_cotton,
                                                                                Kc_grass, Price_crop, Price_fert,
                                                                                start_year, Tsimul, res_parameter,
                                                                                total_area, wage=0,
                                                                                Smx=Smax[prec_id[i]], IC=100000,
                                                                                fert=mod_survey["Fert kg"][i],
                                                                                crop_area=mod_survey["Cott area"][i],
                                                                                other_area=mod_survey["Other area"][i],
                                                                                livestock=mod_survey["Livestock"][i],
                                                                                Family_size=mod_survey["Laborers"][i],
                                                                                irr_system=irr_system,
                                                                                loan_debt=mod_survey["Debt"][i],
                                                                                interest_rate=mod_survey["Interest"][
                                                                                                  i] / 100.,
                                                                                add_water=add_water, iv_max=iv_max,
                                                                                subsidies=1.00)
        benefit14 = VarMat14[:, -1, 0] - VarMat1[:, -1, 0]

        Yield_Boot.append(WbMat1[:, -1, 0].tolist())
        Yield_Boot_Res.append(WbMat14[:, -1, 0].tolist())
        Benefit_Boot.append(benefit14.tolist())

        BootResult_Mat[n, 0] = np.mean(WbMat14[:, -1, 0])
        BootResult_Mat[n, 1] = np.mean(WbMat1[:, -1, 0])
        BootResult_Mat[n, 2] = np.mean(benefit14) / Tsimul

    bootPrecMat = np.array(bootPrecMat)
    names = ['IterationNumber', 'Day', 'CoordID']
    index = pd.MultiIndex.from_product([range(s) for s in bootPrecMat.shape], names=names)
    df_bootPrecMat = pd.DataFrame({'Prec_at_coordID': bootPrecMat.flatten()}, index=index)
    df_bootPrecMat = df_bootPrecMat.unstack(level='CoordID').sort_index()

    bootETMat = np.array(bootETMat)
    names = ['IterationNumber', 'Day', 'CoordID']
    index = pd.MultiIndex.from_product([range(s) for s in bootETMat.shape], names=names)
    df_bootETMat = pd.DataFrame({'ET_at_coordID': bootETMat.flatten()}, index=index)
    df_bootETMat = df_bootETMat.unstack(level='CoordID').sort_index()

    bootIrrMat = np.array(bootIrrMat)
    names = ['IterationNumber', 'Time[Month/Day]', 'Res']
    index = pd.MultiIndex.from_product([range(s) for s in bootIrrMat.shape], names=names)
    df_bootIrrMat = pd.DataFrame({'Irr_at_ResID': bootIrrMat.flatten()}, index=index)
    df_bootIrrMat = df_bootIrrMat.unstack(level='Res').sort_index()

    np.savetxt('Bootstrap/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_BootYield2018.csv',
               np.squeeze(Yield_Boot), delimiter=',')
    np.savetxt('Bootstrap/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_BootYieldRes2018.csv',
               np.squeeze(Yield_Boot_Res), delimiter=',')
    np.savetxt('Bootstrap/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_BootBenefit2018.csv',
               np.squeeze(Benefit_Boot), delimiter=',')
    np.savetxt('Bootstrap/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_BootResultMeanMat2018.csv',
               BootResult_Mat, delimiter=',')
    df_bootPrecMat.to_csv('Bootstrap/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_BootPrecDat2018.csv')
    df_bootETMat.to_csv('Bootstrap/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_BootETDat2018.csv')
    df_bootIrrMat.to_csv('Bootstrap/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_BootIrrDat2018.csv')

    sys.exit()

# %% Sensitivity Analysis
if Sens_bool == 1 and Bootstrap_bool == 0 and Range_bool == 0:
    Yield1 = np.zeros((len(mod_survey), Tsimul))
    Yield14 = np.zeros((len(mod_survey), Tsimul))
    VarMat1, WbMat1 = np.zeros((len(mod_survey), Tsimul + 1, 5)), np.zeros((len(mod_survey), Tsimul, 3))
    VarMat14, WbMat14 = np.zeros((len(mod_survey), Tsimul + 1, 5)), np.zeros((len(mod_survey), Tsimul, 3))
    Coords = np.zeros((len(mod_survey), 2))

    # Dennis: preparing matrix to get daily parameters from WB_yield
    # Dennis: 11 is the number of par, change it according to the number of par from variable Mat in wb_yield
    Yieldmatlen = len(prec[(prec.index.year >= start_year) & (prec.index.year <= start_year + Tsimul - 1)])
    YieldMat = np.zeros((len(mod_survey), Yieldmatlen, 27))

    # these index identification is on a separate loop to save time in the MCS
    prec_id = []
    evap_id = []
    T_id = []
    for i in range(len(mod_survey.index)):
        # Finding closest coordinate for precipication and evaporation
        DisMat = np.sqrt(
            (prec_coor.loc['Lat'] - mod_survey["Lat"][i]) ** 2 + (prec_coor.loc['Long'] - mod_survey["Long"][i]) ** 2)

        # finding index of min prec
        prec_i = DisMat.idxmin()
        prec_id.append(prec_i)
        # finding coord of min prec
        Coords[i] = (extra_data['Lat'][prec_i], extra_data['Long'][prec_i])
        DisMat1 = np.sqrt(
            (ET_coor.loc['Lat'] - mod_survey["Lat"][i]) ** 2 + (ET_coor.loc['Long'] - mod_survey["Long"][i]) ** 2)
        # finding index of min evap
        evap_i = int(DisMat1.idxmin())
        evap_id.append(evap_i)

        DisMat2 = np.sqrt(
            (T_coor.loc['Lat'] - mod_survey["Lat"][i]) ** 2 + (T_coor.loc['Long'] - mod_survey["Long"][i]) ** 2)
        # finding index of min evap
        T_i = int(DisMat2.idxmin())
        T_id.append(T_i)

    nmax = 10000
    #                   CCo    CGC   CCx   CDC   HI   tCCo  CWP  FC   WP  Pup  Plow fs   fsz  fsf  fsl
    # ParMin = np.array([0.003,  0.09, 0.9,  .03,  .25,  7,   15,  .3, .1, .6,  .1,   .01, -.01, -.01, -.01])
    # ParMax = np.array([0.0105, 0.12, 0.99, .045, .4,   30,  18,  .5, .2, .8,  .3,   5,   -8,  -5,   -5])
    #                  HI   tCCo fs   fsz  fsf  fsl
    # ParMin = np.array([.25,  7,  .01, -.01, .01, -.01])
    # ParMax = np.array([.4,   30,  5,   -8,  8,   -8])
    #                  HI   tCCo  fsl
    ParMin = np.array([.25, 7, -.01])
    ParMax = np.array([.4, 30, -10])
    # Sens_Mat = np.zeros((nmax,len(ParMin)+3))
    #                  T_ts_ini     T_ss_ini
    # ParMin = np.array([0,           0])
    # ParMax = np.array([50,          80])
    Sens_Mat = np.zeros((nmax, len(ParMin) + 4))
    Yield_MCS = []
    avg_survey = mod_survey["Yield"].mean()
    print('Running Sensitivity Analysis: %d households' % len(mod_survey))
    for n in range(nmax):
        Rnum = np.random.rand(len(ParMin))
        Par = ParMin + Rnum * (ParMax - ParMin)
        # Par=OptPar
        if n % 20 == 0 or n == 5:
            print("{} out of {} iterations".format(n, nmax))
        for i in range(len(mod_survey.index)):
            Price_fert = np.full(Tsimul, mod_survey["Fert price"][i])
            Price_crop = np.full(Tsimul, mod_survey["Cott price"][i])

            if mod_survey["Tech"][i] == 'drip':
                irr_system = Drip
            elif mod_survey["Tech"][i] == 'sprinkler':
                irr_system = Sprinkler
            elif mod_survey["Tech"][i] == 'furrow':
                irr_system = Furrow
            else:
                irr_system = None
            Constants1['Price_for_chemicals'] = mod_survey['Pesticides'][i]
            Constants1['Price_of_seeds'] = mod_survey['Seed price'][i]
            # add_water = sum(prec.iloc[2:,prec_i])/(max(prec.iloc[2:].index.year) - min(prec.iloc[2:].index.year)) * mod_survey['a_c'][i]

            res_tmp = res[res['RES'] == mod_survey['Res Intake'][i]]
            res_max = mod_survey['Res Vol'].iloc[i]
            res_area = mod_survey['Res Area'].iloc[i]
            res_basin = mod_survey['a_c'][i]
            res_count = mod_survey['Res Count'][i]

            # Dennis: resampling data from monthly to daily if daily_bool==0
            if daily_bool == 0:
                idx = res_tmp.index[-1] + pd.offsets.MonthBegin(1)
                res_tmp = res_tmp.append(res.iloc[[-1]].rename({res_tmp.index[-1]: idx}))

                # resample with forward filling values, remove last helper row
                res_tmp = res_tmp.resample('D').ffill().iloc[:-1]

                # divide by size of months
                res_tmp['FLOW_INcms'] /= res_tmp.resample('MS')['FLOW_INcms'].transform('size')
            res_parameter = np.array([res_tmp, res_max, res_area, res_basin, res_count])

            total_area = mod_survey['Total area'][i]
            Constants1['soil_depth'] = soil_depth[prec_id[i]]
            # add water: None for no reservoir, any input for including reservoirs
            add_water = 1

            iv_max = mod_survey['Res Vol'][i] * 1000 / (mod_survey["Cott area"][i] * 10000)
            VarMat1[i], WbMat1[i], Yield1[i], YieldMat[i] = Householdmodel(prec.iloc[:, prec_id[i]],
                                                                           ET0.iloc[:, evap_id[i]],
                                                                           T_mean_est.iloc[:, T_id[i]], Constants1,
                                                                           Parameters1, Kc_cotton,
                                                                           Kc_grass, Price_crop, Price_fert, start_year,
                                                                           Tsimul, res_parameter, total_area, wage=0,
                                                                           Smx=Smax[prec_id[i]], IC=100000,
                                                                           fert=mod_survey["Fert kg"][i],
                                                                           crop_area=mod_survey["Cott area"][i],
                                                                           other_area=mod_survey["Other area"][i],
                                                                           livestock=mod_survey["Livestock"][i],
                                                                           Family_size=mod_survey["Laborers"][i],
                                                                           irr_system=irr_system,
                                                                           loan_debt=mod_survey["Debt"][i],
                                                                           interest_rate=mod_survey["Interest"][
                                                                                             i] / 100., Par=Par)

        Yield_MCS.append(WbMat1[:, -1, 0].tolist())

        Sens_Mat[n, 0:len(Par)] = Par

        model = LinearRegression().fit(np.array(WbMat1[:, -1, 0]).reshape(-1, 1),
                                       np.array(mod_survey["Yield"]).reshape(-1, 1))
        Sens_Mat[n, -4] = model.score(np.array(WbMat1[:, -1, 0]).reshape(-1, 1),
                                      np.array(mod_survey["Yield"]).reshape(-1, 1))
        # MAE
        Sens_Mat[n, -3] = np.sum(abs(WbMat1[:, -1, 0] - mod_survey["Yield"])) / len(WbMat1[:, -1, 0])
        # NS log
        Sens_Mat[n, -2] = 1 - (np.sum((np.log10(WbMat1[:, -1, 0]) - np.log10(mod_survey["Yield"])) ** 2) / np.sum(
            (np.log10(avg_survey) - np.log10(mod_survey["Yield"])) ** 2))
        # NS
        Sens_Mat[n, -1] = 1 - (np.sum((WbMat1[:, -1, 0] - mod_survey["Yield"]) ** 2) / np.sum(
            (avg_survey - mod_survey["Yield"]) ** 2))

    Yield_MCS = np.squeeze(Yield_MCS)
    np.savetxt('MCS/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_SensitivityResults_r2_MAE_NS_NSlog.csv',
               Sens_Mat, delimiter=',')
    np.savetxt('MCS/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_YieldMCS.csv', Yield_MCS, delimiter=',')

# %% Ranges
if Sens_bool == 0 and Bootstrap_bool == 0 and Range_bool == 1:
    Sens_Mat = genfromtxt("MCS/20210326_SensitivityResults_r2_MAE_NS_NSlog_Combined10000.csv", delimiter=',')
    Sens_Mat = Sens_Mat[~np.all(Sens_Mat == 0, axis=1)]

    OptPar_Range = Sens_Mat[
        (Sens_Mat[:, -4] > 0.003) & (Sens_Mat[:, -3] < 700) & (Sens_Mat[:, -2] > -2) & (Sens_Mat[:, -1] > -1)]

    Yield1 = np.zeros((len(mod_survey), Tsimul))
    Yield14 = np.zeros((len(mod_survey), Tsimul))
    VarMat1, WbMat1 = np.zeros((len(mod_survey), Tsimul + 1, 5)), np.zeros((len(mod_survey), Tsimul, 3))
    VarMat14, WbMat14 = np.zeros((len(mod_survey), Tsimul + 1, 5)), np.zeros((len(mod_survey), Tsimul, 3))
    Coords = np.zeros((len(mod_survey), 2))

    # Dennis: preparing matrix to get daily parameters from WB_yield
    Yieldmatlen = len(prec[(prec.index.year >= start_year) & (prec.index.year <= start_year + Tsimul - 1)])
    YieldMat = np.zeros((len(mod_survey), Yieldmatlen, 27))
    YieldMat14 = np.zeros((len(mod_survey), Yieldmatlen, 27))

    # these index identification is on a separate loop to save time in the MCS
    prec_id = []
    evap_id = []
    T_id = []
    for i in range(len(mod_survey.index)):
        # Finding closest coordinate for precipication and evaporation
        DisMat = np.sqrt(
            (prec_coor.loc['Lat'] - mod_survey["Lat"][i]) ** 2 + (prec_coor.loc['Long'] - mod_survey["Long"][i]) ** 2)

        # finding index of min prec
        prec_i = DisMat.idxmin()
        prec_id.append(prec_i)
        # finding coord of min prec
        Coords[i] = (extra_data['Lat'][prec_i], extra_data['Long'][prec_i])
        DisMat1 = np.sqrt(
            (ET_coor.loc['Lat'] - mod_survey["Lat"][i]) ** 2 + (ET_coor.loc['Long'] - mod_survey["Long"][i]) ** 2)
        # finding index of min evap
        evap_i = int(DisMat1.idxmin())
        evap_id.append(evap_i)

        DisMat2 = np.sqrt(
            (T_coor.loc['Lat'] - mod_survey["Lat"][i]) ** 2 + (T_coor.loc['Long'] - mod_survey["Long"][i]) ** 2)
        # finding index of min evap
        T_i = int(DisMat2.idxmin())
        T_id.append(T_i)

    nmax = len(OptPar_Range)

    Yield_Ranges = []
    Yield_Ranges_Res = []
    Benefit_Ranges = []
    avg_survey = mod_survey["Yield"].mean()
    Ranges_Mat = np.zeros((nmax, 7))
    print('Finding Ranges of Yield and Benefit: %d households' % len(mod_survey))
    for n in range(nmax):
        Par = OptPar_Range[n, :4]
        if n % 20 == 0 or n == 5:
            print("{} out of {} parameter sets".format(n, nmax))
        for i in range(len(mod_survey.index)):
            Price_fert = np.full(Tsimul, mod_survey["Fert price"][i])
            Price_crop = np.full(Tsimul, mod_survey["Cott price"][i])

            if mod_survey["Tech"][i] == 'drip':
                irr_system = Drip
            elif mod_survey["Tech"][i] == 'sprinkler':
                irr_system = Sprinkler
            elif mod_survey["Tech"][i] == 'furrow':
                irr_system = Furrow
            else:
                irr_system = None
            Constants1['Price_for_chemicals'] = mod_survey['Pesticides'][i]
            Constants1['Price_of_seeds'] = mod_survey['Seed price'][i]
            # add_water = sum(prec.iloc[2:,prec_i])/(max(prec.iloc[2:].index.year) - min(prec.iloc[2:].index.year)) * mod_survey['a_c'][i]

            res_tmp = res[res['RES'] == mod_survey['Res Intake'][i]]
            res_max = mod_survey['Res Vol'].iloc[i]
            res_area = mod_survey['Res Area'].iloc[i]
            res_basin = mod_survey['a_c'][i]
            res_count = mod_survey['Res Count'][i]

            # Dennis: resampling data from monthly to daily if daily_bool==0
            if daily_bool == 0:
                idx = res_tmp.index[-1] + pd.offsets.MonthBegin(1)
                res_tmp = res_tmp.append(res.iloc[[-1]].rename({res_tmp.index[-1]: idx}))

                # resample with forward filling values, remove last helper row
                res_tmp = res_tmp.resample('D').ffill().iloc[:-1]

                # divide by size of months
                res_tmp['FLOW_INcms'] /= res_tmp.resample('MS')['FLOW_INcms'].transform('size')
            res_parameter = np.array([res_tmp, res_max, res_area, res_basin, res_count])

            total_area = mod_survey['Total area'][i]
            Constants1['soil_depth'] = soil_depth[prec_id[i]]
            # add water: None for no reservoir, 1 for including reservoirs
            add_water = 1

            iv_max = mod_survey['Res Vol'][i] * 1000 / (mod_survey["Cott area"][i] * 10000)
            VarMat1[i], WbMat1[i], Yield1[i], YieldMat[i] = Householdmodel(prec.iloc[:, prec_id[i]],
                                                                           ET0.iloc[:, evap_id[i]],
                                                                           T_mean_est.iloc[:, T_id[i]], Constants1,
                                                                           Parameters1, Kc_cotton,
                                                                           Kc_grass, Price_crop, Price_fert, start_year,
                                                                           Tsimul, res_parameter, total_area, wage=0,
                                                                           Smx=Smax[prec_id[i]], IC=100000,
                                                                           fert=mod_survey["Fert kg"][i],
                                                                           crop_area=mod_survey["Cott area"][i],
                                                                           other_area=mod_survey["Other area"][i],
                                                                           livestock=mod_survey["Livestock"][i],
                                                                           Family_size=mod_survey["Laborers"][i],
                                                                           irr_system=irr_system,
                                                                           loan_debt=mod_survey["Debt"][i],
                                                                           interest_rate=mod_survey["Interest"][
                                                                                             i] / 100., Par=Par)
            if mod_survey["Tech"][i] == 'drip':
                irr_system = Drip_fp100
            elif mod_survey["Tech"][i] == 'sprinkler':
                irr_system = Sprinkler_fp100
            else:
                irr_system = Furrow_fp100
            VarMat14[i], WbMat14[i], Yield14[i], YieldMat14[i] = Householdmodel(prec.iloc[:, prec_id[i]],
                                                                                ET0.iloc[:, evap_id[i]],
                                                                                T_mean_est.iloc[:, T_id[i]], Constants1,
                                                                                Parameters1, Kc_cotton,
                                                                                Kc_grass, Price_crop, Price_fert,
                                                                                start_year, Tsimul, res_parameter,
                                                                                total_area, wage=0,
                                                                                Smx=Smax[prec_id[i]], IC=100000,
                                                                                fert=mod_survey["Fert kg"][i],
                                                                                crop_area=mod_survey["Cott area"][i],
                                                                                other_area=mod_survey["Other area"][i],
                                                                                livestock=mod_survey["Livestock"][i],
                                                                                Family_size=mod_survey["Laborers"][i],
                                                                                irr_system=irr_system,
                                                                                loan_debt=mod_survey["Debt"][i],
                                                                                interest_rate=mod_survey["Interest"][
                                                                                                  i] / 100.,
                                                                                add_water=add_water, iv_max=iv_max,
                                                                                subsidies=1.00, Par=Par)
        benefit14 = VarMat14[:, -1, 0] - VarMat1[:, -1, 0]

        Yield_Ranges.append(WbMat1[:, -1, 0].tolist())
        Yield_Ranges_Res.append(WbMat14[:, -1, 0].tolist())
        Benefit_Ranges.append(benefit14.tolist())

        Ranges_Mat[n, 0:len(Par)] = Par
        Ranges_Mat[n, -3] = np.mean(WbMat14[:, -1, 0])
        Ranges_Mat[n, -2] = np.mean(WbMat1[:, -1, 0])
        Ranges_Mat[n, -1] = np.mean(benefit14) / Tsimul

    np.savetxt('Ranges/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_RangesYield2018.csv',
               np.squeeze(Yield_Ranges), delimiter=',')
    np.savetxt('Ranges/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_RangesYieldRes2018.csv',
               np.squeeze(Yield_Ranges_Res), delimiter=',')
    np.savetxt('Ranges/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_RangesBenefit2018.csv',
               np.squeeze(Benefit_Ranges), delimiter=',')
    np.savetxt('Ranges/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_RangesMat2018.csv', Ranges_Mat,
               delimiter=',')

# %% Sensitivity Analysis Variable Plots
"""
Sens_Mat = genfromtxt("MCS/20210326_SensitivityResults_r2_MAE_NS_NSlog_Combined10000.csv", delimiter=',')
Sens_Mat = Sens_Mat[~np.all(Sens_Mat == 0, axis=1)]
if Sens_bool == 1:
    NOpt = np.where(Sens_Mat[:, -1] == np.max(Sens_Mat[:, -1]))[0][0]
    OptPar_NS = Sens_Mat[NOpt]
    NOpt = np.where(Sens_Mat[:, -2] == np.max(Sens_Mat[:, -2]))[0][0]
    OptPar_NSlog = Sens_Mat[NOpt]
    NOpt = np.where(Sens_Mat[:, -3] == np.max(Sens_Mat[:, -3]))[0][0]
    OptPar_MAE = Sens_Mat[NOpt]
    NOpt = np.where(Sens_Mat[:, -4] == np.max(Sens_Mat[:, -4]))[0][0]
    OptPar_r2 = Sens_Mat[NOpt]

    h = np.round((np.shape(Sens_Mat)[1] - 2) / 2)
    w = 2

    plt.figure(figsize=(15, 9))
    # plt.title('Nash-Sutcliffe variation with parameter values')
    # NS
    plt.subplot(h, w, 1)
    plt.plot(Sens_Mat[:, 0], Sens_Mat[:, -1], '.')
    plt.xlabel('Harvest Index', fontsize=16)
    plt.ylabel('NS [-]', fontsize=16)

    plt.subplot(h, w, 2)
    plt.plot(Sens_Mat[:, 1], Sens_Mat[:, -1], '.')
    plt.xlabel('Emergence time', fontsize=16)
    plt.ylabel('NS [-]', fontsize=16)

    plt.subplot(h, w, 3)
    plt.plot(Sens_Mat[:, 2], Sens_Mat[:, -1], '.')
    plt.xlabel('Labor factor function shape', fontsize=16)
    plt.ylabel('NS [-]', fontsize=16)
    plt.savefig('Plots/NS_MCS.png', dpi=300, bbox_inches="tight")

    # NS log
    plt.figure(figsize=(15, 9))
    # plt.title('Log of Nash-Sutcliffe variation with parameter values')
    plt.subplot(h, w, 1)
    plt.plot(Sens_Mat[:, 0], Sens_Mat[:, -2], '.')
    plt.xlabel('Harvest Index', fontsize=16)
    plt.ylabel('$NS_{log}$ [-]', fontsize=16)

    plt.subplot(h, w, 2)
    plt.plot(Sens_Mat[:, 1], Sens_Mat[:, -2], '.')
    plt.xlabel('Emergence time', fontsize=16)
    plt.ylabel('$NS_{log}$ [-]', fontsize=16)

    plt.subplot(h, w, 3)
    plt.plot(Sens_Mat[:, 2], Sens_Mat[:, -2], '.')
    plt.xlabel('Labor factor function shape', fontsize=16)
    plt.ylabel('$NS_{log}$ [-]', fontsize=16)
    plt.savefig('Plots/NSlog_MCS.png', dpi=300, bbox_inches="tight")

    # MAE
    plt.figure(figsize=(15, 9))
    # plt.title('Mean absolute error variation with parameter values')
    plt.subplot(h, w, 1)
    plt.plot(Sens_Mat[:, 0], Sens_Mat[:, -3], '.')
    plt.xlabel('Harvest Index', fontsize=16)
    plt.ylabel('MAE [kg/ha]', fontsize=16)

    plt.subplot(h, w, 2)
    plt.plot(Sens_Mat[:, 1], Sens_Mat[:, -3], '.')
    plt.xlabel('Emergence time', fontsize=16)
    plt.ylabel('MAE [kg/ha]', fontsize=16)

    plt.subplot(h, w, 3)
    plt.plot(Sens_Mat[:, 2], Sens_Mat[:, -3], '.')
    plt.xlabel('Labor factor function shape', fontsize=16)
    plt.ylabel('MAE [kg/ha]', fontsize=16)
    plt.savefig('Plots/MAE_MCS.png', dpi=300, bbox_inches="tight")

    # R2
    # plt.title('R^2 variation with parameter values')
    plt.figure(figsize=(15, 9))
    plt.subplot(h, w, 1)
    plt.plot(Sens_Mat[:, 0], Sens_Mat[:, -4], '.')
    plt.xlabel('Harvest Index', fontsize=16)
    plt.ylabel('$r^2$ [-]', fontsize=16)

    plt.subplot(h, w, 2)
    plt.plot(Sens_Mat[:, 1], Sens_Mat[:, -4], '.')
    plt.xlabel('Emergence time', fontsize=16)
    plt.ylabel('$r^2$ [-]', fontsize=16)

    plt.subplot(h, w, 3)
    plt.plot(Sens_Mat[:, 2], Sens_Mat[:, -4], '.')
    plt.xlabel('Labor factor function shape', fontsize=16)
    plt.ylabel('$r^2$ [-]', fontsize=16)
    plt.savefig('Plots/R2_MCS.png', dpi=300, bbox_inches="tight")

    sys.exit()

# %% plots

plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.plot(years, np.mean(WbMat1[:, :, 0], axis=0), label='base run')
plt.plot(years, np.mean(WbMat14[:, :, 0], axis=0), label='with reservoir')
plt.ylabel('mean yield kg/ha')
plt.legend()

plt.subplot(212)
plt.boxplot([np.mean(WbMat1[:, :, 0], axis=1),
             np.mean(WbMat14[:, :, 0], axis=1)],
            vert=False, positions=[1.6, 1.2], labels=['base run', 'with reservoir'], whis=99)
plt.xlabel("Average yield (kg/ha)")

plt.savefig('Plots/Yield Comparison with Reservoir.png',dpi=300, bbox_inches = "tight")

plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.plot(years, np.mean(WbMat1[:, :, 1], axis=0), label='base run')
plt.plot(years, np.mean(WbMat14[:, :, 1], axis=0), label='with reservoir')
plt.ylabel('Average irrigation supplied [mm]')
plt.legend()

plt.subplot(212)
plt.plot(years, np.mean(VarMat1[:, 1:, 0], axis=0) / 10 ** 5, label='base run')
plt.plot(years, np.mean(VarMat14[:, 1:, 0], axis=0) / 10 ** 5, label='with reservoir')
plt.ylabel('Capital (lakh Rs)')
plt.legend()

plt.savefig('Plots/Irrigation and Capital Comparison.png',dpi=300, bbox_inches = "tight")

"""
# %% Benefits
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.plot(np.mean(WbMat14[:, :, 1], axis=1), benefit14, '.')
plt.axhline(0, ls='--', lw=.5, color='k')
plt.ylabel("Benefit (Rs/ha)")
plt.xlabel("Irrigation applied (mm)")
plt.xlim(0, np.max(WbMat14[:, :, 1]))
plt.ylim(np.min(benefit14), np.max(benefit14))

# %% Benefit box plots

# Dennis: farmer number 191 has nan value (it might be the similar issue
# caused by the runtime warning division by 0 in the soil fertility section)

plt.figure(figsize=(10, 8))
plt.subplot(221)
plt.boxplot([mod_survey.loc[np.where(benefit14 > 0)]['Cott area'],
             mod_survey.loc[np.where(benefit14 < 0)]['Cott area']], labels=['positive', 'negative'])
plt.title('farm area')
plt.subplot(222)
plt.boxplot([Smax[np.where(benefit14 > 0)],
             Smax[np.where(benefit14 < 0)]], labels=['positive', 'negative'])
plt.title('Smax')
plt.subplot(223)
plt.boxplot([np.mean(WbMat14[np.where(benefit14 > 0)][:, :, 1], axis=1),
             np.mean(WbMat14[np.where(benefit14 < 0)][:, :, 1], axis=1)])  # ,
plt.title('Irr (mm)')

plt.subplot(224)
plt.boxplot([mod_survey.loc[np.where(benefit14 > 0)]['Cott area'],
             mod_survey.loc[np.where(benefit14 < 0)]['Cott area']], labels=['positive', 'negative'])
plt.title('farm area')
print('Had irrigation before, positive benefit', sum(mod_survey.loc[benefit14 > 0]['Open well']))
print('Had irrigation before, negative benefit', sum(mod_survey.loc[benefit14 < 0]['Open well']))
print('No irrigation before, positive benefit', sum(mod_survey.loc[benefit14 > 0]['Open well'] == False))
print('No irrigation before, negative benefit', sum(mod_survey.loc[benefit14 < 0]['Open well'] == False))

# %% Net Benefit

Net_benefit = plt.figure(figsize=(12, 7))
plt.axhline(0, color='k', ls='--', lw=1.)
plt.plot(np.linspace(0, 100, len(mod_survey)), np.sort(benefit14) / Tsimul, label='benefit with reservoir')
plt.xlabel('Percentage of farmers', size=22)
plt.ylabel('Net benefit per year (Rs)', size=22)
plt.ylim(min(np.min(benefit14) / Tsimul, -5000), np.max(benefit14) / Tsimul)
plt.xlim(0, 100)
plt.legend(title='Intervention', fontsize=12, title_fontsize=15)
plt.title("Benefit per farmer", size=24)
Net_benefit.savefig('Plots/Net_benefit.png', bbox_inches='tight', pad_inches=.05)
print("Farmers that benefit from new pond with 100% subsidy: " + str(
    np.round(sum(np.sort(benefit14) > 0) / len(benefit14) * 100, 1)) + '%')

# %% Obtaining yield and capital from old model
# Dennis: the yield and capital are calculated using the data from the mdp survey in 2019
# model used is tijmen's code: simulator15_cotton
OldYears = np.linspace(2000, 2019, 20)
OldYield = pd.read_csv("../../input_data/Baseline/OldYield.csv", names=OldYears, delimiter=",")
OldCapital = pd.read_csv("../../input_data/Baseline/OldCapital.csv", names=OldYears, delimiter=",")

# %%
results = pd.DataFrame()
results['Res No'] = raw_survey['Reservoir No']
results['Res Vol'] = raw_survey['Res Vol']
results['Intake Point No'] = mod_survey['Res Intake']
results['Shared Intake Count'] = mod_survey['Res Count']
results['Land Holdings [ha]'] = mod_survey['Total area']
results['Existing Irrigation'] = raw_survey["water/area_irrig"] > 0
results['Yield Benefit [kg/y]'] = (Yield14.sum(1) - Yield1.sum(1)) / Tsimul
results['Benefit [INR/y]'] = benefit14 / Tsimul
results["Lat"] = raw_survey["General/GPS_lat"]
results["Long"] = raw_survey["General/GPS_long"]
results['Dist Reservoir-Intake [km]'] = mod_survey['Dist']
results['Viable (<500m)'] = mod_survey['Dist'] < 0.45

# results.to_csv('Benefit_Reservoir_Baseline.csv', index=False)
# results.to_csv('Benefit_Reservoir_Maha.csv', index=False)
# results.to_csv('Benefit_Reservoir_Ghatanji.csv', index=False)
# results.to_csv('Benefit_Reservoir_Hinhanghat.csv', index=False)

# %% Benefit with reservoirs or not
# results_tmp=results[results['Existing Irrigation']==True]
# ExistingIrr_benefit=np.round(np.sum(results_tmp['Benefit'])/len(results_tmp))

# results_tmp=results[results['Existing Irrigation']==False]
# NoIrr_benefit=np.round(np.sum(results_tmp['Benefit'])/len(results_tmp))

Bool = results['Existing Irrigation'].unique()
IrrBenefit = np.zeros(len(Bool))
for i in range(len(Bool)):
    results_tmp = results[results['Existing Irrigation'] == Bool[i]]
    IrrBenefit[i] = np.round(np.sum(results_tmp['Benefit [INR/y]']) / len(results_tmp))

colors_ben = np.array(['g'] * len(Bool))
colors_ben[IrrBenefit < 0] = 'r'
x = np.array(['had irrigation', 'no irrigation'])
plt.figure(figsize=(3, 7))
bars2 = plt.bar(x, IrrBenefit, color=colors_ben)
plt.ylabel("Mean Benefit per farmer [Rs/y]")
plt.title("Benefit")

xlocs, xlabs = plt.xticks()
xlocs = [i for i in x]
xlabs = [i for i in x]
plt.xticks(xlocs, xlabs)

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + 0.1, yval + 50, yval)

plt.savefig('Plots/BenefitBarIrr.png',dpi=300, bbox_inches = "tight")

# %% State variables and fluxes over tsimul
# evap=ET0.iloc[:,evap_i]
# WD_cotton=Kc_cotton[1:]*evap[evap.index.year==2001]

print('Calculating state variables and fluxes over simulation period')
# Finding average soil moisture per year
names = ['Farmer', 'Day', 'Par']
index = pd.MultiIndex.from_product([range(s) for s in YieldMat.shape], names=names)
df_YieldMat = pd.DataFrame({'YieldMat': YieldMat.flatten()}, index=index)

df_YieldMat = df_YieldMat.unstack(level='Par').swaplevel().sort_index()

df_YieldMat.columns = ['Ta', 'Ea', 'Em', 'SM', 'SM_tmp', 'F', 'Ks', 'Ks_T', 'LAI',
                       'CC', 'Tp', 'Es', 'Ed', 'Evap', 'WD', 'Z_eff', 'ETc', 'Ke',
                       'Kr', 'De', 'Runoff', 'Irr', 'SM_fc', 'SM_wp', 'm', 'm_max', 'Year']
df_YieldMat.index.names = ['Day', 'Farmers']

SM = []
YieldMat_Year = []
CC_farmer = []
CC = []
CC_min = []
CC_max = []
LAI = []
Ks = []
Ta = []
Tp = []
Ea = []
Ed = []
Em = []
WD = []
Z_eff = []
ETc = []
Ke = []
Kr = []
De = []
Runoff = []
Irr = []
# Ta_ts=[]
# Ta_ss=[]
m = []
m_max = []
for i in range(df_YieldMat.index.get_level_values('Day')[-1] + 1):
    par_tmp = df_YieldMat.iloc[df_YieldMat.index.get_level_values('Day') == i]
    SM_tmp = par_tmp['SM'].mean()
    year_tmp = par_tmp['Year'].iloc[0]
    SM.append(SM_tmp)
    YieldMat_Year.append(year_tmp)

    # Finding CC evolution per year
    CC_farm = par_tmp['CC'].iloc[0]
    CC_tmp = par_tmp['CC'].mean()
    _CC_min = par_tmp['CC'].min()
    _CC_max = par_tmp['CC'].max()
    CC_farmer.append(CC_farm)
    CC.append(CC_tmp)
    CC_min.append(_CC_min)
    CC_max.append(_CC_max)

    # Finding average LAI evolution per year
    LAI_tmp = par_tmp['LAI'].mean()
    LAI.append(LAI_tmp)

    # Finding average Ks evolution per year
    Ks_tmp = par_tmp['Ks'].mean()
    Ks.append(Ks_tmp)

    # Transpiration and soil evaporation
    Ta_tmp = par_tmp['Ta'].mean()
    Tp_tmp = par_tmp['Tp'].mean()
    Ea_tmp = par_tmp['Ea'].mean()
    Ta.append(Ta_tmp)
    Tp.append(Tp_tmp)
    Ea.append(Ea_tmp)

    Ed_tmp = par_tmp['Ed'].mean()
    Em_tmp = par_tmp['Em'].mean()
    Ed.append(Ed_tmp)
    Em.append(Em_tmp)

    # Water demand
    WD_tmp = par_tmp['WD'].mean()
    WD.append(WD_tmp)

    # root zone
    Z_tmp = par_tmp['Z_eff'].mean()
    Z_eff.append(Z_tmp)

    # Evapotranspiration WD cotton, Ke (soil coeff) and Kr (soil evap reduction)
    ETc_tmp = par_tmp['ETc'].mean()
    Ke_tmp = par_tmp['Ke'].mean()
    Kr_tmp = par_tmp['Kr'].mean()
    ETc.append(ETc_tmp)
    Ke.append(Ke_tmp)
    Kr.append(Kr_tmp)

    # depletion
    De_tmp = par_tmp['De'].mean()
    De.append(De_tmp)

    # runoff and irr
    Runoff_tmp = par_tmp['Runoff'].mean()
    Runoff.append(Runoff_tmp)

    Irr_tmp = par_tmp['Irr'].mean()
    Irr.append(Irr_tmp)

    m_tmp = par_tmp['m'].mean()
    m.append(m_tmp)

    m_max_tmp = par_tmp['m_max'].mean()
    m_max.append(m_max_tmp)

YieldMat_avg = pd.DataFrame()
YieldMat_avg['Year'] = YieldMat_Year
YieldMat_avg['SM [mm]'] = SM
YieldMat_avg['CC_farmer [-]'] = CC_farmer
YieldMat_avg['CC [-]'] = CC
YieldMat_avg['CC_min [-]'] = CC_min
YieldMat_avg['CC_max [-]'] = CC_max
YieldMat_avg['LAI [-]'] = LAI
YieldMat_avg['Ks [-]'] = Ks
YieldMat_avg['Ta [mm]'] = Ta
YieldMat_avg['Tp [mm]'] = Tp
YieldMat_avg['Ea [mm]'] = Ea
YieldMat_avg['Ed [mm]'] = Ed
YieldMat_avg['Em [mm]'] = Em
YieldMat_avg['WD [mm]'] = WD
YieldMat_avg['Z_eff [mm]'] = Z_eff
YieldMat_avg['ETc [mm]'] = ETc
YieldMat_avg['Ke [-]'] = Ke
YieldMat_avg['Kr [-]'] = Kr
YieldMat_avg['De [mm]'] = De
YieldMat_avg['Runoff [mm]'] = Runoff
YieldMat_avg['Irr [mm]'] = Irr
YieldMat_avg['Biomass [kg]'] = m
YieldMat_avg['Biomass_max [kg]'] = m_max

# %%
print('Plotting state variables and fluxes over simulation period')
plt.figure(figsize=(12, 8))
for i in range(len(years)):
    y = YieldMat_avg[YieldMat_avg['Year'] == years[i]]['SM [mm]']
    x = np.arange(1, len(y) + 1, 1)
    plt.plot(x, y, label=int(years[i]))
plt.title('Average Soil Moisture', fontsize=16)
plt.xlabel('Days of year', fontsize=16)
plt.ylabel('Average Soil Moisture [mm]', fontsize=16)
plt.legend(loc=2)
plt.savefig('Plots/SoilMoisture.png',dpi=300, bbox_inches = "tight")

plt.figure(figsize=(12, 8))
for i in range(len(years)):
    y = YieldMat_avg[YieldMat_avg['Year'] == years[i]]['CC_farmer [-]']
    x = np.arange(1, len(y) + 1, 1)
    plt.plot(x, y, label=int(years[i]))
plt.title('Canopy Cover of farmer', fontsize=16)
plt.xlabel('Days of year', fontsize=16)
plt.ylabel('Canopy Cover [-]', fontsize=16)
plt.xlim(150, len(y) + 1)
plt.legend()
plt.savefig('Plots/CC.png',dpi=300, bbox_inches = "tight")

plt.figure(figsize=(12, 8))
for i in range(len(years)):
    y = YieldMat_avg[YieldMat_avg['Year'] == years[i]]['Ks [-]']
    x = np.arange(1, len(y) + 1, 1)
    plt.plot(x, y, label=int(years[i]))
plt.title('Water Stress (Ks)', fontsize=16)
plt.xlabel('Days of year', fontsize=16)
plt.ylabel('Water Stress [-]', fontsize=16)
plt.xlim(150, len(y) + 1)
plt.legend()
plt.savefig('Plots/WaterStress.png',dpi=300, bbox_inches = "tight")

# %% average precipitation over the year

# finding average precipitation at the farmers location
# prec_id = list(set(prec_id))
prec_avg = prec[prec_id].mean(axis=1)
# evap_id = list(set(evap_id))
# evap_avg=ET0[evap_id].mean(axis=1)
plt_year = 2018

fig, ax1 = plt.subplots(figsize=(12, 8))
y1 = YieldMat_avg[YieldMat_avg['Year'] == plt_year]['ETc [mm]']
y2 = YieldMat_avg[YieldMat_avg['Year'] == plt_year]['Ta [mm]']
y3 = YieldMat_avg[YieldMat_avg['Year'] == plt_year]['Ea [mm]']
y4 = y2 + y3
y5 = YieldMat_avg[YieldMat_avg['Year'] == plt_year]['Biomass [kg]']
x = np.arange(1, len(y1) + 1, 1)
ax2 = ax1.twinx()
ax1.plot(x, y1, label='$ET_c$ in %d' % plt_year, color='b', linewidth=2)
ax1.plot(x, y2, label='$T_a$ in %d' % plt_year, color='r', linewidth=2)
ax1.plot(x, y3, label='$E_s$ in %d' % plt_year, color='orange', linewidth=2)
ax1.plot(x, y4, label='$ET_a$ in %d' % plt_year, color='g', linewidth=2)
ax2.plot(x, y5, label='Daily biomass growth in %d' % plt_year, color='c', linewidth=2)
plt.title('$ET_c$, $T_a$, $E_s$, $ET_a$, and biomass in %d' % plt_year, fontsize=18)
ax1.set_xlabel('Day of year', fontsize=18)
ax1.set_ylabel('$ET_c$, $T_a$, $E_s$, and $ET_a$ [mm]', fontsize=18)
ax2.set_ylabel('Daily biomass growth [kg/d]', fontsize=18)
plt.axvline(353, color='k', ls='--', lw=1.)
plt.axvline(158, color='k', ls='--', lw=1.)
ax2.text(125, 13, 'Planting', fontsize=14)
ax2.text(322, 13, 'Harvest', fontsize=14)
ax2.set_ylim(0, 15)
ax1.set_ylim(0, 8)
ax1.set_xlim(0, 365)
ax1.legend(loc=2, facecolor='white', framealpha=1, fontsize=14)
ax2.legend(loc=1, facecolor='white', framealpha=1, fontsize=14)
plt.savefig('Plots/Biomass.png', dpi=300, bbox_inches="tight")

def plot_CC_ks():
    fig, ax1 = plt.subplots(figsize=(12, 8))
    y1 = prec_avg[prec_avg.index.year == plt_year]
    y2 = YieldMat_avg[YieldMat_avg['Year'] == plt_year]['Ta [mm]']
    y3 = YieldMat_avg[YieldMat_avg['Year'] == plt_year]['Ea [mm]']
    y4 = YieldMat_avg[YieldMat_avg['Year'] == plt_year]['SM [mm]']
    y5 = YieldMat_avg[YieldMat_avg['Year'] == plt_year]['CC [-]']
    y6 = YieldMat_avg[YieldMat_avg['Year'] == plt_year]['Ks [-]']
    y7 = YieldMat_avg[YieldMat_avg['Year'] == plt_year]['Irr [mm]']
    x = np.arange(1, len(y1) + 1, 1)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()
    ax3.bar(x, y1, label='Rainfall in %d' % plt_year, color='b', width=1, alpha=0.5)
    ax3.bar(x, y2, label='Transpiration in %d' % plt_year, color='g', width=1, alpha=0.5)
    ax4.bar(x, y3, label='Soil Evaporation in %d' % plt_year, color='r', width=1, alpha=0.5)
    ax1.plot(x, y4, label='Soil Moisture in %d' % plt_year, color='b', linewidth=2)
    ax2.plot(x, y5, label='CC in %d' % plt_year, color='lime', linewidth=2)
    ax2.plot(x, y6, label='$K_s$ in %d' % plt_year, color='magenta', linewidth=2)
    # ax3.bar(x, y7, label='Irr in %d' % plt_year, color='orange', width=1, alpha=0.5)
    ax1.axvline(353, color='k', ls='--', lw=1.)
    ax1.axvline(158, color='k', ls='--', lw=1.)
    ax1.set_xlabel('Day of year', fontsize=18)
    ax1.set_ylabel('Soil Moisture [mm]', fontsize=18)
    ax2.set_ylabel('CC / $K_s$ [-]', fontsize=18)
    ax3.spines["left"].set_position(("axes", -0.1))
    ax3.spines["left"].set_visible(True)
    ax3.yaxis.set_label_position('left')
    ax3.yaxis.set_ticks_position('left')
    ax3.set_ylabel('Rainfall / Transpiration [mm]', fontsize=18)

    ax4.spines["right"].set_position(("axes", 1.1))
    ax4.spines["right"].set_visible(True)
    ax4.set_ylabel('Soil Evaporation [mm]', fontsize=18)

    plt.title('Rainfall, Evaporation, SM, Irrigation, CC, and $K_s$ evolution in %d' % plt_year, fontsize=18)
    ax2.set_ylim(0, 1.1)
    ax4.set_ylim(0, 1.1)
    ax2.text(125, .92, 'Planting', fontsize=14)
    ax2.text(322, .92, 'Harvest', fontsize=14)
    ax1.set_xlim(0, 365)
    fig.legend(bbox_to_anchor=(0.12, 0.87), loc='upper left')
    plt.savefig('Plots/CC_Ks.png', dpi=300, bbox_inches="tight")
plot_CC_ks()
# %%Yield difference

plt.figure(figsize=(15, 7))
plt.plot(mod_survey.index, OldYield[2018].iloc[0:len(mod_survey) - 1] - mod_survey["Yield"],
         label='Old Yield - Survey 2019')
plt.plot(mod_survey.index, (WbMat1[:, -1, 0]) - mod_survey["Yield"], label='Yield - Survey 2019')
plt.ylabel("Yield [kg/hectare]")
plt.xlabel("Respondents")
plt.title("Yield Difference")
plt.legend()

a = mod_survey.index, OldYield[2018] - mod_survey["Yield"]
b = mod_survey.index, (WbMat1[:, -1, 0]) - mod_survey["Yield"]
c = mod_survey.index, (WbMat14[:, -1, 0]) - mod_survey["Yield"]
bins = np.linspace(-3000, 2000, 50)
# labels=['Old Model','Current Model','Current Model + Res']
labels = ['Old Model', 'Current Model']
plt.figure(figsize=(12, 7))
# plt.hist([a[1],b[1],c[1]], bins,rwidth=0.8,label=labels)
plt.hist([a[1], b[1]], bins, rwidth=0.8, label=labels)
plt.title('Yield Difference Model-Survey')
plt.xlabel('Yield Difference [kg/ha]')
plt.ylabel('Percentage (%)')
plt.legend()
plt.savefig('Plots/YieldDiffModelSurvey.png', dpi=300, bbox_inches="tight")

yield_survey = raw_survey_original["price_cotton_factors/cotton_yield"] * 2.4711 * 100
avg_survey = yield_survey.mean()

# mean absolute error
MAE = np.sum(abs(WbMat1[:, -1, 0] - mod_survey["Yield"])) / len(WbMat1[:, -1, 0])
MAE_res = np.sum(abs(WbMat14[:, -1, 0] - mod_survey["Yield"])) / len(WbMat14[:, -1, 0])
MAEOld = np.sum(abs(OldYield[2018] - mod_survey["Yield"])) / len(WbMat1[:, -1, 0])
print('Mean Absolute Error of Current Model: %d [kg/ha]' % MAE)
print('Mean Absolute Error of Current Model + Res: %d [kg/ha]' % MAE_res)
print('Mean Absolute Error of Old Model: %d [kg/ha]' % MAEOld)

xCOD = np.array(WbMat1[:, -1, 0]).reshape(-1, 1)
xCOD_res = np.array(WbMat14[:, -1, 0]).reshape(-1, 1)
xCOD_old = np.array(OldYield[2018]).reshape(-1, 1)
yCOD = np.array(mod_survey["Yield"]).reshape(-1, 1)
r2_COD = LinearRegression().fit(xCOD, yCOD).score(xCOD, yCOD)
r2_COD_res = LinearRegression().fit(xCOD_res, yCOD).score(xCOD_res, yCOD)
r2_COD_old = LinearRegression().fit(xCOD_old, yCOD).score(xCOD_old, yCOD)
COD = r2_COD
COD_res = r2_COD_res
CODOld = r2_COD_old
print('R2 of Current Model: %f [-]' % COD)
print('R2 of Current Model + Res: %f [-]' % COD_res)
print('R2 of Old Model: %f [-]' % CODOld)

NS_cur_log = 1 - (np.sum((np.log10(WbMat1[:, -1, 0]) - np.log10(mod_survey["Yield"])) ** 2) / np.sum(
    (np.log10(avg_survey) - np.log10(mod_survey["Yield"])) ** 2))
NS_cur_log_res = 1 - (np.sum((np.log10(WbMat14[:, -1, 0]) - np.log10(mod_survey["Yield"])) ** 2) / np.sum(
    (np.log10(avg_survey) - np.log10(mod_survey["Yield"])) ** 2))
NS_old_log = 1 - (np.sum((np.log10(OldYield[2018]) - np.log10(mod_survey["Yield"])) ** 2) / np.sum(
    (np.log10(avg_survey) - np.log10(mod_survey["Yield"])) ** 2))
print('Nash Sutcliffe Log of Current Model: %f [-]' % NS_cur_log)
print('Nash Sutcliffe Log of Current Model + Res: %f [-]' % NS_cur_log_res)
print('Nash Sutcliffe Log of Old Model: %f [-]' % NS_old_log)

NS_cur = 1 - (np.sum((WbMat1[:, -1, 0] - mod_survey["Yield"]) ** 2) / np.sum((avg_survey - mod_survey["Yield"]) ** 2))
NS_cur_res = 1 - (
            np.sum((WbMat14[:, -1, 0] - mod_survey["Yield"]) ** 2) / np.sum((avg_survey - mod_survey["Yield"]) ** 2))
NS_old = 1 - (np.sum((OldYield[2018] - mod_survey["Yield"]) ** 2) / np.sum((avg_survey - mod_survey["Yield"]) ** 2))
print('Nash Sutcliffe of Current Model: %f [-]' % NS_cur)
print('Nash Sutcliffe of Current Model + Res: %f [-]' % NS_cur_res)
print('Nash Sutcliffe of Old Model: %f [-]' % NS_old)

# %% Precipitation plot per year - finding the least MAE to extrapolate irrigation from 2014-2019
prec_sum = prec.sum(axis=1)
prec_sum = prec_sum.resample('MS').sum()
prec_year = np.arange(2000, 2020, 1)
plt.figure(figsize=(12, 8))
for i in range(len(prec_year)):
    y = prec_sum[prec_sum.index.year == prec_year[i]]
    x = np.arange(1, 13, 1)
    plt.plot(x, y, label=prec_year[i])
plt.title('Precipitation')
plt.xlabel('Month')
plt.ylabel('Precipitation [mm]')
# plt.xlim(100,len(y)+1)
plt.legend()

year_proj = np.arange(2014, 2020, 1)
year_tot = np.unique(res.index.year)
year_iter = np.setdiff1d(year_tot, year_proj)
prec_id_mae = pd.DataFrame(index=year_proj, columns=['Year_Similar', 'MAE'])
for i in range(len(year_proj)):
    MAEmat = pd.DataFrame(index=year_iter, columns=['MAE'])
    for j in range(len(year_iter)):
        a = prec_sum[prec_sum.index.year == year_proj[i]]
        a = a.reset_index(drop=True)
        b = prec_sum[prec_sum.index.year == year_tot[j]]
        b = b.reset_index(drop=True)
        MAEmat['MAE'].iloc[j] = np.sum(abs(a - b))
    # finding index of min prec
    MAEmat['MAE'] = pd.to_numeric(MAEmat['MAE'])
    prec_i_mae = MAEmat.idxmin()
    prec_id_mae['Year_Similar'].iloc[i] = prec_i_mae[0]
    prec_id_mae['MAE'].iloc[i] = MAEmat['MAE'].loc[prec_i_mae[0]]

# %% precipitation comparison
plt.figure(figsize=(14, 15))
plt.title('Precipitation')
plt.xlabel('Month')
plt.ylabel('Total precipitation in the region [mm]')
for i in range(len(prec_id_mae)):
    year1 = prec_id_mae.index[i]
    year2 = prec_id_mae['Year_Similar'].iloc[i]
    y1 = prec_sum[prec_sum.index.year == year1]
    y2 = prec_sum[prec_sum.index.year == year2]
    x = np.arange(1, 13, 1)
    plt.subplot(3, 2, i + 1)
    plt.plot(x, y1, label=year1, linewidth=3)
    plt.plot(x, y2, label=year2, linewidth=3)
    plt.legend()
plt.text(-1, 280000, 'Precipitation comparisons', ha='center', fontsize=22)
plt.text(-1, -18000, 'Months', ha='center', fontsize=22)
plt.text(-17, 140000, 'Total precipitation in the region [mm]', va='center', rotation='vertical', fontsize=22)

plt.savefig('Plots/PrecipitationCompareForIrrigation.png',dpi=300, bbox_inches = "tight")

plt.savefig('Plots/PrecipitationCompareForIrrigation2.png', bbox_inches = "tight")
plt.savefig('Plots/PrecipitationCompareForIrrigation.png', dpi=300, bbox_inches="tight")

# %% Coordinates for reservoirs
# res_coor_gha=pd.read_csv('indat/reservoir locations solidaridad.csv', header = 0, sep = ",")
# res_coor_gha_new=pd.read_csv('indat/reservoir_coordinates_ghatanji.csv', header = 0, sep = ",")
# res_coor_hin=pd.read_csv('indat/reservoir_coordinates_hinhanghat.csv', header = 0, sep = ",")
# Farmer_sol= pd.read_excel('indat/NewPondSEC_Maha.xlsx', header = 0, sep = ",")
# res_coor_sol=pd.read_csv('indat/reservoir locations solidaridad.csv', header = 0, sep = ",")

yielddif_color = mod_survey["Yield"] - mod_survey.index, (WbMat1[:, -1, 0])
color_val = np.asarray(yielddif_color)
plt.figure(figsize=(12, 10))
plt.scatter(raw_survey["General/GPS_long"], raw_survey["General/GPS_lat"], c=color_val[1, :], cmap='coolwarm',
            norm=colors.TwoSlopeNorm(vmin=-1500., vcenter=0., vmax=3000), label='Farmers')
plt.legend()
plt.xlabel('Long')
plt.ylabel('Lat')
plt.colorbar()
plt.title("Calculated - Observed Yield Difference Map")
# plt.savefig('Plots/YieldDiffMap.png', dpi=300, bbox_inches="tight")

yielddif_color = benefit14 / Tsimul
color_val = np.asarray(yielddif_color)
plt.figure(figsize=(12, 10))
plt.scatter(raw_survey["General/GPS_long"], raw_survey["General/GPS_lat"], c=color_val, cmap='coolwarm_r',
            norm=colors.TwoSlopeNorm(vmin=-5000., vcenter=0., vmax=5000), label='Farmers')
plt.legend()
plt.xlabel('Long')
plt.ylabel('Lat')
plt.colorbar()
plt.title("Yearly Benefit Map [Rs/y]")
# plt.savefig('Plots/BenefitYearly.png', dpi=300, bbox_inches="tight")

# %% plotting ranges
"""
Ranges_Mat = genfromtxt("Ranges/2021-03-15-15-08-29_RangesMat2018.csv", delimiter=',')
Benefit_Ranges = genfromtxt("Ranges/2021-03-15-15-08-29_RangesBenefit2018.csv", delimiter=',')
Benefit_Boot = genfromtxt("Bootstrap/2021-03-16-23-41-48_BootBenefit2018.csv", delimiter=',')

Yield_Ranges = genfromtxt("Ranges/2021-03-17-21-06-29_RangesYield2018.csv", delimiter=',')
Yield_Ranges_Res = genfromtxt("Ranges/2021-03-17-21-06-29_RangesYieldRes2018.csv", delimiter=',')

Benefit_mean = np.sort(np.mean(Benefit_Ranges, axis=0))
Benefit_min = np.sort(np.min(Benefit_Ranges, axis=0))
Benefit_max = np.sort(np.max(Benefit_Ranges, axis=0))
Net_benefit = plt.figure(figsize=(12, 7))
plt.axhline(0, color='k', ls='--', lw=1.)
yerr_up = Benefit_max - Benefit_mean
yerr_do = Benefit_mean - Benefit_min
yerr = np.vstack((yerr_up, yerr_do))
plt.errorbar(np.linspace(0, 100, len(mod_survey)), Benefit_mean, yerr, fmt='r^', label='Annual yield per farmer',
             elinewidth=0.5,
             marker='.', markersize='1', markeredgecolor='blue', ecolor=['red'], barsabove=False)
plt.xlabel('Percentage of farmers', size=22)
plt.ylabel('Net benefit per year (INR)', size=22)
plt.xlim(0, 100)
plt.legend(title='Intervention', fontsize=12, title_fontsize=15)
plt.title("Benefit per farmer with model uncertainties", size=24)
# Net_benefit.savefig('Plots/Net_benefit_modeluncertainties.png', bbox_inches='tight', pad_inches=.05)

Benefit_mean_boot = np.sort(np.mean(Benefit_Boot, axis=0))
Benefit_min_boot = np.sort(np.min(Benefit_Boot, axis=0))
Benefit_max_boot = np.sort(np.max(Benefit_Boot, axis=0))
Net_benefit = plt.figure(figsize=(12, 7))
plt.axhline(0, color='k', ls='--', lw=1.)
# plt.scatter(np.linspace(0,100,len(mod_survey)), np.sort(Benefit_max)/Tsimul, label='Max benefit', marker='.')
yerr_up = Benefit_max_boot - Benefit_mean_boot
yerr_do = Benefit_mean_boot - Benefit_min_boot
yerr = np.vstack((yerr_up, yerr_do))
plt.errorbar(np.linspace(0, 100, len(mod_survey)), Benefit_mean_boot, yerr, fmt='r^', label='Annual yield per farmer',
             elinewidth=0.5,
             marker='.', markersize='1', markeredgecolor='blue', ecolor=['red'], barsabove=False)
plt.xlabel('Percentage of farmers', size=22)
plt.ylabel('Net benefit per year (INR)', size=22)
plt.xlim(0, 100)
plt.legend(title='Intervention', fontsize=12, title_fontsize=15)
plt.title("Benefit per farmer with variability", size=24)
# Net_benefit.savefig('Plots/Net_benefit_boot.png', bbox_inches='tight', pad_inches=.05)


# %% plotting bootstrap+ranges
Yield_Boot = genfromtxt("Bootstrap/2021-03-16-23-41-48_BootYield2018.csv", delimiter=',')
y_yield = np.mean(Yield_Boot, axis=0)
x_yield = np.mean(Yield_Ranges, axis=0)
model = LinearRegression().fit(x_yield.reshape(-1, 1), y_yield.reshape(-1, 1))
r2_yield = model.score(x_yield.reshape(-1, 1), y_yield.reshape(-1, 1))
fit = np.poly1d(np.polyfit(x_yield, y_yield, 1))

plt.figure(figsize=(10, 7))
xerr_up = np.max(Yield_Boot, axis=0) - np.mean(Yield_Boot, axis=0)
xerr_do = np.mean(Yield_Boot, axis=0) - np.min(Yield_Boot, axis=0)
xerr = np.vstack((xerr_up, xerr_do))
yerr_up = np.max(Yield_Ranges, axis=0) - np.mean(Yield_Ranges, axis=0)
yerr_do = np.mean(Yield_Ranges, axis=0) - np.min(Yield_Ranges, axis=0)
yerr = np.vstack((yerr_up, yerr_do))
plt.errorbar(x_yield, y_yield, xerr, yerr, fmt='r^', label='Annual yield per farmer', elinewidth=0.5,
             marker='o', markersize='5', markeredgecolor='k', ecolor=['red'], barsabove=False)
plt.plot(np.unique(x_yield), np.poly1d(np.polyfit(x_yield, y_yield, 1))(np.unique(x_yield)), color="black",
         linestyle='--', linewidth=2)
plt.plot(np.arange(5000), np.arange(5000), color="black")
plt.title('Model Uncertainty and External Variation for Yearly Yield', fontsize=16)
plt.ylabel('Yield variation from bootstrap [kg/ha/y]', fontsize=16)
plt.xlabel('Yield uncertainty from model [kg/ha/y]', fontsize=16)
plt.xlim(0, 3500)
plt.ylim(0, 3500)
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.legend(loc=2)
# plt.savefig('Plots/YieldBootstrapUncertainty.png',dpi=300, bbox_inches = "tight")

# %% whole survey vs rainfed farmers
# combined
Yield_Boot = genfromtxt("Bootstrap/2021-03-16-23-41-48_BootYield2018.csv", delimiter=',')

# Sens_Mat=genfromtxt("MCS/20210326_SensitivityResults_r2_MAE_NS_NSlog_Combined10000.csv", delimiter=',')
# Sens_Mat=Sens_Mat[~np.all(Sens_Mat == 0, axis=1)]
# _Yield_Boot=genfromtxt("MCS/20210326_YieldMCS_Combined10000.csv", delimiter=',')
# arr_index=np.where((Sens_Mat[:,-4]>0.003)&(Sens_Mat[:,-3]<600)&(Sens_Mat[:,-2]>-2)&(Sens_Mat[:,-1]>-1))
# Yield_Boot=_Yield_Boot[arr_index]

# Boot_Mat=Boot_Mat[~np.all(Boot_Mat == 0, axis=1)]
x_yield = np.mean(Yield_Boot, axis=0)
y_yieldSur = np.array(mod_survey["Yield"])

# plotfit=np.arange(100,2600,100)
plotfit = np.arange(100, 3000, 100)

fit = np.poly1d(np.polyfit(x_yield, y_yieldSur, 1))
model = LinearRegression().fit(x_yield.reshape(-1, 1), y_yieldSur.reshape(-1, 1))
r2_yield_sur = model.score(x_yield.reshape(-1, 1), y_yieldSur.reshape(-1, 1))
plt.figure(figsize=(10, 7))
xerr_up = np.max(Yield_Boot, axis=0) - np.mean(Yield_Boot, axis=0)
xerr_do = np.mean(Yield_Boot, axis=0) - np.min(Yield_Boot, axis=0)
xerr = np.vstack((xerr_up, xerr_do))
yerr = 0
# plt.errorbar(x_yield, y_yieldSur, yerr, xerr, fmt='r^',label='Annual yield per farmer', elinewidth =0.5,
#               c="red",s=20, edgecolor='k',marker='x', markersize='5',markeredgecolor='blue', ecolor=['red'],barsabove=False)
plt.errorbar(x_yield, y_yieldSur, yerr, xerr, fmt='r^', label='Annual yield per farmer', elinewidth=2,
             c="red", marker='o', markersize='5', markeredgecolor='k', ecolor=['red'], barsabove=False)
plt.plot(np.unique(plotfit), fit(plotfit), color="black", linestyle='--', linewidth=3)  # , label='Best fit line')
# plt.text(max(x_yield)+500, max(y_yieldSur),'y={}x+{}'.format(round(fit[1],2),round(fit[0])), ha='left', va='center')
plt.text(max(x_yield) + 500, max(y_yieldSur) - 200, '$r^2$ = {}'.format(round(r2_yield_sur, 4)), ha='left', va='center',
         fontsize=18)
plt.title('Comparison of predicted vs. observed yield for all farmers', fontsize=16)
# plt.xlabel('Predicted yields from model and uncertainties [kg/ha/y]',fontsize=16)
plt.xlabel('Predicted yields from model and variability [kg/ha/y]', fontsize=16)
plt.ylabel('Observed yields from survey [kg/ha/y]', fontsize=16)
plt.plot(np.arange(5000), np.arange(5000), color="black", linewidth=2)  # label='x=y line',
plt.xlim(0, 5000)
plt.ylim(0, 5000)
plt.legend()
plt.grid(color='k', linestyle='-', linewidth=0.1)
# plt.savefig('Plots/Survey_Yield_Uncertainties.png',dpi=300, bbox_inches = "tight")
# # plt.savefig('Plots/Survey_Yield_Boot2018.png',dpi=300, bbox_inches = "tight")

maxrange = x_yield + xerr[0, :]
minrange = x_yield - xerr[1, :]
ncount = 0
for i in range(len(x_yield)):
    if minrange[i] <= y_yieldSur[i] <= maxrange[i]:
        ncount += 1
print(ncount / len(x_yield))

# rainfed
# plotfit=np.arange(100,2600,100)
plotfit = np.arange(800, 2600, 100)

tmp = mod_survey[pd.isna(mod_survey['Tech'])]
x_yield_rain = x_yield[pd.isnull(mod_survey['Tech']).nonzero()[0]]
y_yieldSur_rain = np.array(tmp['Yield'])
fit = np.poly1d(np.polyfit(x_yield_rain, y_yieldSur_rain, 1))
model = LinearRegression().fit(x_yield_rain.reshape(-1, 1), y_yieldSur_rain.reshape(-1, 1))
r2_yield_sur_rain = model.score(x_yield_rain.reshape(-1, 1), y_yieldSur_rain.reshape(-1, 1))
plt.figure(figsize=(10, 7))
xerr_up = np.max(Yield_Boot, axis=0) - np.mean(Yield_Boot, axis=0)
xerr_do = np.mean(Yield_Boot, axis=0) - np.min(Yield_Boot, axis=0)
xerr = np.vstack((xerr_up, xerr_do))
xerr_rain = xerr[:, pd.isnull(mod_survey['Tech']).nonzero()[0]]
yerr = 0
plt.errorbar(x_yield_rain, y_yieldSur_rain, yerr, xerr_rain, fmt='r^', label='Annual yield per farmer', elinewidth=2,
             c="red", marker='o', markersize='5', markeredgecolor='k', ecolor=['red'], barsabove=False)
plt.plot(plotfit, fit(plotfit), color="black", linestyle='--', linewidth=3)  # , label='Best fit line')
# plt.text(max(x_yield_rain)+1000, max(y_yieldSur_rain)-800,'y={}x+{}'.format(round(fit[1],2),round(fit[0])), ha='left', va='center')
plt.text(max(x_yield_rain) + 500, 4500, '$r^2$ = {}'.format(round(r2_yield_sur_rain, 4)), ha='left', va='center',
         fontsize=18)
plt.title('Comparison of predicted vs. observed yield for rainfed farmers only', fontsize=16)
# plt.xlabel('Predicted yields from model and uncertainties [kg/ha/y]',fontsize=16)
plt.xlabel('Predicted yields from model and variability [kg/ha/y]', fontsize=16)
plt.ylabel('Observed yields from survey [kg/ha/y]', fontsize=16)
plt.plot(np.arange(5000), np.arange(5000), color="black", linewidth=2)  # ,label='x=y line'
plt.xlim(0, 5000)
plt.ylim(0, 5000)
plt.legend()
plt.grid(color='k', linestyle='-', linewidth=0.1)
# plt.savefig('Plots/Survey_Yield_Uncertainties_Rainfed.png',dpi=300, bbox_inches = "tight")
# plt.savefig('Plots/Survey_Yield_Boot2018_Rainfed.png',dpi=300, bbox_inches = "tight")

maxrange = x_yield_rain + xerr_rain[0, :]
minrange = x_yield_rain - xerr_rain[1, :]
ncount_rain = 0
for i in range(len(x_yield_rain)):
    if minrange[i] <= y_yieldSur_rain[i] <= maxrange[i]:
        ncount_rain += 1
print(ncount_rain / len(x_yield_rain))

# %% print input for spss
YieldRanges = genfromtxt("Ranges/2021-03-17-21-06-29_RangesYield2018.csv", delimiter=',')
YieldRangesRes = genfromtxt("Ranges/2021-03-17-21-06-29_RangesYieldRes2018.csv", delimiter=',')

spss_mat = np.zeros((len(mod_survey), 7))
for i in range(len(mod_survey)):
    temp = prec[prec_id[i]]
    temp = temp[temp.index.year == 2018]
    temp = temp[(150 < temp.index.dayofyear) & (temp.index.dayofyear < 310)].sum()
    spss_mat[i, 0] = temp

    temp = ET0[evap_id[i]]
    temp = temp[temp.index.year == 2018]
    temp = temp[(150 < temp.index.dayofyear) & (temp.index.dayofyear < 310)].sum()
    spss_mat[i, 1] = temp

dfTemp = df_YieldMat[df_YieldMat['Year'] == 2018]
IrrSum = []
for i in range(dfTemp.index.get_level_values('Farmers')[-1] + 1):
    par_tmp = dfTemp.iloc[dfTemp.index.get_level_values('Farmers') == i]
    IrrTmp = par_tmp['Irr'].sum()
    IrrSum.append(IrrTmp)

spss_mat[:, 2] = IrrSum
spss_mat[:, 3] = mod_survey["Yield"] - YieldRanges.mean(axis=0)
spss_mat[:, 4] = YieldRanges.mean(axis=0)
spss_mat[:, 5] = mod_survey["Yield"]
spss_mat[:, 6] = YieldRangesRes.mean(axis=0)

spss_df = pd.DataFrame(spss_mat)
spss_df.to_excel(excel_writer="SPSS_WorkDir/PrecYield2018_v3.xlsx")

# %% Yield comparison old and new model
Yield_Boot = genfromtxt("Ranges/2021-02-10-00-20-13_RangesYield_MCS_10000.csv", delimiter=',')
x1 = np.mean(Yield_Boot, axis=0)
x2 = OldYield[2018]
y_yieldSur = np.array(mod_survey["Yield"])

x1 = np.array(x1).reshape(-1, 1)
x2 = np.array(x2).reshape(-1, 1)

fit = np.poly1d(np.polyfit(x_yield, y_yieldSur, 1))
model = LinearRegression().fit(x1.reshape(-1, 1), y_yieldSur.reshape(-1, 1))
r2_new = model.score(x1.reshape(-1, 1), y_yieldSur.reshape(-1, 1))
model_old = LinearRegression().fit(x2.reshape(-1, 1), y_yieldSur.reshape(-1, 1))
r2_old = model_old.score(x2.reshape(-1, 1), y_yieldSur.reshape(-1, 1))

plt.figure(figsize=(10, 7))
plt.scatter(x1, y_yieldSur, color="b", marker='x', label='Yield of new model')
plt.scatter(x2, y_yieldSur, color='orange', marker='x', label='Yield of old model')
plt.text(max(x1), max(y_yieldSur) - 200, '$r^2$ new={}'.format(round(r2_new, 4)), ha='left', va='center', fontsize=16)
plt.text(max(x1), max(y_yieldSur) - 400, '$r^2$ old={}'.format(round(r2_old, 5)), ha='left', va='center', fontsize=16)
plt.title('Yield comparison between old and new model', fontsize=16)
plt.xlabel('Yields from old and new model [kg/ha/y]', fontsize=16)
plt.ylabel('Observed yields from survey [kg/ha/y]', fontsize=16)
plt.plot(np.arange(5000), np.arange(5000), color="black")
plt.xlim(0, 5000)
plt.ylim(0, 5000)
plt.legend()
# plt.savefig('Plots/Old_New_Yield.png',dpi=300, bbox_inches = "tight")

"""
# %% Formatting gleam and NDVI data to be plotted
ncSM = '../../input_data/NDVI_GLEAM/SMroot_2018_GLEAM_v3.7b.nc'  # 'indat\SMroot_2018_GLEAM_v3.5b.nc'
fh = Dataset(ncSM, mode='r')

lons = fh.variables['lon'][:]
lats = fh.variables['lat'][:]
SMroot = fh.variables['SMroot'][:]
SMroot_units = fh.variables['SMroot'].units
fh.close()

# Get some parameters for the Stereographic Projection
lon_0 = lons.mean()
lat_0 = lats.mean()
doy = [180, 210, 240, 270, 300, 330, 345]

# NDVI=pd.read_csv('indat/NDVI_Farmers.csv', header = 0, sep = ",")
TestCoord = pd.read_csv('../../input_data/NDVI_GLEAM/TestingCoordFarm30_Edited.csv', header=0, sep=",")

# NDVI=NDVI.drop(['fid'],axis=1)
SM_farmers = pd.read_csv('../../input_data/NDVI_GLEAM/TestingCoordFarm.csv', header=0, sep=",")

# SM_farmers=SM_farmers.drop(SM_farmers.columns[[5,6]],axis=1)
SM_id = np.zeros((len(TestCoord), 2))
for i in range(len(TestCoord)):
    # DisMat = np.sqrt((lats - SM_farmers["Lat"][i])**2 + (lons- SM_farmers["Long"][i])**2)
    DisMat_lat = (np.array(lats) - TestCoord["Lat"][i]) ** 2
    DisMat_lon = (np.array(lons) - TestCoord["Long"][i]) ** 2
    # finding index of min prec
    SM_latid = DisMat_lat.argmin()
    SM_lonid = DisMat_lon.argmin()
    SM_id[i, 0] = SM_latid
    SM_id[i, 1] = SM_lonid
SM_id = SM_id.astype(int)

# getting SM from GLEAM data
SM_mat = np.zeros((len(doy), len(TestCoord)))
for i in range(len(doy)):
    for j in range(len(SM_id)):
        SM_mat[i, j] = SMroot[doy[i], SM_id[j, 1], SM_id[j, 0]]
FarmID = []
for i in range(len(TestCoord)):
    tmp = mod_survey[
        (mod_survey['Lat'] == TestCoord['LatRef'].iloc[i]) & (mod_survey['Long'] == TestCoord['LongRef'].iloc[i])]
    # FarmID.append(tmp.index[0])
    FarmID.append(tmp.index.to_numpy())

SM_mat_mm = np.zeros((len(doy), len(TestCoord)))
SM_mat_mm_min = np.zeros((len(doy), len(TestCoord)))
SM_mat_mm_max = np.zeros((len(doy), len(TestCoord)))
for i in range(len(FarmID)):
    SM_mat_mm[:, i] = SM_mat[:, i] * soil_depth[FarmID[i]].mean()
    SM_mat_mm_min[:, i] = SM_mat[:, i] * soil_depth[FarmID[i]].min()
    SM_mat_mm_max[:, i] = SM_mat[:, i] * soil_depth[FarmID[i]].max()

# getting SM from model
SM_mat_mm_model = np.zeros((len(doy), len(TestCoord)))
SM_mat_mm_model_min = np.zeros((len(doy), len(TestCoord)))
SM_mat_mm_model_max = np.zeros((len(doy), len(TestCoord)))
for i in range(len(doy)):
    for j in range(len(TestCoord)):
        tmp = df_YieldMat[df_YieldMat['Year'] == 2018]
        # tmp=tmp.iloc[tmp.index.get_level_values('Farmers') == FarmID[j]]
        tmp = tmp.iloc[tmp.index.get_level_values('Farmers').isin(FarmID[j])]
        _tmp = tmp.mean(level=0)
        SM_mat_mm_model[i, j] = _tmp.iloc[doy[i]].SM
        _tmp = tmp.min(level=0)
        SM_mat_mm_model_min[i, j] = _tmp.iloc[doy[i]].SM
        _tmp = tmp.max(level=0)
        SM_mat_mm_model_max[i, j] = _tmp.iloc[doy[i]].SM

# getting CC from model
CC_mat_model = np.zeros((len(doy), len(TestCoord)))
CC_mat_model_min = np.zeros((len(doy), len(TestCoord)))
CC_mat_model_max = np.zeros((len(doy), len(TestCoord)))
for i in range(len(doy)):
    for j in range(len(TestCoord)):
        tmp = df_YieldMat[df_YieldMat['Year'] == 2018]
        # tmp=tmp.iloc[tmp.index.get_level_values('Farmers') == FarmID[j]]
        tmp = tmp.iloc[tmp.index.get_level_values('Farmers').isin(FarmID[j])]
        _tmp = tmp.mean(level=0)
        CC_mat_model[i, j] = _tmp.iloc[doy[i]].CC
        _tmp = tmp.min(level=0)
        CC_mat_model_min[i, j] = _tmp.iloc[doy[i]].CC
        _tmp = tmp.max(level=0)
        CC_mat_model_max[i, j] = _tmp.iloc[doy[i]].CC

# %%
# plotting basemap
"""
m = Basemap(width=5000000, height=3500000,
            resolution='l', projection='stere', \
            lat_ts=40, lat_0=20, lon_0=78)
# # Because our lon and lat variables are 1D,
# # use meshgrid to create 2D arrays
# # Not necessary if coordinates are already in 2D arrays.
lon, lat = np.meshgrid(lons, lats)
xi, yi = m(lon, lat)

plt.figure(figsize=(10, 7))

# # Plot Data
cs = m.pcolor(xi, yi, np.squeeze(SMroot[doy[0], :, :].T))
# Add Grid Lines
m.drawparallels(np.arange(-80., 81., 10.), labels=[1, 0, 0, 0], fontsize=10)
m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], fontsize=10)
# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()
# Add Colorbar
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label(SMroot_units)
# Add Title
plt.title('Mean SM in Root Zone')
plt.show()
"""
# prec.iloc[:,prec_id[i]], ET0.iloc[:,evap_id[i]]
precPlot = prec[(prec.index.year == 2018)].sum(axis=0)
etPlot = ET0[(ET0.index.year == 2018)].sum(axis=0)
TPlot = T_mean_est[(T_mean_est.index.year == 2018)].mean(axis=0)

plt.figure(figsize=(7, 5))
plt.scatter(prec_coor.loc['Long'], prec_coor.loc['Lat'], c=precPlot, s=500, cmap='Blues', vmin=0, vmax=2000, marker='s')
plt.title('Rainfall in 2018')
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.colorbar().set_label('Rainfall [mm]', rotation=270, labelpad=15)
plt.savefig('Plots/Rainfall.png')

plt.figure(figsize=(7, 5))
plt.scatter(ET_coor.loc['Long'], ET_coor.loc['Lat'], c=etPlot, s=8000, cmap='Blues', vmin=1800, vmax=2000, marker='s')
plt.title('Evapotranspiration in 2018')
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.colorbar().set_label('Evapotranspiration [mm]', rotation=270, labelpad=15)
plt.savefig('Plots/ETc.png')

plt.figure(figsize=(7, 5))
plt.scatter(ET_coor.loc['Long'], ET_coor.loc['Lat'], c=TPlot, s=8000, cmap='Blues', vmin=25, vmax=30, marker='s')
plt.title('Mean Temperature in 2018')
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.colorbar().set_label('Mean Temperature [degree C]', rotation=270, labelpad=15)
plt.savefig('Plots/Tmean.png')

# %%
yPlot = np.array(SM_mat_mm[:, 0:25].mean(axis=1))
yPlot_model = np.array(SM_mat_mm_model[:, 0:25].mean(axis=1))
xPlot = doy
plt.figure(figsize=(10, 7))
plt.plot(xPlot, SM_mat_mm.min(axis=1), color='royalblue', linewidth=0.1)
plt.plot(xPlot, yPlot, c="blue", label="GLEAM SM range")
plt.plot(xPlot, SM_mat_mm.max(axis=1), color='royalblue', linewidth=0.1)
plt.fill_between(xPlot, SM_mat_mm.min(axis=1), SM_mat_mm.max(axis=1), color='skyblue', alpha='0.5')
plt.plot(xPlot, SM_mat_mm_model.min(axis=1), color='tomato', linewidth=0.1)
plt.plot(xPlot, yPlot_model, c="red", label="Model SM range")
plt.plot(xPlot, SM_mat_mm_model.max(axis=1), color='tomato', linewidth=0.1)
plt.fill_between(xPlot, SM_mat_mm_model.min(axis=1), SM_mat_mm_model.max(axis=1), color='orangered', alpha='0.5')
plt.ylim(0, 250)
plt.xlim(175, 350)
plt.title('Mean evolution of GLEAM SM and model SM from sampled locations')
plt.ylabel('Soil moisture [mm]')
plt.xlabel('Day of year')
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.legend()
# plt.savefig('Plots/SM_Gleam_DOY_mean.png',dpi=300, bbox_inches = "tight")

_yPlot = np.array(TestCoord.iloc[0:25, 7:]).T
yPlot = _yPlot.mean(axis=1)
# yPlot=np.array(NDVI.iloc[FarmID].mean())
yPlot_model = np.array(CC_mat_model[:, 0:25].mean(axis=1))
xPlot = doy
plt.figure(figsize=(10, 7))
plt.plot(xPlot, _yPlot.min(axis=1), color='royalblue', linewidth=0.1)
plt.plot(xPlot, yPlot, c="blue", label="NDVI range")
plt.plot(xPlot, _yPlot.max(axis=1), color='royalblue', linewidth=0.1)
plt.fill_between(xPlot, _yPlot.min(axis=1), _yPlot.max(axis=1), color='skyblue', alpha='0.5')
plt.plot(xPlot, yPlot_model, c="red", label="Model CC")
plt.ylim(0, 1)
plt.xlim(175, 350)
plt.title('Mean evolution of NDVI and model CC in sampled locations')
plt.ylabel('Canopy Cover/NDVI [-]')
plt.xlabel('Day of year')
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.legend()
# plt.savefig('Plots/CC_NDVI_DOY_mean.png',dpi=300, bbox_inches = "tight")

_yPlot = np.array(TestCoord.iloc[25:, 7:]).T
yPlot = _yPlot.mean(axis=1)
# yPlot=np.array(NDVI.iloc[FarmID].mean())
# yPlot_model=np.array(CC_mat_model[:,0:25].mean(axis=1))
xPlot = doy
plt.figure(figsize=(10, 7))
plt.plot(xPlot, _yPlot.min(axis=1), color='royalblue', linewidth=0.1)
plt.plot(xPlot, yPlot, c="blue", label="NDVI range")
plt.plot(xPlot, _yPlot.max(axis=1), color='royalblue', linewidth=0.1)
plt.fill_between(xPlot, _yPlot.min(axis=1), _yPlot.max(axis=1), color='skyblue', alpha='0.5')
# plt.plot(xPlot,yPlot_model,c="red", label="Model CC")
plt.ylim(0, 1)
plt.xlim(175, 350)
plt.title('Mean evolution of NDVI and model CC in sampled locations')
plt.ylabel('Canopy Cover/NDVI [-]')
plt.xlabel('Day of year')
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.legend()
# plt.savefig('Plots/CC_NDVI_DOY_Xinjiang.png',dpi=300, bbox_inches = "tight")

# %%
def get_region_id(region: str):
    return dict( wardha=0, ghatanji=1, yavatmal=2, amravati=3, hinhanghat=4)[region]
def plot_figure16(region: str):
    PlotID = get_region_id(region)

    VarID = TestCoord['PlaceID'] == PlotID

    yPlot_model = SM_mat_mm_model[:, VarID]
    yPlot_model_min = SM_mat_mm_model_min[:, VarID]
    yPlot_model_max = SM_mat_mm_model_max[:, VarID]
    yPlot = SM_mat_mm[:, VarID]
    xPlot = doy
    fig, axes = plt.subplots(2, 1, figsize=(10, 14))
    axes[0].plot(xPlot, yPlot.mean(axis=1), c="blue", label="GLEAM SM")
    axes[0].plot(xPlot, yPlot_model.mean(axis=1), c="red", label="Model SM")
    axes[0].fill_between(xPlot, yPlot_model_min.mean(axis=1), yPlot_model_max.mean(axis=1), color='skyblue', alpha='0.5')
    axes[0].ylim(0, 250)
    axes[0].xlim(175, 350)
    axes[0].title('Evolution of GLEAM SM and model SM in {}'.format(region), fontsize=16)
    axes[0].ylabel('Soil moisture [mm]', fontsize=16)
    axes[0].xlabel('Day of year', fontsize=16)
    axes[0].grid(color='k', linestyle='-', linewidth=0.1)
    axes[0].legend()

    yPlot_model = CC_mat_model[:, VarID]
    yPlot = np.array(TestCoord[VarID].iloc[:, 7:]).T
    xPlot = doy
    axes[1].plot(xPlot, yPlot.min(axis=1), color='royalblue', linewidth=0.1)
    axes[1].plot(xPlot, yPlot.mean(axis=1), c="blue", label="NDVI range")
    axes[1].plot(xPlot, yPlot.max(axis=1), color='royalblue', linewidth=0.1)
    axes[1].fill_between(xPlot, yPlot.min(axis=1), yPlot.max(axis=1), color='skyblue', alpha='0.5')
    axes[1].plot(xPlot, yPlot_model.mean(axis=1), c="red", label="Model CC")
    axes[1].ylim(0, 1)
    axes[1].xlim(175, 350)
    axes[1].title('Evolution of NDVI and model CC in {}'.format(SM_farmers['Loc'].iloc[PlotID]), fontsize=16)
    axes[1].ylabel('Canopy Cover/NDVI [-]', fontsize=16)
    axes[1].xlabel('Day of year', fontsize=16)
    axes[1].grid(color='k', linestyle='-', linewidth=0.1)
    axes[1].legend()
    axes[1].savefig('Plots/CC_NDVI_DOY_{}.png'.format(SM_farmers['Loc'].iloc[PlotID]),dpi=300, bbox_inches = "tight")

plot_figure16("Wardha")
