# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:21:18 2015

@author: Nadezhda
"""

import numpy as np
# import pandas as pd


# from datetime import datetime

# parse2 = lambda x: datetime.strptime(x, "%d-%m-%y")
# precip_evap = pd.read_csv('INDAT/Point_Rainfall_Evap.csv', sep=';', header=0, names=None, index_col=[0], date_parser=parse2) #,index_col=[0], parse_dates=[0]
#
##Het volgende kan je gebruiken om je jaren op te indexeren/selecteren
# precip = precip_evap['Precipitation [mm/day]'][(precip_evap.index.year==1983)] #vervang jaartal straks dus door i
#
# WD = pd.read_csv('INDAT/Water_demand.csv', sep=';', header=0, names=None) #I left this out because otherwiseit would start at [1] index_col=[0]
# WD_cotton = WD['Water demand cotton']
#
# Grass_demand= np.zeros(len(precip)) #just filled in something
#
#
##Import to function
# Sumt=30
# frac=0.5
# IrriBool=0


def Storage(Constants, precip1, frac, Sumt, WD_cotton, WD_grass, irr_system=None, add_water=None, iv_max=None):
    ##Hydrology calculated at daily basis
    ##assuming no deep percolation.

    Mat = np.empty((len(precip1), 4))
    SM = np.zeros(len(precip1) + 1)
    Ea_s = np.ones(len(precip1))  # Actual Evaporation # to be calculated
    SumIrr = 0
    if irr_system is not None:
        c_irr = irr_system['irr_eff']
    if add_water is not None:
        S_add = add_water
    else:
        S_add = 0.  # After every season, the well is empty.

    for i in range(len(precip1)):
        # soil moisture balance equation: dSM/dt * delt = min(SM+(P-E-DP)*delt, SMmax)
        SM[0] = Sumt

        # Grass transpiration
        Ea_g = (1. - frac) * min(WD_grass[i], SM[i])

        # Crop transpiration
        Ea_s = frac * min(WD_cotton[i], SM[i])

        # Irrigation:
        # Applied when lower than 20% of max capacity, in cotton season and water available for irrigation.
        if SM[i] < 0.20 * Constants['Su_max'] and i > 156 and i < 337 and S_add > 0.:
            Irr = min(S_add * frac, max((Constants['Su_max'] - SM[i]) / c_irr, 0))
            SumIrr += Irr / frac
            S_add -= Irr / frac
            SM[i] += Irr * c_irr
        # Calculating runoff
        # if no intervention, S_add is capped by open well capacity at 12mm of storage. 
        # Well is 10m deep by 6m diameter and is used for every 25000 m^2 of farmland.
        R = max(0, SM[i] + precip1[i] - Constants['Su_max'])
        if irr_system is not None and add_water is None:
            S_add = min(S_add + R / frac, 12.)
        elif iv_max is not None:
            S_add = min(S_add + R / frac, iv_max)

        # SM differential equation
        SM[i + 1] = max(min(SM[i] + precip1[i] - Ea_s - Ea_g - R, Constants['Su_max']), 0)
        # Save water balance
        Mat[i] = np.array([R, Ea_s / frac, Ea_g / (1. - frac), SM[i]])

        # Use last soil moisture value for next year's iteration:
    Sumt = SM[-1]
    return Mat, Sumt, SumIrr
