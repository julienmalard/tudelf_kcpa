# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:53:05 2020

@author: denni
"""
import numpy as np


def Irrigation(res_parameter, IrrPar, WBPar):
    # unpacking parameters
    res_max = res_parameter[1]
    res_max = res_max * 1000  # m3 to mm m2
    res_area = res_parameter[2]
    res_basin = res_parameter[3]
    res_count = res_parameter[4]

    add_water = IrrPar[0]
    total_area = IrrPar[1]
    irr_system = IrrPar[2]
    De = IrrPar[3]
    PlantSeason = IrrPar[4]

    precip1 = WBPar[0]
    evap1 = WBPar[1]
    Inflow = WBPar[2]
    SM = WBPar[3]
    TAW = WBPar[4]
    ResStorage = WBPar[5]
    SM_wp_max = WBPar[6]

    # defining existing well parameters
    well_area = np.pi * 1.5 ** 2  # well with diameter 3m
    well_max = well_area * 10 * 1000  # 10m deep and changing from m3 to mm m2

    if irr_system is not None:
        c_irr = irr_system['irr_eff']

    if add_water is not None:
        if SM - SM_wp_max < 0.50 * TAW and PlantSeason == 1 and irr_system is not None:
            Irr = min(ResStorage, max((De) * total_area / c_irr, 0))
            Eo = evap1 * (res_area + well_area)
            max_store = res_max + well_max
            ResIn = Inflow + precip1 * (res_area + well_area)
            ResStorage_tmp = max(0, min(ResStorage + ResIn - Eo - Irr, max_store))
        else:
            Irr = 0
            Eo = evap1 * (res_area + well_area)
            max_store = res_max + well_max
            ResIn = Inflow + precip1 * (res_area + well_area)
            ResStorage_tmp = max(0, min(ResStorage + ResIn - Eo - Irr, max_store))
    else:  # no new reservoir built, only using existing well
        if SM - SM_wp_max < 0.50 * TAW and PlantSeason == 1 and irr_system is not None:
            Irr = min(ResStorage, max((De) * total_area / c_irr, 0))
            Eo = evap1 * well_area
            max_store = well_max
            ResIn = Inflow + precip1 * well_area  # this ResIn is used when using qswat data: inflow stays in here because it is assumed that the inflow is also used to fill the well
            # ResIn=res_basin*precip1[i]*0.5 #This resin is for not using qswat data: runoff coefficient from https://www.researchgate.net/publication/335109010_Experimental_Study_of_Runoff_Coefficients_for_Different_Hill_Slope_Soil_Profiles
            ResStorage_tmp = max(0, min(ResStorage + ResIn - Eo - Irr,
                                        max_store))  # changing reservoir vol unit from m to mm
        else:
            Irr = 0
            Eo = evap1 * well_area
            max_store = well_max
            ResIn = Inflow + precip1 * well_area  # this ResIn is used when using qswat data: inflow stays in here because it is assumed that the inflow is also used to fill the well
            # ResIn=res_basin*precip1[i]*0.5 #This resin is for not using qswat data: runoff coefficient from https://www.researchgate.net/publication/335109010_Experimental_Study_of_Runoff_Coefficients_for_Different_Hill_Slope_Soil_Profiles
            ResStorage_tmp = max(0, min(ResStorage + ResIn - Eo - Irr, max_store))

    IrrMat = [Irr, ResStorage_tmp, Eo]
    return IrrMat
