# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:21:18 2015

@author: Dennis
"""
import numpy as np

from src.sh_model.Irrigation import Irrigation


# import matplotlib.pyplot as plt
# from globalparameters import Original, Adjusted
# import pandas as pd
# from datetime import datetime
def WB_Yield(Constants, precip1, evap1, T, res_tmp_y, res_parameter, crop_area, grassland_area, total_area, frac,
             YieldSumt, YieldRes, WD_cotton, Kc_cotton, irr_system, reservoir=0, add_water=None, Par=None):
    # Return matrix array [transpiration,soil evaporation, water demand met,soil moisture]
    Mat = np.empty((len(precip1), 26))
    EsMat = np.empty((len(precip1), 7))

    if irr_system is not None:
        c_irr = irr_system['irr_eff']

    Sumt = YieldSumt

    total_area = total_area * 10000
    CropArea = crop_area * 10000  # m2
    GrassArea = grassland_area * 10000  # m2

    ####################OTHER PARAMETERS NEEDED########################
    if Par is not None:
        # CC_o = Par[0]
        # CGC  = Par[1]
        # CCx  = Par[2]
        # CDC  = Par[3]
        HI_o = Par[0]
        t_CCo = int(round(Par[1]))
        # CWP = Par[6]
        # FC = Par[7]
        # WP = Par[8]
        # P_upper=Par[9]
        # P_lower=Par[10]
        # f_shape=Par[2] #shape of the convex, >0 for convex curves
        # f_shape_z=Par[3]


    else:
        HI_o = 0.374
        t_CCo = 22

        # testing OptPar values
        # CC_o=8.43129979e-03 #initial surface covered 5-7 cm2/plant and there is ~ 60k-150k plants / ha (from aquacrop annex)
        # CGC=1.14158159e-01 #0.0965 # canopy growth coefficient [increase of fraction ground cover per day or growing degree day]
        # CCx=9.85469814e-01 #maximum canopy cover [fraction ground cover] #end of season in time (days)
        # CDC=4.19548347e-02 # from annex and 0.05 from page 19 manual
        # HI_o=2.62023487e-01 #fraction of Yield from Biomass - can change throughout the date of year depending on crop type.
        # t_CCo=int(round(2.65518598e+01)) #growth days - or t after planting when CC_o occurs [whem 90% of sprout appear]
        # CWP=1.54714330e+01 # Normalized Crop Water Productivity (g/m2/mm) - Can be adjusted by various factors.
        # FC = 4.03163371e-01 #[-] Water holding capacity:http://jocpr.com/vol8-iss1-2016/JCPR-2016-8-1-153-160.pdf
        # WP = 1.44414456e-01 #[-] Soil wilting point :http://jocpr.com/vol8-iss1-2016/JCPR-2016-8-1-153-160.pdf
        # P_upper=7.58376997e-01 # Water Stress Threshold
        # P_lower=2.85907461e-014
        # f_shape=9.54998422e-01
        # f_shape_z=-5.35841819e+00

    # Original values
    CC_o = 0.0063  # initial surface covered 5-7 cm2/plant and there is ~ 60k-150k plants / ha (from aquacrop annex)
    CGC = 0.0965  # 0.0965 # canopy growth coefficient [increase of fraction ground cover per day or growing degree day]
    CCx = 0.95  # maximum canopy cover [fraction ground cover] #end of season in time (days)
    CDC = 0.03  # from annex and 0.05 from page 19 manual
    # t_CCo = 15 #growth days - or t after planting when CC_o occurs [whem 90% of sprout appear]
    CWP = 15.  # Normalized Crop Water Productivity (g/m2/mm) - Can be adjusted by various factors.
    FC = 0.42  # [-] Water holding capacity:https://www.jocpr.com/articles/study-of-two-indian-soils.pdf
    WP = 0.17  # [-] Soil wilting point :https://www.jocpr.com/articles/study-of-two-indian-soils.pdf
    P_upper = 0.7  # Water Stress Threshold
    P_lower = 0.2
    f_shape = 5
    f_shape_z = -4

    plantdate = 158  # Day in year that you start growing
    # t_end=50#50#180 #for cotton end of growing season
    # f_hi : Change of HI_o depending on the stress. #f_hi is affected by water stress, but water stress is already taken into account.
    # Dennis: this can be affected by air temperature stress
    #       so the comment below isnt true when air temp is implemented
    f_hi = 1.  # I THINK we can leave this out - value of 1 means no change to Harvest Index (HI) value.
    t_grow = 195  # growth time from t_CCo until senescence or harvest #Total growth
    t_Ss = 60  # time of senescence

    Z_max = min(Constants['soil_depth'], 1400)  # max effective rooting depth [mm]
    Z_n = 300  # minimum root depth, check: the minimum mentioned in the appendix is 0.3m which is for the most part more than the soil depth
    Z_o = Z_n / 2  # starting depth of root zone expansion
    n = 1.5  # shape factor of root zone development, from aquacrop annex

    # SM_fc = Z_max * FC #Field capacity
    # SM_wp = Z_max * WP #wilting point
    SM_fc_max = Z_max * FC
    SM_wp_max = Z_max * WP
    TAW_max = SM_fc_max - SM_wp_max

    # Soil specific
    Kb = 0.6  # For soil evaporation, fraction of potential evaporation that evaporates from the soil
    K_sat = 1000. / 365.  # mean hydraulic conductivity for silty loess, what is the unit

    ####################STEP 1########################
    # Kcb
    # t parameters taken from fao evapotranspiration manual: http://www.fao.org/3/X0490E/x0490e0b.htm#TopOfPage
    t_ini = 30  # duration of initial stage (t until 10% of ground covered)
    t_dev = 50  # duration of development stage
    t_mid = 60  # duration after t_dev to plant maturity
    t_late = 55  # duration from maturity to harvest

    Kcb_ini = 0.15
    Kcb_mid = 1.1
    Kcb_end = 0.4

    Kcb_cotton = np.zeros(366)
    Kcb_cotton[plantdate - 1:plantdate + t_ini - 1] = Kcb_ini
    Kcb_cotton[plantdate + t_ini - 1:plantdate + t_ini + t_dev - 1] = np.linspace(Kcb_ini, Kcb_mid, t_dev)
    Kcb_cotton[plantdate + t_ini + t_dev - 1:plantdate + t_ini + t_dev + t_mid - 1] = Kcb_mid
    Kcb_cotton[plantdate + t_ini + t_dev + t_mid - 1:plantdate + t_ini + t_dev + t_mid + t_late - 1] = np.linspace(
        Kcb_mid, Kcb_end, t_late)
    Kcmax_cotton = Kcb_cotton + 0.05

    ETc = np.zeros(len(precip1))

    D = 0.  # threshold for interception

    LAI = np.zeros(len(precip1))  # Leaf area index
    Ks = np.zeros(len(precip1))  # Soil moisture stress
    Ks_sto = np.ones(len(precip1))  # Soil moisture stress in the root zone
    Ks_T = np.zeros(len(precip1))  # Air temperature stress
    CGC_adj = np.zeros(len(precip1))  # Canopy growth coefficient
    CC = np.zeros(len(precip1))  # Canopy cover
    #    Kc = np.zeros(t_end)
    Ta = np.zeros(len(precip1))
    Tp = np.zeros(len(precip1))
    Ea = np.zeros(len(precip1))
    Es = np.zeros(len(precip1))  # soil evap
    Eo = np.zeros(len(precip1))  # open water evap
    Kr = np.zeros(len(precip1))  # soil evaporation reduction
    Ke = np.zeros(len(precip1))  # daily evaporation coefficient
    Rt = np.zeros(len(precip1))  # deep percolation
    Z_eff = np.zeros(len(precip1))
    SM_fc = np.zeros(len(precip1))  # fielc capacity in terms of SM [mm]
    SM_wp = np.zeros(len(precip1))  # wilting point in terms of SM [mm]
    TAW = np.zeros(len(precip1))  # total water available (sm_fc-sm_wp) [mm]
    De = np.zeros(len(precip1))  # depletion [mm]
    Runoff = np.zeros(
        len(precip1))  # runoff from field (not used for anything in this model yet, just recording the water lost)
    Irr_vol = np.zeros(len(precip1))  # Irrigation total volume
    Irr = np.zeros(len(precip1))  # Irrigation in mm per m2
    # values taken from Allen et al : https://www.researchgate.net/publication/228721933_Estimating_Evaporation_from_Bare_Soil_and_the_Crop_Coefficient_for_the_Initial_Period_Using_Common_Soils_Information
    TEW = 35  # maximum evaporable water for Z ~ 0.1 to 0.15 m
    REW = 10  # readily evaporable water

    # soil moisture and reservoir storage
    SM = np.zeros(len(precip1) + 1)
    SM[0] = Sumt
    ResStorage = np.zeros(len(precip1) + 1)
    CC[plantdate + t_CCo] = CC_o

    stage2 = False
    stageSen = False
    tz = 0  # t counter for root zone starts at t_CCo/2
    t_x = 100  # t to reach max root depth, check the number, now just based on the cc graph, from planting to max cc
    t1 = 0  # t counter after growing starts
    t2 = 0  # t counter after senecsence starts
    GDD1 = np.zeros(len(precip1))  # GDD for growing stages
    GDD2 = np.zeros(len(precip1))  # GDD after senescense

    # Air temperature stress threshold (in degree C)
    # Threshold values taken from aquacrop manual page 3-92
    T_base = 12  # for cotton used for cotton development (using GDD calculation NOTE: different from T up and T bot for biomass temp stress)
    T_up = 20
    T_bot = 0.1
    A = 999  # Dennis: not sure how to explain this, big number means low intercept for stress value, we want it close to 0 at T_bottom
    k = np.log((1 / 0.999 - 1) / A) / (T_up - T_bot)  # Dennis: rate of change of the logistic function
    # print(k)

    # Reservoir info
    # #unpacking res_parameter
    # res_max=res_parameter[1]
    # res_area=res_parameter[2]
    # res_basin=res_parameter[3]
    res_count = res_parameter[4]

    Inflow = res_tmp_y['FLOW_INcms'] * 86400 * 1000 / res_count  # unit is cubic meter / second to mm m2/d

    m = np.zeros(len(precip1))  # daily biomass obtained
    m_max = np.zeros(len(precip1))
    # cco_bool=0
    for i in range(len(precip1)):  #

        ResStorage[0] = YieldRes

        ##INTERCEPTION
        I = np.minimum(precip1[i], D)

        ##(IN)FILTRATION
        F = precip1[i] - I

        # root zone
        t_zo = round(t_CCo / 2)
        if plantdate < i <= plantdate + int(t_grow - t_Ss):
            # Root zone growth
            tz += 1
            if tz - t_zo <= 0:
                Z_eff[i] = min(Z_n, Z_max)
            else:
                # Z_eff[i]=min(max(Z_o+(Z_max-Z_o)*((tz-t_zo)/(t_x-t_zo))**(1/n),Z_n),Z_max)

                dz_eff = -((Z_max - Z_o) * ((tz - t_zo) / (t_x - t_zo)) ** (1 / n)) / (n * (t_zo - tz))

                # convex shape (not too strong effects)
                # dz_eff_adj=dz_eff*((np.exp(Ks_sto[i-1]*f_shape_z)-1)/(np.exp(f_shape_z)-1))

                # linear (strong effects)
                dz_eff_adj = dz_eff * Ks_sto[i - 1]

                Z_eff[i] = min(max(Z_eff[i - 1] + dz_eff_adj, Z_n), Z_max)

                # Z_eff[i]=min((tz-t_zo)*Z_o+(Z_max-Z_o)*((tz/(t_x-t_zo))**(1/n)),Z_max)
                # Z_eff[i]=min(max(tz-t_zo)Z_o+(Z_max-Z_o)*((t/(t_x-t_zo))**(1/n),Z_n),Z_max)

        SM_fc[i] = Z_eff[i] * FC  # Field capacity
        SM_wp[i] = Z_eff[i] * WP  # wilting point

        TAW[i] = SM_fc[i] - SM_wp[i]
        Dr_upper = P_upper * TAW[i]
        Dr_lower = P_lower * TAW[i]
        De[i] = SM_fc_max - SM[i]

        P_upper_sto = 0.65
        Dr_upper_sto = P_upper_sto * TAW[i]

        if plantdate + t_CCo < i <= plantdate + int(t_grow - t_Ss):  # and cco_bool==1:  #

            # GDD1[i]=GDD1[i-1]+max(T[i]-T_base,0)
            t1 += 1
            ####################STEP 4########################
            # Calculate water stress factor
            # assuming a linear relation
            # if SM[i]-SM_wp[i] >= Dr_upper:
            #     Ks[i]=1.

            # elif SM[i]-SM_wp[i] <= Dr_lower:
            #     Ks[i]=0.

            # elif Dr_lower < SM[i]-SM_wp[i] < Dr_upper:
            #     Ks[i]=(SM[i]-SM_wp[i]-Dr_lower)/(Dr_upper-Dr_lower)
            #     # Ks[i]=(SM[i]-SM_wp)/(Dr_upper-Dr_lower)

            # water stress
            if SM_fc[i] - De[i] >= Dr_upper:
                Ks[i] = 1.

            elif SM_fc[i] - De[i] <= Dr_lower:
                Ks[i] = 0.

            elif Dr_lower < SM_fc[i] - De[i] < Dr_upper:

                # linear
                Ks[i] = (SM_fc[i] - De[i] - Dr_lower) / (Dr_upper - Dr_lower)

                # convex
                # S_rel=1-((SM_fc[i] - De[i] - Dr_lower)/(Dr_upper-Dr_lower))
                # Ks[i]=1-(np.exp(S_rel*f_shape)-1)/(np.exp(f_shape)-1)

            # water stress in the root zone
            if SM_fc[i] - De[i] >= Dr_upper_sto:
                Ks_sto[i] = 1.

            elif SM_fc[i] - De[i] <= 0:
                Ks_sto[i] = 0.

            elif 0 < SM_fc[i] - De[i] < Dr_upper_sto:
                Ks_sto[i] = 1 - (SM_fc[i] - De[i]) / (Dr_upper_sto)

                ####################STEP 5########################
            CGC_adj[i] = Ks[i] * CGC  # [adjust the canopy growth coefficient to water stress]

            ####################STEP 6########################
            ###Calculating the crop growth###
            # using GDD
            # if (CC[i-1] <= CCx/2) and (stage2==False) and stageSen==False:
            #     GDD1[i]=GDD1[i-1]+max(T[i]-T_base,0)
            #     dCC = CC_o * CGC_adj[i] * np.exp(GDD1[i]*CGC_adj[i])
            #     CC[i] = np.maximum(np.minimum(CC[i-1] + dCC,CCx),0)

            # #stage 2:
            # elif ((CC[i-1] > CCx/2) or (stage2==True)) and stageSen==False:
            #     stage2=True
            #     GDD1[i]=GDD1[i-1]+max(T[i]-T_base,0)
            #     dCC=0.25*((CCx)**2./CC_o)*np.exp(-(GDD1[i]*CGC_adj[i]))*CGC_adj[i]
            #     CC[i]= np.maximum(np.minimum(CC[i-1]+dCC,CCx),0)

            # using days
            if (CC[i - 1] <= CCx / 2) and (stage2 == False) and stageSen == False:
                dCC = CC_o * CGC_adj[i] * np.exp(t1 * CGC_adj[i])
                CC[i] = np.maximum(np.minimum(CC[i - 1] + dCC, CCx), 0)

            # stage 2:
            elif ((CC[i - 1] > CCx / 2) or (stage2 == True)) and stageSen == False:
                stage2 = True
                dCC = 0.25 * ((CCx) ** 2. / CC_o) * np.exp(-(t1 * CGC_adj[i])) * CGC_adj[i]
                CC[i] = np.maximum(np.minimum(CC[i - 1] + dCC, CCx), 0)

            # print(dCC)

            ##LEAF AREA INDEX
            # https://www.researchgate.net/publication/266895709_Parameterization_and_Evaluation_of_FAO_AquaCrop_Model_for_Full_and_Deficit_Irrigated_Cotton
            LAI[i] = 100. / 77. * np.log(1. / (1. - CC[i]))

            ##TRANSPIRATION
            # Paredes2014 uses CC instead of LAI
            # Tp[i] = WD_cotton[i]*Ks[i]* np.minimum(LAI[i],1)

            ##SOIL EVAPORATION
            # Es[i] = evap1[i]*Kb*np.maximum(1.-LAI[i],0)   
            # Remove stress factor

            if (De[i]) <= REW:
                Kr[i] = 1
            elif (De[i]) > REW:
                Kr[i] = max((TEW - (De[i])) / (TEW - REW), 0)
            Ke[i] = Kcmax_cotton[i] - Kcb_cotton[i]
            ETc[i] = (Kcb_cotton[i] + Ke[i]) * evap1[i]

            Es[i] = Kr[i] * (1 - CC[i]) * Ke[i] * evap1[i]
            # dennis: fix this kcb val
            Tp[i] = Ks[i] * Kcb_cotton[i] * CC[i] * evap1[i]  # kcb_mid is the max value for kcb

            # Es[i] = np.maximum(evap1[i]*Kb*((SM[i]-SM_wp_max)/(SM_fc_max-SM_wp_max))*np.maximum(1.-CC[i],0),0)

        elif plantdate + (t_grow - t_Ss) < i <= plantdate + t_grow:  # len(precip1):
            # GDD2[i]=GDD2[i-1]+max(T[i]-T_base,0)
            t2 += 1
            if t2 == 1:
                CCsen = CC[i - 1]

            # using GDD
            # dCC = -0.05*CDC*np.exp(CDC*GDD2[i]/CCsen)
            # CC[i]=np.maximum(CC[i-1]+dCC,0)

            # using days
            dCC = -0.05 * CDC * np.exp(CDC * t2 / CCsen)
            CC[i] = np.maximum(CC[i - 1] + dCC, 0)
            # CC[i]=max(CCsen*(1-0.05*(np.exp(CDC/CCsen * t2)-1)),0)

            ##LEAF AREA INDEX
            # https://www.researchgate.net/publication/266895709_Parameterization_and_Evaluation_of_FAO_AquaCrop_Model_for_Full_and_Deficit_Irrigated_Cotton
            # faharani et al 2009
            LAI[i] = 100. / 77. * np.log(1. / (1. - CC[i]))
            # print(LAI[i],CC[i])

            ##TRANSPIRATION
            # remove stress factor: Tp[i] = WD_cotton[i]*Ks[i]* np.minimum(LAI[i],1)
            # Tp[i] = WD_cotton[i]*Ks[i]* np.minimum(LAI[i],1)
            # Tp[i] = Ks[i]*Kcmax_cotton[i]*CC[i]*evap1

            ##SOIL EVAPORATION
            # Es[i] = evap1[i]*Kb*np.maximum(1.-LAI[i],0)   
            # Remove stress factor
            # equation from: https://www.researchgate.net/publication/228721933_Estimating_Evaporation_from_Bare_Soil_and_the_Crop_Coefficient_for_the_Initial_Period_Using_Common_Soils_Information
            if (De[i]) <= REW:
                Kr[i] = 1
            elif (De[i]) > REW:
                Kr[i] = max((TEW - (De[i])) / (TEW - REW), 0)
            Ke[i] = Kcmax_cotton[i] - Kcb_cotton[i]
            ETc[i] = (Kcb_cotton[i] + Ke[i]) * evap1[i]

            Es[i] = Kr[i] * (1 - CC[i]) * Ke[i] * evap1[i]
            Tp[i] = Ks[i] * Kcb_cotton[i] * CC[i] * evap1[i]

            # Es[i] = np.maximum(evap1[i]*Kb*((SM[i]-SM_wp_max)/(SM_fc_max-SM_wp_max))*np.maximum(1.-CC[i],0),0)


        else:
            Tp[i] = 0
            # Es[i] = evap1[i]*Kb*np.maximum(1.-LAI[i],0)
            if (De[i]) <= REW:
                Kr[i] = 1
            elif (De[i]) > REW:
                Kr[i] = max((TEW - (De[i])) / (TEW - REW), 0)
            Ke[i] = Kcmax_cotton[i] - Kcb_cotton[i]
            ETc[i] = (Kcb_cotton[i] + Ke[i]) * evap1[i]
            Es[i] = Kr[i] * (1 - CC[i]) * Ke[i] * evap1[i]
            # Es[i] = np.maximum(evap1[i]*Kb*((SM[i]-SM_wp_max)/(SM_fc_max-SM_wp_max))*np.maximum(1.-CC[i],0),0)

        # Using pond to irrigate
        if i > plantdate and i < plantdate + (t_grow):
            PlantSeason = 1
        else:
            PlantSeason = 0
        IrrPar = [add_water, total_area, irr_system, De[i], PlantSeason]
        WBPar = [precip1[i], evap1[i], Inflow.iloc[i], SM[i], TAW[i], ResStorage[i], SM_wp_max]

        IrrMat = Irrigation(res_parameter, IrrPar, WBPar)
        Irr_vol[i] = IrrMat[0]
        Irr[i] = Irr_vol[i] / total_area
        ResStorage[i + 1] = IrrMat[1]
        Eo[i] = IrrMat[2]

        # if add_water is not None:
        #     if SM[i]-SM_wp_max < 0.50*TAW[i] and i>plantdate and i<plantdate+(t_grow) and irr_system is not None:
        #         Irr=min(ResStorage[i], max((Z_max - SM[i])*total_area/c_irr, 0))
        #         SumIrr += Irr
        #         Eo[i]=evap1[i]*res_area
        #         max_store=res_max+well_max
        #         ResIn=Inflow.iloc[i]+precip1[i]*(res_area+well_area)
        #         ResStorage[i+1]=max(0,min(ResStorage[i]+ResIn-Eo[i]-Irr,max_store))
        #     else:
        #         Irr=0
        #         Eo[i]=evap1[i]*res_area
        #         SumIrr += Irr
        #         max_store=res_max+well_max
        #         ResIn=Inflow.iloc[i]+precip1[i]*(res_area+well_area)
        #         ResStorage[i+1]=max(0,min(ResStorage[i]+ResIn-Eo[i]-Irr,max_store)) 
        # else: #no new reservoir built, only using existing well
        #     if SM[i]-SM_wp_max < 0.50*TAW[i] and i>plantdate and i<plantdate+(t_grow) and irr_system is not None:
        #         Irr=min(ResStorage[i], max((Z_max - SM[i])*total_area/c_irr, 0))
        #         SumIrr += Irr
        #         Eo[i]=evap1[i]*well_area
        #         max_store=well_max
        #         ResIn=Inflow.iloc[i]+precip1[i]*well_area #this ResIn is used when using qswat data: inflow stays in here because it is assumed that the inflow is also used to fill the well
        #         # ResIn=res_basin*precip1[i]*0.5 #This resin is for not using qswat data: runoff coefficient from https://www.researchgate.net/publication/335109010_Experimental_Study_of_Runoff_Coefficients_for_Different_Hill_Slope_Soil_Profiles
        #         ResStorage[i+1]=max(0,min(ResStorage[i]+ResIn-Eo[i]-Irr,max_store)) #changing reservoir vol unit from m to mm
        #     else:
        #         Irr=0
        #         SumIrr += Irr
        #         Eo[i]=evap1[i]*well_area
        #         max_store=well_max
        #         ResIn=Inflow.iloc[i]+precip1[i]*well_area #this ResIn is used when using qswat data: inflow stays in here because it is assumed that the inflow is also used to fill the well
        #         # ResIn=res_basin*precip1[i]*0.5 #This resin is for not using qswat data: runoff coefficient from https://www.researchgate.net/publication/335109010_Experimental_Study_of_Runoff_Coefficients_for_Different_Hill_Slope_Soil_Profiles
        #         ResStorage[i+1]=max(0,min(ResStorage[i]+ResIn-Eo[i]-Irr,max_store)) 

        ## total demand
        Ed = Tp[i] + Es[i]

        ## demand met
        Em = np.minimum(min(TAW_max, SM[i] + F), Ed)

        ##SOIL MOISTURE
        SM_tmp = np.maximum(0, np.minimum(SM[i] + F - Em + Irr[i] + Rt[i], SM_fc_max))  # rt is negative
        Runoff[i] = max(SM[i] + F - Em + Irr[i] - SM_fc_max, 0)

        # ##RECHARGE
        # R = (1/K_sat)*np.maximum(SM_tmp-SM_fc,0)

        ##FINAL SOIL MOISTURE
        SM[i + 1] = SM_tmp  # - R #+1+1

        if Ed != 0:
            ## Transpiration possible: in proportion to demands
            Ta[i] = Tp[i] * Em / Ed

            ## Soil evaporation possible
            Ea[i] = Es[i] * Em / Ed
        else:
            Ta[i] = 0
            Ea[i] = 0

        if T[i] <= T_bot:
            Ks_T[i] = 0
        if T[i] >= T_up:
            Ks_T[i] = 1
        if T_bot < T[i] < T_up:
            Ks_T[i] = 1 / (1 + A * np.exp(k * (T[i] - T_bot)))

        if plantdate + t_CCo < i < plantdate + t_grow and ETc[i] > 0:
            m[i] = Ks_T[i] * CWP * (Ta[i] / ETc[i])
            m_max[i] = CWP

        # Matrix array to store information
        Mat[i] = np.array([Ta[i], Ea[i], Em, SM[i], SM_tmp, F, Ks[i], Ks_T[i], LAI[i],
                           CC[i], Tp[i], Es[i], Ed, evap1[i], ETc[i],
                           Z_eff[i], ETc[i], Ke[i], Kr[i], De[i], Runoff[i], Irr[i],
                           SM_fc[i], SM_wp[i], m[i], m_max[i]])

    B = np.sum(m)
    B_max = np.sum(m_max)

    # print(Constants['max_Crop_yield'])
    conversion = 10000 / 1000  # changing the unit from g/m2/y to kg/ha/y
    Yield = f_hi * HI_o * B * conversion  # Yield; kg/ha/y
    Constants['max_Crop_yield'] = HI_o * B_max * conversion

    Sumt = SM[-1]
    SumIrr = np.sum(Irr)
    YieldRes = ResStorage[-1]

    # print(np.sum(Ta[plantdate+t_CCo:plantdate+t_grow]),np.sum(WD_cotton[plantdate+t_CCo:plantdate+t_grow]))
    # if Constants['max_Crop_yield']<Yield:
    # print(Constants['max_Crop_yield'])
    # print(np.shape(Mat),np.shape(Sumt),np.shape(())
    return Mat, Sumt, SumIrr, YieldRes, Yield
