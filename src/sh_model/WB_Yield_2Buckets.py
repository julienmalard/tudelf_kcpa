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
    Mat = np.empty((len(precip1), 31))
    EsMat = np.empty((len(precip1), 7))

    if irr_system is not None:
        c_irr = irr_system['irr_eff']
    # if add_water is not None:
    #     S_add = add_water
    # else :
    #     S_add = 0.      # After every season, the well is empty.

    total_area = total_area * 10000
    CropArea = crop_area * 10000  # m2
    GrassArea = grassland_area * 10000  # m2

    ####################OTHER PARAMETERS NEEDED########################
    # if Par is not None:
    #     # CC_o = Par[0]
    #     # CGC  = Par[1]
    #     # CCx  = Par[2]
    #     # CDC  = Par[3]
    #     HI_o   = Par[0]
    #     t_CCo = int(round(Par[1]))
    #     # CWP = Par[6]
    #     # FC = Par[7]
    #     # WP = Par[8]
    #     # P_upper=Par[9]
    #     # P_lower=Par[10]
    #     # f_shape=Par[2] #shape of the convex, >0 for convex curves
    #     # f_shape_z=Par[3]

    # else:
    #     HI_o   = 0.293
    #     t_CCo = 13       

    # testing OptPar values - Good
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
    HI_o = 0.293
    t_CCo = 13

    plantdate = 158  # Day in year that you start growing
    # t_end=50#50#180 #for cotton end of growing season
    # f_hi : Change of HI_o depending on the stress. #f_hi is affected by water stress, but water stress is already taken into account.
    # Dennis: this can be affected by air temperature stress
    #       so the comment below isnt true when air temp is implemented
    f_hi = 1.  # I THINK we can leave this out - value of 1 means no change to Harvest Index (HI) value.
    t_grow = 195  # growth time from t_CCo until senescence or harvest #Total growth
    t_Ss = 30  # time of senescence

    Z_max = min(Constants['soil_depth'], 1400)  # max effective rooting depth [mm]
    Z_n = 300  # minimum root depth, check: the minimum mentioned in the appendix is 0.3m which is for the most part more than the soil depth
    Z_o = Z_n / 2  # starting depth of root zone expansion
    n = 1.5  # shape factor of root zone development, from aquacrop annex

    Topsoil = 150  # mm, evaporable region of soil
    SM_fc_ts = Topsoil * FC
    SM_fc_ss = (Z_max - Topsoil) * FC
    SM_wp_ts = Topsoil * WP
    SM_wp_ss = (Z_max - Topsoil) * WP
    SM_fc_max = SM_fc_ts + SM_fc_ss
    SM_wp_max = SM_wp_ts + SM_wp_ss
    TAW_max = (SM_fc_ts - SM_wp_ts) + (SM_fc_ss - SM_wp_ss)

    Sumt_ts = max(0, min(YieldSumt[0], SM_fc_ts))
    Sumt_ss = max(0, min(YieldSumt[1], SM_fc_ss))

    alpha = Topsoil / (Topsoil + Z_max)  # fraction of transpiration taken from topsoil

    # Soil specific
    Kb = 0.6  # For soil evaporation, fraction of potential evaporation that evaporates from the soil
    K_topsoil = 8.64  # permeability from topsoil for silty clay texture mm/d

    # Kcb
    # t parameters taken from fao evapotranspiration manual: http://www.fao.org/3/X0490E/x0490e0b.htm#TopOfPage
    t_ini = 30  # duration of initial stage (t until 10% of ground covered)
    t_dev = 50  # duration of development stage
    t_mid = 60  # duration after t_dev to plant maturity
    t_late = 55  # duration from maturity to harvest

    Kcb_ini = 0.15
    Kcb_mid = 1.1
    Kcb_end = 0.4

    # Kcb_grass = 0.75*np.zeros(366)# Grass Kc = 0.75
    Kcb_cotton = np.zeros(366)
    Kcb_cotton[plantdate - 1:plantdate + t_ini - 1] = Kcb_ini
    Kcb_cotton[plantdate + t_ini - 1:plantdate + t_ini + t_dev - 1] = np.linspace(Kcb_ini, Kcb_mid, t_dev)
    Kcb_cotton[plantdate + t_ini + t_dev - 1:plantdate + t_ini + t_dev + t_mid - 1] = Kcb_mid
    Kcb_cotton[plantdate + t_ini + t_dev + t_mid - 1:plantdate + t_ini + t_dev + t_mid + t_late - 1] = np.linspace(
        Kcb_mid, Kcb_end, t_late)
    Kcmax_cotton = Kcb_cotton + 0.05
    # Kcmax_cotton=np.zeros(366)
    # for i in range(len(Kcb_cotton)):
    #     if Kcb_cotton[i]!=0:
    #         Kcmax_cotton[i]=Kcb_cotton[i]+0.05

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
    SM_fc = np.zeros(len(precip1))  # field capacity in terms of SM [mm]
    SM_wp = np.zeros(len(precip1))  # wilting point in terms of SM [mm]
    # TAW_ts = np.zeros(len(precip1))        #total water available topsoil (sm_fc-sm_wp) [mm]
    # TAW_ss = np.zeros(len(precip1))        #total water available subsoil (sm_fc-sm_wp) [mm]
    De_ts = np.zeros(len(precip1))  # depletion topsoil [mm]
    De_ss = np.zeros(len(precip1))  # depletion subsoil [mm]
    De = np.zeros(len(precip1))  # depletion total [mm]
    Runoff = np.zeros(
        len(precip1))  # runoff from field (not used for anything in this model yet, just recording the water lost)
    Irr_vol = np.zeros(len(precip1))  # Irrigation total volume
    Irr = np.zeros(len(precip1))  # Irrigation in mm per m2
    # values taken from Allen et al : https://www.researchgate.net/publication/228721933_Estimating_Evaporation_from_Bare_Soil_and_the_Crop_Coefficient_for_the_Initial_Period_Using_Common_Soils_Information
    TEW = 35  # maximum evaporable water for Z ~ 0.1 to 0.15 m
    REW = 10  # readily evaporable water

    # soil moisture and reservoir storage
    SM_ts = np.zeros(len(precip1) + 1)
    SM_ss = np.zeros(len(precip1) + 1)
    SM = np.zeros(len(precip1) + 1)
    SM_ts[0] = Sumt_ts
    SM_ss[0] = Sumt_ss
    SM[0] = Sumt_ts + Sumt_ss
    ResStorage = np.zeros(len(precip1) + 1)
    CC[plantdate + t_CCo] = CC_o

    # t_CCo=7 #growth days - or t after planting when CC_o occurs
    # t_grow=170
    # precip1 = precip1.ix[:,0] #to index over numbers

    stage2 = False
    stageSen = False
    tz = 0  # t counter for root zone starts at t_CCo/2
    t_x = 100  # t to reach max root depth, check the number, now just based on the cc graph, from planting to max cc
    t1 = 0  # t counter after growing starts
    t2 = 0  # t counter after senecsence starts
    GDD1 = np.zeros(len(precip1))  # GDD for growing stages
    GDD2 = np.zeros(len(precip1))  # GDD after senescense
    # GDD3=np.zeros(len(precip1))

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
    # res_max=res_max*1000 #m3 to mm m2
    # well_area=np.pi * 1.5**2 #well with diameter 3m
    # well_max=well_area*10*1000 #10m deep and changing from m3 to mm m2
    # # res_resol=0.1*1000000 #km2 to m2
    # SumIrr=0

    # max number of days during which field capacity can be exceeded after high infiltration rates
    # k_Rt=3 #days   

    m = np.zeros(len(precip1))  # daily biomass obtained
    m_max = np.zeros(len(precip1))
    # cco_bool=0
    for i in range(len(precip1)):  #

        ResStorage[0] = YieldRes

        ##INTERCEPTION
        I = np.minimum(precip1[i], D)

        ##(IN)FILTRATION
        F = precip1[i] - I

        # Rt[i] = min(SM[i]-SM_fc_max/k_Rt,0)

        # if Constants['soil_depth']>=1400:
        #     Rt[i] = min(SM[i]-SM_fc_max/k_Rt,0)
        # else:
        #     Rt[i]=0

        # root zone
        t_zo = round(t_CCo / 2)
        if plantdate < i <= plantdate + int(t_grow - t_Ss):
            # Root zone growth
            tz += 1
            if tz - t_zo <= 0:
                Z_eff[i] = min(Z_n, Z_max)
            else:
                Z_eff[i] = min(max(Z_o + (Z_max - Z_o) * ((tz - t_zo) / (t_x - t_zo)) ** (1 / n), Z_n), Z_max)

                # with root zone stress
                # dz_eff=-((Z_max-Z_o)*((tz-t_zo)/(t_x-t_zo))**(1/n))/(n*(t_zo-tz))

                # convex shape (not too strong effects)
                # dz_eff_adj=dz_eff*((np.exp(Ks_sto[i-1]*f_shape_z)-1)/(np.exp(f_shape_z)-1))

                # linear (strong effects)
                # dz_eff_adj=dz_eff*Ks_sto[i-1]

                # Z_eff[i]=min(max(Z_eff[i-1]+dz_eff_adj,Z_n),Z_max)

                # Z_eff[i]=min((tz-t_zo)*Z_o+(Z_max-Z_o)*((tz/(t_x-t_zo))**(1/n)),Z_max)
                # Z_eff[i]=min(max(tz-t_zo)Z_o+(Z_max-Z_o)*((t/(t_x-t_zo))**(1/n),Z_n),Z_max)

        SM_fc[i] = FC  # Z_eff[i] * FC #Field capacity
        SM_wp[i] = WP  # Z_eff[i] * WP #wilting point

        TAW_ts = SM_fc_ts - SM_wp_ts
        TAW_ss = SM_fc_ss - SM_wp_ss
        Dr_upper = P_upper * TAW_max
        Dr_lower = P_lower * TAW_max
        De_ts[i] = SM_fc_ts - SM_ts[i]
        De_ss[i] = SM_fc_ss - SM_ts[i]
        De[i] = De_ts[i] + De_ss[i]

        # P_upper_sto = 0.65
        # Dr_upper_sto = P_upper_sto * TAW[i]

        if plantdate + t_CCo < i <= plantdate + int(t_grow - t_Ss):  # and cco_bool==1:  #

            # GDD1[i]=GDD1[i-1]+max(T[i]-T_base,0)
            t1 += 1

            # water stress
            if SM_fc_max - De[i] >= Dr_upper:
                Ks[i] = 1.

            elif SM_fc_max - De[i] <= Dr_lower:
                Ks[i] = 0.

            elif Dr_lower < SM_fc_max - De[i] < Dr_upper:

                # linear
                Ks[i] = (SM_fc_max - De[i] - Dr_lower) / (Dr_upper - Dr_lower)

                # convex
                # S_rel=1-((SM_fc[i] - De[i] - Dr_lower)/(Dr_upper-Dr_lower))
                # Ks[i]=1-(np.exp(S_rel*f_shape)-1)/(np.exp(f_shape)-1)

            # water stress in the root zone
            # if SM_fc[i] - De[i] >= Dr_upper_sto:
            #     Ks_sto[i]=1.

            # elif SM_fc[i] - De[i] <= 0:
            #     Ks_sto[i]=0.

            # elif 0 < SM_fc[i] - De[i] < Dr_upper_sto:
            #     Ks_sto[i]=1-(SM_fc[i] - De[i])/(Dr_upper_sto)   

            ####################STEP 5########################
            CGC_adj[i] = Ks[i] * CGC  # [adjust the canopy growth coefficient to water stress]

            ####################STEP 6########################
            ###Calculating the crop growth###
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

            if (De_ts[i]) <= REW:
                Kr[i] = 1
            elif (De_ts[i]) > REW:
                Kr[i] = max((TEW - (De_ts[i])) / (TEW - REW), 0)
            Ke[i] = Kcmax_cotton[i] - Kcb_cotton[i]
            ETc[i] = (Kcb_cotton[i] + Ke[i]) * evap1[i]

            # SOIL EVAP AND TRANSPIRATION
            Es[i] = Kr[i] * (1 - CC[i]) * Ke[i] * evap1[i]
            Tp[i] = Ks[i] * Kcb_cotton[i] * CC[i] * evap1[i]

        elif plantdate + (t_grow - t_Ss) < i <= plantdate + t_grow:  # len(precip1):
            t2 += 1
            if t2 == 1:
                CCsen = CC[i - 1]

            # using days
            # dCC = -0.05*CDC*np.exp(CDC*t2/CCsen)
            # CC[i]=np.maximum(CC[i-1]+dCC,0)        
            CC[i] = max(CCsen * (1 - 0.05 * (np.exp(CDC / CCsen * t2) - 1)), 0)

            ##LEAF AREA INDEX
            # https://www.researchgate.net/publication/266895709_Parameterization_and_Evaluation_of_FAO_AquaCrop_Model_for_Full_and_Deficit_Irrigated_Cotton
            # faharani et al 2009
            LAI[i] = 100. / 77. * np.log(1. / (1. - CC[i]))

            # Remove stress factor
            # equation from: https://www.researchgate.net/publication/228721933_Estimating_Evaporation_from_Bare_Soil_and_the_Crop_Coefficient_for_the_Initial_Period_Using_Common_Soils_Information
            if (De_ts[i]) <= REW:
                Kr[i] = 1
            elif (De_ts[i]) > REW:
                Kr[i] = max((TEW - (De_ts[i])) / (TEW - REW), 0)
            Ke[i] = Kcmax_cotton[i] - Kcb_cotton[i]
            ETc[i] = (Kcb_cotton[i] + Ke[i]) * evap1[i]

            # SOIL EVAP AND TRANSPIRATION
            Es[i] = Kr[i] * (1 - CC[i]) * Ke[i] * evap1[i]
            Tp[i] = Ks[i] * Kcb_cotton[i] * CC[i] * evap1[i]


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

        # Using pond to irrigate
        if i > plantdate and i < plantdate + (t_grow):
            PlantSeason = 1
        else:
            PlantSeason = 0
        IrrPar = [add_water, total_area, irr_system, Z_max, PlantSeason]
        WBPar = [precip1[i], evap1[i], Inflow.iloc[i], SM_ts[i], TAW_ts, ResStorage[i], SM_wp_ts]

        IrrMat = Irrigation(res_parameter, IrrPar, WBPar)
        Irr_vol[i] = IrrMat[0]
        Irr[i] = Irr_vol[i] / total_area
        ResStorage[i + 1] = IrrMat[1]
        Eo[i] = IrrMat[2]

        ## total demand
        Ed = Tp[i] + Es[i]

        ## demand met
        Em = np.minimum(min(TAW_max, SM_ts[i] + SM_ss[i] + F), Ed)

        # condition for topsoil downward flux
        if SM_ts[i] < 0.1 * SM_fc_ts:
            fts = 0
        else:
            if SM_ts[i] > SM_fc_ts:
                fts = 1
            else:
                if 0.1 * SM_fc_ts <= SM_ts[i] <= SM_fc_ts:
                    fts = (SM_ts[i] - (0.1 * SM_fc_ts)) / (SM_fc_ts - (0.1 * SM_fc_ts))

        # change in scl due to flux
        if SM_ss[i] == SM_fc_ss:
            L_ts = 0
        else:
            L_ts = min(K_topsoil * fts, SM_fc_ss - SM_ss[i])
            L_ts = max(0, min(K_topsoil * fts, SM_fc_ss - SM_ss[i]))

        if Ed != 0:
            ## Transpiration possible: in proportion to demands
            Ta[i] = Tp[i] * Em / Ed

            ## Soil evaporation possible
            Ea[i] = Es[i] * Em / Ed
        else:
            Ta[i] = 0
            Ea[i] = 0

        ##SOIL MOISTURE
        Ta_ts = alpha * Ta[i]
        Ta_ss = (1 - alpha) * Ta[i]
        SM_tmp_ts = np.maximum(0, np.minimum(SM_ts[i] + F - Es[i] - Ta_ts + Irr[i] - L_ts, SM_fc_ts))
        SM_tmp_ss = np.maximum(0, np.minimum(SM_ss[i] - Ta_ss + L_ts, SM_fc_ss))
        Runoff[i] = max(SM_ts[i] + F - Es[i] - Ta_ts + Irr[i] - SM_fc_ts, 0)

        # ##RECHARGE
        # R = (1/K_sat)*np.maximum(SM_tmp-SM_fc,0)

        ##FINAL SOIL MOISTURE
        SM_ts[i + 1] = SM_tmp_ts
        SM_ss[i + 1] = SM_tmp_ss
        SM[i + 1] = SM_tmp_ts + SM_tmp_ss

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
        Mat[i] = np.array([Ta[i], Ea[i], Em, SM[i], SM_ts[i], SM_ss[i], F, Ks[i], Ks_T[i], LAI[i],
                           CC[i], L_ts, Tp[i], Es[i], Ed, evap1[i], ETc[i],
                           Z_eff[i], ETc[i], Ke[i], Kr[i], De[i], Runoff[i], Irr[i],
                           SM_fc[i], SM_wp[i], Rt[i], Ta_ts, Ta_ss, m[i], m_max[i]])
    B = np.sum(m)
    B_max = np.sum(m_max)

    # print(Constants['max_Crop_yield'])
    conversion = 10000 / 1000  # changing the unit from g/m2/y to kg/ha/y
    Yield = f_hi * HI_o * B * conversion  # Yield; kg/ha/y
    Constants['max_Crop_yield'] = HI_o * B_max * conversion

    YieldSumt[0] = SM_ts[-1]
    YieldSumt[1] = SM_ss[-1]
    SumIrr = np.sum(Irr)
    YieldRes = ResStorage[-1]

    # print(np.sum(Ta[plantdate+t_CCo:plantdate+t_grow]),np.sum(WD_cotton[plantdate+t_CCo:plantdate+t_grow]))
    # if Constants['max_Crop_yield']<Yield:
    # print(Constants['max_Crop_yield'])

    return Mat, YieldSumt, SumIrr, YieldRes, Yield
