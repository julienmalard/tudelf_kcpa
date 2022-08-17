# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 17:07:44 2015

@author: Nadezhda
"""

##Import Mododules needed

##import homemade Modules
# from globalparameters import *
# from globalparameters import Parameters, Constants

from Expenditure import *
from Net_household_income import *
from WB_Yield import WB_Yield
from crop_cost_balance import *
from crop_cost_compute import *
from livestock_costs_balance import *
from livestock_costs_compute import *


def Householdmodel(precip, evap, T, Constants, Parameters, Kc_cotton,
                   Kc_grass, Price_crop, Price_fert, start_year, Tsimul, res_parameter,
                   total_area, wage=0, Smx=300, IC=50000, fert=66, crop_area=2,
                   other_area=0, livestock=2, Family_size=6, irr_system=None,
                   loan_debt=25000, interest_rate=0.08, add_water=None, iv_max=None,
                   subsidies=0, Par=None):
    'This is the householdmodel version #1 in python based, this function simulates five states defining the socio-hydrological situation of a smallholder. If you want to reference to this model reference to Pande and Savenije (2016) and Den Besten et al (2016), doi will follow'
    # start_year = max(min(precip.index.year),min(evap.index.year))
    # Tsimul = int(min(max(precip.index.year),max(evap.index.year)) - start_year + 1)

    # Parameters
    # mandays
    mandays = 250.
    # Householdmandays
    H_mandays = Family_size * mandays
    # Savinginterestrate
    saving_interest_rate = 0.  # so if it is zero why put that in?
    # depreciationrate
    dep_rate = 0.03  # assuming 3%
    # rateofdepriciation
    delt = 0.05
    # length_L
    length_L = 100.  # used in soil fertility equations
    # inflationrate
    inf_rate = 0.  # causes weird stuff to happen if not zero

    # Specify which crop!!!
    #     WD_cotton = WD['Water demand cotton'] #WD_cotton = WD['Water demand cotton']
    #     Grass_demand = WD['Water demand grass'] #change this one to grass!!

    # Creating empty array to store state variables
    VarMat = np.zeros((Tsimul + 1, 5))
    WbMat = np.zeros((Tsimul, 3))  # Crop yield, Sum irrigation, sum crop transpiration, soil moisture
    Yield = np.zeros((Tsimul))
    # YieldSumt = [50*(Constants['soil_depth']/541.5),80*(Constants['soil_depth']/541.5)] #initial storage in topsoil and subsoil respectively if higher than field capacity the excess will be removed
    # YieldSumt = [Par[0]*(Constants['soil_depth']/541.5),Par[1]*(Constants['soil_depth']/541.5)] #initial storage in topsoil and subsoil respectively if higher than field capacity the excess will be removed
    # YieldSumt = [50,80] #initial storage in topsoil and subsoil respectively if higher than field capacity the excess will be removed
    YieldSumt = 50
    YieldRes = 0
    # Sumt=0

    # grasslandarea
    grassland_area = max(0.24 * livestock,
                         crop_area * 0.2)  # average livelstockholding in Aurangabadis 2,0.24ha/milchanimal of fodderarea in MH. Otherwise it's 20% of cropland
    frac = crop_area / (crop_area + grassland_area)
    Constants["Crop_area"] = crop_area
    add_income = other_area * 50000

    # Dictionary values that need to be calculated beforehand:
    Constants['Price_per_manday'] = wage
    Constants['Su_max'] = Smx
    Constants['max_crop_labour'] = 260. * crop_area
    Parameters['max_crop_labour'] = 260. * crop_area
    Constants['crop_area'] = crop_area
    Parameters['crop_area'] = crop_area

    # commercial fertilizers
    Constants['commercial_fertilizer'] = min(fert, Constants['max_N_app']) * crop_area

    # These are very important
    crop_labour_in = 0.68 * H_mandays
    Off_farm_agric_labour_in = 0.10 * H_mandays
    Off_farm_non_agric_labour_in = 0.15 * H_mandays

    ##Creating empty arrays   
    Mat = np.zeros((12, 7))
    Smat = np.zeros((1, 7))
    GMat = np.zeros(2)
    Fertmat = np.zeros(
        6)  # {'soil fertilityt','nitrogen fixation','commercial fertilizer','manures','natural losses','plant uptake'}
    Natlmat = np.zeros(3)  # {'erosion loss','leaching','gaseous losses'}
    Soermat = np.zeros(3)  # {'crop management factor','standard mean soil loss','topographic factor'}

    # preserving the original set of prices
    Price_for_chemicals_o = Constants['Price_for_chemicals']
    Price_of_crop_o = Constants['Price_of_crop']  # in Rs/kg; 2000 Rs/quintal, was 145*20
    # Price of fertilizer
    Price_of_fertiliser_o = Constants['Price_of_fertiliser']  # Rs/kg of Urea (india)
    # Price of livestock
    Price_of_livestock_o = Constants['Price_of_livestock']  # 7000, Price of cow in MH, Rs.

    yr1 = np.arange(start_year - 1, start_year + Tsimul)

    _YieldMat = []

    for t in range(Tsimul + 1):
        # print(t,Tsimul)
        # Initialize conditions:
        if t == 0:
            VarMat[0, 0] = IC  # Capitalt
            VarMat[0, 1] = 25.  # Knowledget
            VarMat[0, 2] = livestock  # Livestockt
            VarMat[0, 3] = 25. * grassland_area  # Grasst
            VarMat[0, 4] = 500.  # soil_fertilityt # mass balance run at monthly and summed over the year

        else:
            payl2 = loan_debt * interest_rate
            if np.isnan(loan_debt) or np.isnan(interest_rate):
                payl2 = 0

            # For the farmer to make new decisions each year:
            Constants['Off_farm_agric_labour'] = Off_farm_non_agric_labour_in
            Parameters['Off_farm_agric_labour'] = Off_farm_non_agric_labour_in
            Constants['Off_farm_non_agric_labour'] = Off_farm_agric_labour_in
            Parameters['Off_farm_non_agric_labour'] = Off_farm_agric_labour_in
            Constants['crop_labour'] = crop_labour_in
            Parameters['crop_labour'] = crop_labour_in

            #### HYDROLOGY ####
            # Hydrology calculated at daily basis
            # assuming no deep percolation. Black clay soils  

            precip1 = precip[(precip.index.year == yr1[t])]
            Constants['annual_rainfall_P'] = np.sum(precip1)
            evap1 = evap[evap.index.year == yr1[t]]
            T1 = T[T.index.year == yr1[t]]
            if len(evap1) == 365:
                WD_cotton = Kc_cotton[1:] * evap1
                WD_grass = Kc_grass[1:] * evap1
            else:
                WD_cotton = Kc_cotton * evap1
                WD_grass = Kc_grass * evap1

            res_tmp_y = res_parameter[0][res_parameter[0]['YEAR'] == yr1[t]]
            # print(res_tmp_y)
            # print(yr1[t])
            # Water Balance and Yield using Water Stress
            YieldMat, YieldSumt, WbMat[t - 1, 1], YieldRes, Yield[t - 1] = WB_Yield(Constants, precip1, evap1, T1,
                                                                                    res_tmp_y, res_parameter, crop_area,
                                                                                    grassland_area, total_area, frac,
                                                                                    YieldSumt, YieldRes, WD_cotton,
                                                                                    Kc_cotton, irr_system=irr_system,
                                                                                    reservoir=0, add_water=add_water,
                                                                                    Par=Par)

            a = np.zeros((np.shape(YieldMat)[0], np.shape(YieldMat)[1] + 1))
            a[:, :-1] = YieldMat
            a[:, -1] = yr1[t]
            _YieldMat.extend(a)

            # YieldMat= [actual transp,actual soil evap,demand met,soil moisture,SM_tmp,water stress,leaf area index,canopy cover]
            # YieldSumt= [transpiration,soil evap,potential evap,soil evap factor,wilting point,field capacity,water demand cotton]

            # # # Water Balance:
            # SmatM,Sumt,WbMat[t-1,1] = Storage(Constants,precip1,frac,Sumt,WD_cotton,WD_grass,irr_system=irr_system,add_water=add_water,iv_max=iv_max)

            # Evapotranspiration for crop
            # Dennis: crop transpiration
            Constants['Ta_Tm'] = np.sum(YieldMat[157:327, 0]) / np.sum(WD_cotton[157:337])
            WbMat[t - 1, 2] = np.sum(YieldMat[157:327, 0])

            # Constants['Ta_Tm'] = np.sum(SmatM[157:337,1])/np.sum(WD_cotton[157:337])         
            # WbMat[t-1,2] = np.sum(SmatM[157:337,1])
            # Evapotranspiration for grass
            Constants['ETa_ETm'] = np.sum(
                WD_grass)  # assuming grass demand always met, old: np.sum(SmatM[:,2])/np.sum(WD_grass)

            #### CROP COSTS ####
            # Reading prices of crops and fertilizer
            Constants['Price_of_crop'] = Price_crop[t - 1]
            Constants['Price_of_fertiliser'] = Price_fert[t - 1]

            # original crop costs
            if t == 0:
                crop_cost_compute(Constants, Parameters, crop_area)  # update calculations;
                Constants['tax'] = Constants['tax_rate'] * Constants['Farm_Sales']

            Constants, N = Net_household_income(Constants, VarMat[t - 1, 2], Yield[t - 1], VarMat[t - 1, 4], Par=Par)

            # dennis: yield unit is kg/ha
            WbMat[t - 1, 0] = Constants['Crop_yield']

            Parameters, Constants, E = Expenditure(Constants, Parameters)

            # Intervention costs
            if irr_system is not None:
                # isys_costs = ((1-subsidies)*irr_system['fp_price']*(irr_system['interest_rate']+1./50.)+irr_system['price']*(irr_system['interest_rate']+1./irr_system['life_time'])+irr_system['upkeep']+irr_system['op_cost']*WbMat[t-1,1])*Constants["Crop_area"]
                isys_costs = (irr_system['upkeep'] + irr_system['op_cost'] * WbMat[t - 1, 1]) * Constants["Crop_area"]
                # print(isys_costs)
            else:
                isys_costs = 0

            #### CAPITAL DIFFERENTIAL EQUATION ####
            # delt=depreciation,N = income (INR), E=expenses, payl2=loan payment, isys_costs=intervention cost
            VarMat[t, 0] = (1 - delt) * VarMat[t - 1, 0] + N - E - payl2 - isys_costs + add_income
            # print(VarMat[t,0],)
            # if add_water is None:
            #     print(N,E)

            #### EXPENDITURE CUT EQUATIONS #### ========================================================
            expCut_o = np.array([Constants['investment'], Constants['school_fees'], Constants['Food_bought'],
                                 Constants['Interest_on_loan'], Constants['tax'], Constants['other_expense'],
                                 Constants['livestock_costs'], Constants['crop_cost']])
            #            

            Constants['livestock_sold'] = 0
            # adapt if Capitaltp1 <0
            if VarMat[t, 0] < 0:  # Capitaltp1 <0 :
                # bring expenditure close to income
                Constants['deficit'] = VarMat[t, 0]  # Capitaltp1

                # Sell Livestock to offset deficit
                Constants['livestock_sold'] = min(-Constants['deficit'] / Constants['Price_of_livestock'],
                                                  VarMat[t - 1, 2])
                Constants['deficit'] = Constants['deficit'] + Constants['livestock_sold'] * Constants[
                    'Price_of_livestock']
                expCut = expCut_o

                # Cut expenditures to offset deficit
                for i in range(len(expCut)):
                    if Constants['deficit'] >= 0:
                        break
                    Constants['deficit'] = Constants['deficit'] + expCut[i]
                    expCut[i] = np.maximum(Constants['deficit'], 0)
                    Constants['deficit'] = np.minimum(Constants['deficit'], 0)

                    if i >= len(expCut) - 1:
                        #                         print('Debt is bigger than expenditure cuts')
                        break

                # assign parameter structure of crop costs to pars;
                Parameters, Constants = crop_cost_balance(Parameters, Constants, expCut[7])

                # pause
                Parameters, Constants = crop_cost_compute(Constants, Parameters)  # update calculations

                Parameters, Constants = livestock_costs_balance(Parameters, Constants, expCut[6])  # update parameters;

                Constants = livestock_costs_compute(Parameters, Constants)  # update calculations

                # update the parameter structure
                Parameters['investment'] = expCut[0]

                Constants['savings'] = Parameters['investment']
                Parameters['school_fees'] = expCut[1]

                Parameters['Food_bought'] = expCut[2]

                if Constants['loan_size'] > 0 and expCut[3] < expCut_o[3]:
                    Parameters['loan_interest_rate'] = expCut[3] / Constants['loan_size']

                if Constants['Farm_Sales'] > 0 and expCut[4] < expCut_o[4]:
                    Parameters['tax_rate'] = expCut[4] / Constants['Farm_Sales']

                Parameters['other_expense'] = expCut[5]

            ##############################################################################
            #### LIVESTOCK EQUATIONS ####

            # Livestock
            Constants['liv_Quantity_purchases'] = np.copy(Parameters['liv_Quantity_purchases'])
            Constants['veterinary_costs'] = np.copy(Parameters['veterinary_costs'])

            # consuming
            consuming = np.fmin(VarMat[t - 1, 3], (
                        VarMat[t - 1, 2] * Constants['households_with_livestock'] * Constants['feed_requirement'] * (
                            1. - Constants['feed_residues_rate'])))

            # Grass per livestock
            grass_per_livestock = consuming / np.fmax(Constants['households_with_livestock'] * VarMat[t - 1, 2], 1)
            #
            # Conversion factor
            conversion_factor = np.fmax(
                (Constants['the_weight_gain_rate'] * (grass_per_livestock - Constants['feed_maintaining_requirement'])),
                0)

            # using logistic growth function
            # carrying capacity
            carr_cap = VarMat[t - 1, 3] / Constants['feed_requirement'] * (1. - Constants['feed_residues_rate'])

            # LIVESTOCK DIFFERENTIAL EQUATION
            VarMat[t, 2] = np.fmax((VarMat[t - 1, 2] + (Constants['birth_rate'] + conversion_factor) * VarMat[
                t - 1, 2] * (1. - VarMat[t - 1, 2] / carr_cap) + Constants['liv_Quantity_purchases'] - Constants[
                                        'livestock_sold']), 0)
            Constants, N = Net_household_income(Constants, VarMat[t - 1, 2], Yield[t - 1], VarMat[t - 1, 4], Par=Par)

            Parameters, Constants, E = Expenditure(Constants, Parameters)

            #### CAPITAL DIFFERENTIAL EQUATION ####
            # Dennis: payl2 is loan interest rate, isnt it already taken into account in E?
            # Dennis: i think there is something else substracted here but i dont know, check it
            VarMat[t, 0] = (1 - delt) * VarMat[t - 1, 0] + N - E - payl2 - isys_costs
            # old capital * deprication + Net income - expenditures - loan shark interest - intervention costs

            Constants['tax_rate'] = 0.2

            #### KNOWLEDGE DIFF EQ. ####
            obsolete = VarMat[t - 1, 1] * Constants['knowledge_obsolete_rate']
            learning = Constants['Farming_experience_gained'] + Constants['education']

            VarMat[t, 1] = VarMat[t - 1, 1] + learning - obsolete

            #### NATURAL RESOURCE USE ####
            # Grass water factor
            grass_water_factor = np.maximum(0., np.minimum(1., 1. - Constants['Kygrass'] * (1. - Constants['ETa_ETm'])))
            # generating
            generating = Constants['grassland_area'] * grass_water_factor * Constants['max_grass_yield']
            # Grass
            VarMat[t, 3] = np.minimum(VarMat[t - 1, 3] + generating - consuming,
                                      Constants['grassland_area'] * Constants['max_grass_yield'])
            # Grassmatrix
            GMat = np.array([generating, consuming])

            #### SOIL FERTILITY ####

            # rainfall energy
            rainfall_energy_E = Constants['annual_rainfall_P'] * (
                        11.9 + 8.7 * np.log10(Constants['rainfall_intensity_I']))  # is this working properly?
            # mean soil loss
            standard_mean_soil_loss_K = np.exp((0.4681 + 0.7663 * Constants['soil_erodibility_factor_F']) * np.log(
                rainfall_energy_E) + 2.884 - 8.2109 * Constants[
                                                   'soil_erodibility_factor_F'])  # not sure about this exp and log syntax
            # topographic factor
            topographic_factor_X = np.sqrt(length_L) * (
                        0.76 + 0.53 * Constants['slope_s'] + 0.076 * Constants['slope_s'] * Constants[
                    'slope_s']) / 25.65  # twee keer slope_s??!!
            # soil erosion Z
            soil_erosion_Z = Constants['crop_management_factor_C'] * standard_mean_soil_loss_K * topographic_factor_X
            Soermat = np.array([Constants['crop_management_factor_C'], standard_mean_soil_loss_K, topographic_factor_X])

            # EROSION LOSS
            erosion_loss = VarMat[t - 1, 4] * Constants['enrichment_ratio'] * soil_erosion_Z / Constants[
                'soil_density'] / 10000. / Constants['top_soil_depth']
            # natural losses
            natural_losses = erosion_loss + Constants['leaching'] + Constants['gaseous_losses']
            Natlmat = [erosion_loss, Constants['leaching'], Constants['gaseous_losses']]

            # GRAIN UPTAKE
            # for sugarcane
            grain_uptake = Constants['Crop_yield'] * Constants['N_concentration_in_grain']
            # Straw uptake
            straw_uptake = Constants['Crop_yield'] * (1 / Constants['harvest_indext'] - 1) * Constants[
                'N_concentration_in_straw']
            # plant uptake
            plant_uptake = grain_uptake + straw_uptake

            # NITROGEN FIXATION
            nitrogen_fixation = Constants['biological_fixation'] + Constants['rain_fixation']
            # Constants['fertilisation']
            Constants['fertilisation'] = nitrogen_fixation + Constants['commercial_fertilizer'] + Constants['manures']

            # SOIL FERTILITY DIFFERENTIAL EQUATION
            VarMat[t, 4] = np.fmax(VarMat[t - 1, 4] + Constants['fertilisation'] - plant_uptake * crop_area,
                                   0)  # natural_losses

            Fertmat = np.array(
                [VarMat[t - 1, 4], nitrogen_fixation, Constants['commercial_fertilizer'], Constants['manures'],
                 natural_losses, plant_uptake])

            #### UPDATING PARAMETERS ####
            Parameters['investment'] = Constants['investment']
            Parameters['school_fees'] = Constants['school_fees']
            Parameters['Food_bought'] = Constants['Food_bought']
            Parameters['Interest_on_loan'] = Constants['Interest_on_loan']
            Parameters['tax'] = Constants['tax']
            Parameters['other_expense'] = Constants['other_expense']
            Parameters['livestock_costs'] = Constants['livestock_costs']
            Parameters['crop_cost'] = Constants['crop_cost']

            Parameters['labour_domestic'] = Constants['labour_domestic']
            Parameters['labour_firewood_collection'] = Constants['labour_firewood_collection']
            Parameters['labour_water_collection '] = Constants['labour_water_collection']
            Parameters['crop_labour'] = Constants['crop_labour']
            Parameters['livestock_labour'] = Constants['livestock_labour']
            Parameters['Chemicals'] = Constants['Chemicals']
            Parameters['commercial_fertilizer'] = Constants['commercial_fertilizer']
            Parameters['liv_Quantity_purchases'] = Constants['liv_Quantity_purchases']
            Parameters['investment'] = Constants['investment']
            Parameters['tax_rate'] = Constants['tax_rate']
            Parameters['loan_interest_rate'] = Constants['loan_interest_rate']

            # inflation adjusted wage rate per manday
            Constants['Price_per_manday'] = Constants['Price_per_manday'] * (1 + inf_rate - inf_rate)
            Constants['Price_for_chemicals'] = Constants['Price_for_chemicals'] * (1 + inf_rate)
            Constants['Price_of_crop'] = Constants['Price_of_crop'] * (1 + inf_rate - 0.)
            Constants['Price_of_crop'] = np.maximum(Constants['Price_of_crop'], 0.)
            Constants['Price_of_fertiliser'] = Constants['Price_of_fertiliser'] * (1. + inf_rate)
            Constants['Price_of_livestock'] = Constants['Price_of_livestock'] * (1 + inf_rate)
            Constants['Price_of_livestock'] = np.maximum(Constants['Price_of_livestock'], 0.)

            # discount future income stream to present
            # NPV = NPV + VarMat[-1,1]/((1+np.maximum(Constants['saving_interest_rate'],0))**(t-1)); #in matlab:NPV = NPV + VarMat(end,1)/((1+max(saving_interest_rate,0))^(t-1));
            # Proportion of time below 30K, minimum standard of living

            # ProbThreshLow = np.mean(VarMat(-1,0)-Thresh<0)

    return VarMat, WbMat, Yield, _YieldMat
