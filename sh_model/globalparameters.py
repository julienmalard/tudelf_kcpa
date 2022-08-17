# -*-coding:utf-8-*-
"""
CreatedonThuAug2016:34:142015

@author:Nadezhda
"""

##Useful link: http://www.dotnetperls.com/dictionary-python

import numpy as np

##Parameters
##mandays
# mandays=250.
##Familysize
# Family_size=6.
##Householdmandays
# H_mandays=Family_size*mandays
##Savinginterestrate
# saving_interest_rate=0.#soifitiszerowhyputthatin?
##depreciationrate
# dep_rate=0.03#assuming3%
##rateofdepriciation
# delt=0.05#was0.05-->whatdoesthismean?
##length_L
# length_L=100.#lengthof?
##inflationrate
# inf_rate=0.#assuminginflationratetobe5%sowhynot0.05?


Original = {'crop_area': 2.,
            'birth_rate': 0.15,
            'Chemicals': 0.,
            'crop_labour': 0.68 * 1500.,  # H_mandays.
            'crop_for_home_consumption': 0.,
            'death_rate': 0.15,
            'education': 0.,
            'Family_size': 6.,
            'Farming_experience_gained': 10.,
            'feed_maintaining_requirement': 0.,
            'fert_N_factor': 0.50,  # 0.40, # 46% for Urea and DAP, 60% for Potassium. Unknown what proportion is used
            'Food_bought': 0.,
            'Gifts': 0.,
            'investment': 10000.,
            'knowledge_obsolete_rate': 0.2,
            'Ky': 0.85,
            # yield response factor, from FAO for cotton, <1 means tolerant to water deficit, >1 sensitive, =1 is reduction is directly proportional to water reduction
            'labour_domestic': 0.01 * 1500.,  # H_mandays.
            'livestock_labour': 0.04 * 1500.,  # H_mandays.
            'Labour_factor': 0.4,
            'labour_firewood_collection': 0.01 * 1500.,  # H_mandays
            'labour_water_collection': 0.01 * 1500.,  # H_mandays
            'Labour_balance': 0.,
            'livestock_retained': 0.,
            'liv_Quantity_purchases': 0.25,
            'loan_interest_rate': 0.35,
            'loan_size': 0.,
            'mandays': 250.,
            'max_crop_labour': 260.,
            'max_Crop_yield': 3750.,  # 15 quintals/acre* 2.5 acres/ha* 100kg/quintal# 1000.*0.35,
            'max_N_app': 156.,
            # 156kg N, 36 kg P2O5, 151 kg K2O # http://www.fao.org/3/a0443e/a0443e04.pdf # Was 250 before
            'min_Crop_yield': 375,  # 10% seems reasonable # 0*0.35,
            'Nmin_grain': 9.,
            'Nmin_straw': 4.,
            'Non_agricultural_products': 0.,
            'N_recovery_factor': 0.4,
            'Off_farm_agric_labour': 0.10 * 1500.,  # H_mandays
            'Off_farm_non_agric_labour': 0.15 * 1500.,  # H_mandays
            'other_expense': 3000.,
            'Pension': 0.,
            'potential_transpiration': 0.,  # justfilledinsomething
            'precipitation': 0.,  # justfilledinsomehtingNOTUSED?
            'Price_for_chemicals': 0.,
            'Price_of_crop': 67. * 3.5,  # dennis: what is the unit? INR/kg/ha?
            'Price_of_fertiliser': (276. / 50.) / 4.,
            'Price_of_livestock': 25000.,
            'Price_of_seeds': 3500,  # 700 per bag, 5 bags per ha
            'Price_per_manday': 0.,  # This is wage, set in opening of household model
            'Remitance': 3000.,
            'runoff_coefficient': 0.,  # was0.6
            'savings': 0.,
            'capital_interest_rate': -0.03,
            'saving_interest_rate': 0.,  # justfilledinsomething
            'school_fees': 0.,
            'Su_max': 0.,
            'tax_rate': 0.2,
            'tax_rate_o': 0.,
            'the_weight_gain_rate': 0.13,
            'veterinary_costs': 5000.,
            'annual_rainfall_P': np.ones(12),  # Ichangedthis
            'annual_transpiration': 0.,  # justfilledinsomething
            'biological_fixation': 0.,
            'commercial_fertilizer': 0.,
            'energy_intercepted_i': 0.6,
            'crop_management_factor_C': 2.3,
            'enrichment_ratio': 2.5,
            'feed_requirement': 9.125,  # Feed requirement = 25kg/ha/day for a milchanimal
            'gaseous_losses': 10.,
            'grassland_area': 0.5,
            'grassland_runoff_coefficient': 0.06,
            'Kygrass': 0.9,
            'harvest_indext': 0.4,
            'households_with_livestock': 1.,
            'leaching': 0.,
            'length_L': 100.,
            'livestock_manure_factor': 75. * 0.05,
            'manure_N_factor': 4.6,
            'max_grass_yield': 25.,
            'N_concentration_in_grain': 0.88 / 1000.,
            'N_concentration_in_straw': 0.,
            'rainfall_intensity_I': 50.,
            'rain_fixation': 8.,
            'slope_s': 15.,
            'soil_density': 1.3,
            'soil_erodibility_factor_F': 5.,
            'top_soil_depth': 0.1,  # vanaf hier zitten ze niet in die par lijst
            'p_g': 0.,  # filledinsomething
            'p_c': 0.,  # filledinsomething
            'p_q': 0.,  # filledinsomething
            'Ta_Tm': 0.,  # dezewordtuitgerekend sum of crop (evapotranspiration/frac) / sum of crop water demand
            'ETa_ETm': 0.,  # dezewordtuitgerekend
            'Farm_Sales': 0.,  # dezewordtuitgerekend
            'livestock_sold': 0.,
            'Crop_yield': 0.,  # dezewordtuitgerekend#filledinsomething
            'manures': 0.,  # kanhetnietvindeninhouseholdmodel#filledinsomething
            'livestock_costs': 0.,  # dezewordtuitgerekend#filledinsomething
            'crop_cost': 0.,  # dezewordtuitgerekend#nulalsrandombeginwaardeingevuld
            'Interest_on_loan': 0.,  # kanhetnietvindeninhouseholdmodel#filledinsomething
            'tax': 0.,  # dezewordtuitgerekend#filledinsomething
            'Crop_sales': 0.,  # kanhetnietvindeninhouseholdmodel#filledinsomething
            'tax': 0.,  # dezewordtuitgerekend#filledinsomething
            'MV_lab': 0.,  # kanhetnietvindeninhouseholdmodel
            'feed_residues_rate': 0.3,
            'expmat': np.empty(8),  # Ifilledthisin
            'incmat': np.empty(7),  # IfilledthisinIdon'tknowifitwillworklikethis
            'fertilisation': 0.,
            'deficit': 0.,
            'household_activities': 0.,
            'soil_depth': 300.  # mm
            }

Adjusted = {'crop_area': 2.,
            'birth_rate': 0.15,
            'Chemicals': 0.,
            'crop_labour': 0.68 * 1500.,  # H_mandays
            'crop_for_home_consumption': 0.,
            'death_rate': 0.15,
            'education': 0.,
            'Family_size': 6.,
            'Farming_experience_gained': 10.,
            'feed_maintaining_requirement': 0.,
            'fert_N_factor': 0.40,
            'Food_bought': 0.,
            'Gifts': 0.,
            'investment': 10000.,
            'knowledge_obsolete_rate': 0.2,
            'Ky': 0.85,
            'labour_domestic': 0.01 * 1500.,  # H_mandays
            'livestock_labour': 0.04 * 1500.,  # H_mandays
            'Labour_factor': 0.4,
            'labour_firewood_collection': 0.01 * 1500.,  # H_mandays
            'labour_water_collection': 0.01 * 1500.,  # H_mandays
            'Labour_balance': 0.,
            'livestock_retained': 0.,
            'liv_Quantity_purchases': 0.25,
            'loan_interest_rate': 0.35,
            'loan_size': 0.,
            'mandays': 250.,
            'max_crop_labour': 260.,
            'max_Crop_yield': 1000. * 0.35,
            'max_N_app': 50.,
            'min_Crop_yield': 0 * 0.35,
            'Nmin_grain': 9.,
            'Nmin_straw': 4.,
            'Non_agricultural_products': 0.,
            'N_recovery_factor': 0.4,
            'Off_farm_agric_labour': 0.10 * 1500.,  # H_mandays
            'Off_farm_non_agric_labour': 0.15 * 1500.,  # H_mandays
            'other_expense': 3000.,
            'Pension': 0.,
            'potential_transpiration': 0.,  # justfilledinsomething
            'precipitation': 0.,  # justfilledinsomehtingNOTUSED?
            'Price_for_chemicals': 0.,
            'Price_of_crop': 67. * 3.5,
            'Price_of_fertiliser': (276. / 50.) / 4.,
            'Price_of_livestock': 25000.,
            'Price_per_manday': 0.,  # isthiscorrect?itwaswage
            'Remitance': 3000.,
            'runoff_coefficient': 0.,  # was0.6
            'savings': 0.,
            'capital_interest_rate': 0. - 0.03,
            'saving_interest_rate': 0.,  # justfilledinsomething
            'school_fees': 0.,
            'Su_max': 0.,
            'tax_rate': 0.2,
            'tax_rate_o': 0.,
            'the_weight_gain_rate': 0.13,
            'veterinary_costs': 5000.,
            'annual_rainfall_P': np.ones(12),  # Ichangedthis
            'annual_transpiration': 0.,  # justfilledinsomething
            'biological_fixation': 0.,
            'commercial_fertilizer': 0.,
            'energy_intercepted_i': 0.6,
            'crop_management_factor_C': 2.3,
            'enrichment_ratio': 2.5,
            'feed_requirement': 12.,
            'gaseous_losses': 10.,
            'grassland_area': 0.5,
            'grassland_runoff_coefficient': 0.06,
            'Kygrass': 0.9,
            'harvest_indext': 0.4,
            'households_with_livestock': 1.,
            'leaching': 0.,
            'length_L': 100.,
            'livestock_manure_factor': 75. * 0.05,
            'manure_N_factor': 4.6,
            'max_grass_yield': 15.,
            'N_concentration_in_grain': 0.88 / 1000.,
            'N_concentration_in_straw': 0.,
            'rainfall_intensity_I': 50.,
            'rain_fixation': 8.,
            'slope_s': 15.,
            'soil_density': 1.3,
            'soil_erodibility_factor_F': 5.,
            'top_soil_depth': 0.1,  # vanaf hier zitten ze niet in die par lijst
            'p_g': 0.,  # filledinsomething
            'p_c': 0.,  # filledinsomething
            'p_q': 0.,  # filledinsomething
            'Ta_Tm': 0.,  # dezewordtuitgerekend
            'ETa_ETm': 0.,  # dezewordtuitgerekend
            'Farm_Sales': 0.,  # dezewordtuitgerekend
            'livestock_sold': 0.,
            'Crop_yield': 0.,  # dezewordtuitgerekend#filledinsomething
            'manures': 0.,  # kanhetnietvindeninhouseholdmodel#filledinsomething
            'livestock_costs': 0.,  # dezewordtuitgerekend#filledinsomething
            'crop_cost': 0.,  # dezewordtuitgerekend#nulalsrandombeginwaardeingevuld
            'Interest_on_loan': 0.,  # kanhetnietvindeninhouseholdmodel#filledinsomething
            'tax': 0.,  # dezewordtuitgerekend#filledinsomething
            'Crop_sales': 0.,  # kanhetnietvindeninhouseholdmodel#filledinsomething
            'tax': 0.,  # dezewordtuitgerekend#filledinsomething
            'MV_lab': 0.,  # kanhetnietvindeninhouseholdmodel
            'feed_residues_rate': 0.3,
            'expmat': np.empty(8),  # Ifilledthisin
            'incmat': np.empty(7),  # IfilledthisinIdon'tknowifitwillworklikethis
            'fertilisation': 0.,
            'deficit': 0.,
            'household_activities': 0.}
