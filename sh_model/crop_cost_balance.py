# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:38:56 2015

@author: Nadezhda
"""
# delete these later!!

from crop_cost_compute import *


# from globalparameters import *

def crop_cost_balance(Parameters, Constants, target):
    'this function updates the parameters that define the crop_cost based on target. Assuming pars is of form *.*'

    # insert globalparameters ;

    # pars_upd = pars ;

    Parameters, Constants = crop_cost_compute(Constants, Parameters)  # runt deze dan?? JA! checked

    #    print Constants['crop_cost']
    #    print target

    target = Constants['crop_cost'] - target

    # print 'CCtarget', target

    if target > 0.:
        if target < Parameters['Chemicals'] * Constants['Price_for_chemicals']:
            # print('CChere1')
            Parameters['Chemicals'] = target / (Constants['crop_area'] * Constants['Price_for_chemicals'])
        elif target >= Parameters['Chemicals'] * Constants['Price_for_chemicals'] and target <= Constants[
            'cropinput_cost']:
            # print('CChere2')
            Parameters['Chemicals'] = 0.
            # Dennis: look into why and what causes price of fert to be 0
            if Constants['Price_of_fertiliser'] != 0:
                Parameters['commercial_fertilizer'] = (Constants['cropinput_cost'] - target) / Constants[
                    'Price_of_fertiliser']
        elif (np.maximum(0., -Constants['Labour_balance']) * Constants['Price_per_manday']) > 0.:
            # print('CChere3')
            if target > Constants['cropinput_cost']:  # %andand target - Constants['cropinput_cost'] < Labour_balance
                Parameters['Chemicals'] = 0.
                Parameters['commercial_fertilizer'] = 0.
                # print('CChere4')
                if target - Constants['cropinput_cost'] >= Parameters['labour_domestic'] * Constants[
                    'Price_per_manday'] and target - Constants['cropinput_cost'] < (
                        Parameters['labour_domestic'] + Constants['labour_firewood_collection']) * Constants[
                    'Price_per_manday']:
                    Parameters['labour_firewood_collection'] = (target - Constants['cropinput_cost'] - Constants[
                        'labour_domestic'] * Constants['Price_per_manday']) / Constants['Price_per_manday']
                    Parameters['labour_domestic'] = 0.
                    # print('CChere5')
                elif target - Constants['cropinput_cost'] >= (
                        Parameters['labour_domestic'] + Parameters['labour_firewood_collection']) * Constants[
                    'Price_per_manday'] and target - Constants['cropinput_cost'] < Constants['household_activities'] * \
                        Constants['Price_per_manday']:  # Constants['household_activities']
                    Parameters['labour_domestic'] = 0.
                    Parameters['labour_firewood_collection'] = 0.
                    Parameters['labour_water_collection'] = (target - Constants['cropinput_cost'] - (
                                Parameters['labour_domestic'] + Parameters['labour_firewood_collection']) * Constants[
                                                                 'Price_per_manday']) / Constants['Price_per_manday']
                    # print('CChere6')
                elif target - Constants['cropinput_cost'] >= Constants['household_activities'] * Constants[
                    'Price_per_manday'] and target - Constants['cropinput_cost'] < (
                        Constants['household_activities'] + Parameters['livestock_labour']) * Constants[
                    'Price_per_manday']:
                    Parameters['labour_domestic'] = 0.
                    Parameters['labour_firewood_collection'] = 0.
                    Parameters['labour_water_collection'] = 0.
                    Parameters['livestock_labour'] = (target - Constants['cropinput_cost'] - Constants[
                        'household_activities'] * Constants['Price_per_manday']) / Constants['Price_per_manday']
                    # print('CChere7')
                elif target - Constants['cropinput_cost'] >= (
                        Constants['household_activities'] + Constants['livestock_labour']) * Constants[
                    'Price_per_manday'] and target - Constants['cropinput_cost'] < Constants['labour_requirements'] * \
                        Constants['Price_per_manday']:
                    Parameters['labour_domestic'] = 0.
                    Parameters['labour_firewood_collection'] = 0.
                    Parameters['labour_water_collection'] = 0.
                    Parameters['livestock_labour'] = 0.
                    Parameters['crop_labour'] = (target - Constants['cropinput_cost'] - Constants[
                        'household_activities'] * Constants['Price_per_manday']) / (Constants['Price_per_manday'])
                    # print('CChere8')
                else:
                    Parameters['labour_domestic'] = 0.
                    Parameters['labour_firewood_collection'] = 0.
                    Parameters['labour_water_collection'] = 0.
                    Parameters['livestock_labour'] = 0.
                    Parameters['crop_labour'] = 0.
                    # print('CChere9')
    return (Parameters, Constants)
