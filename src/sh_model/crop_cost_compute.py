# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:26:40 2015

@author: Nadezhda
"""
# remove these in the end
import numpy as np


# from globalparameters import *


def crop_cost_compute(Constants, Parameters, crop_area=2):
    'this subroutine computes the crop cost for the given set of parameters. Rest are assumed to be global parameters. It also computes costs of agricultural inputs'

    # pars
    labour_available = Constants['Family_size'] * Constants['Labour_factor'] * Constants['mandays'] - Constants[
        'Off_farm_agric_labour'] - Constants['Off_farm_non_agric_labour']

    #
    # should various labor activities be proportional to labor available TODO
    # Household activities
    Constants['household_activities'] = Parameters['labour_domestic'] + Parameters['labour_firewood_collection'] + \
                                        Parameters['labour_water_collection']

    # Labour requirements
    # labour for crop production is based on whether the wage rate is larger or smaller than the marginal value of crop prooduction.

    if Constants['Price_per_manday'] >= Constants['MV_lab']:
        Parameters['crop_labour'] = 0.

    Constants['labour_requirements'] = Parameters['crop_labour'] + Parameters['livestock_labour'] + Constants[
        'household_activities']

    # Labour balance
    Constants['Labour_balance'] = labour_available - Constants['labour_requirements']

    # crop cost
    Constants['crop_cost'] = crop_area * (Constants['Price_for_chemicals'] + Constants['Price_of_seeds']) + Parameters[
        'commercial_fertilizer'] * Constants['Price_of_fertiliser'] + (
                                         np.maximum(0., -Constants['Labour_balance']) * Constants['Price_per_manday'])

    #
    Constants['cropinput_cost'] = crop_area * (Constants['Price_for_chemicals'] + Constants['Price_of_seeds']) + \
                                  Parameters['commercial_fertilizer'] * Constants['Price_of_fertiliser']

    return (Parameters, Constants)
