# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:09:15 2015

@author: Nadezhda
"""

from livestock_costs_compute import *


# from globalparameters import *

def livestock_costs_balance(Parameters, Constants, target):
    'this function updates the internal parameters of the livestock costs if its total costs are foreced to be lowered.'

    # insert globalparameters

    # Constants['liv_Quantity_purchases']
    # pars.Constants['veterinary_costs']

    # what is the target, assuming target that is input is the figure to which
    # the cuts need to be made.

    Constants = livestock_costs_compute(Parameters, Constants)

    target = Constants['livestock_costs'] - target

    if target == (Parameters['liv_Quantity_purchases'] * Constants['Price_of_livestock']) + Parameters[
        'veterinary_costs']:
        # print 'LShere1'
        Parameters['liv_Quantity_purchases'] = 0.
        Parameters['veterinary_costs'] = 0.
    elif target >= Parameters['veterinary_costs'] and target <= Parameters['liv_Quantity_purchases'] * Constants[
        'Price_of_livestock'] + Parameters['veterinary_costs']:
        target = target - Parameters['veterinary_costs']
        Parameters['veterinary_costs'] = 0.
        Parameters['liv_Quantity_purchases'] = (Parameters['liv_Quantity_purchases'] * Constants[
            'Price_of_livestock'] - target) / Constants['Price_of_livestock']
    elif target < Parameters['veterinary_costs']:
        Parameters['veterinary_costs'] = Parameters['veterinary_costs'] - target
    return (Parameters, Constants)
