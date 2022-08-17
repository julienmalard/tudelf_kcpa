# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:14:06 2015

@author: Nadezhda
"""
# deze mogen weg als


# from globalparameters import *


def livestock_costs_compute(Parameters, Constants):
    'this function computes the livestock costs for given pars value'

    Constants['livestock_costs'] = (Parameters['liv_Quantity_purchases'] * Constants['Price_of_livestock']) + \
                                   Parameters['veterinary_costs']

    return Constants
