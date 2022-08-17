# Expenditure is the sum of expenses that a smallholder faces during a year and reduces her capital.

##import other functions
from crop_cost_compute import *
from livestock_costs_compute import *


# from globalparameters import *

def Expenditure(Constants, Parameters):
    'this is the expenditure function blablabla'

    # insert Global_parameters

    # interest on loan:
    Constants['Interest_on_loan'] = Parameters['loan_interest_rate'] * Constants[
        'loan_size']  # update de library now before constants go into crop_compute?

    # calls crop cost compute function
    Parameters, Constants = crop_cost_compute(Constants, Parameters)

    # calls livestock compute function
    Constants = livestock_costs_compute(Parameters, Constants)

    # Purchase of non-food items
    Purchase_of_non_food_items = Constants['crop_cost'] + Parameters['investment'] + Parameters['school_fees'] + \
                                 Constants['livestock_costs']

    Constants['tax'] = Constants['Farm_Sales'] * Parameters['tax_rate']

    # food bought
    Constants['Food_bought'] = Parameters['Food_bought']
    #
    #    #other expenses
    Constants['other_expense'] = Parameters['other_expense']

    # total
    E = Constants['Food_bought'] + Constants['Interest_on_loan'] + Constants['tax'] + Constants[
        'other_expense'] + Purchase_of_non_food_items

    # want we hebben toch alleen maar Constants['crop_cost'][0] nodig voor de berekening??

    expmat = np.array(
        [Constants['Food_bought'], Constants['Interest_on_loan'], Constants['tax'], Constants['other_expense'],
         Constants['crop_cost'], Parameters['investment'], Parameters['school_fees'], Constants['livestock_costs']])

    return (Parameters, Constants, E)

##Notes
# Where are the maintenance costs farm??
