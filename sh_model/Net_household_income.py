import numpy as np


def Net_household_income(Constants, VarMat, Yield, VarMatSoil, Par=None):
    'this function describes the net household income'  # This is actually gross income, not net.

    if Par is not None:
        # f_shape_fert=Par[2]    #shape of the convex, >0 for convex curves
        f_shape_lab = Par[2]  # shape of the convex, >0 for convex curves

    else:
        # f_shape_fert=0.3        #shape of the convex, >0 for concave curves
        f_shape_lab = -9.65  # shape of the convex, >0 for convex curves

    # f_shape_fert=0.3        #shape of the convex, >0 for concave curves
    # f_shape_lab=-10

    Livestockt = VarMat

    # Interest on saving
    interest_on_savings = Constants['saving_interest_rate'] * Constants['savings']

    # Private
    Private = Constants['Gifts'] + Constants['Pension'] + Constants['Remitance']

    # b
    b = Constants['min_Crop_yield'] / Constants['max_Crop_yield']

    # Fmax
    Fmax = Constants[
        'max_N_app']  # /N_recovery_factor ; %was *15 N_recovery factor =1, we assume 0 fertilizer leads to Ymin, optimal fertilizer is 250 kg/ha

    # total quantity of manure
    manures = Livestockt * Constants['livestock_manure_factor']  # in Kg N per year

    # converting kg application of commercial fertilizer to N kg by
    commercial_fertilizer_app = Constants['commercial_fertilizer'] * Constants['fert_N_factor']

    # Fapp\Fmax
    Fapp_Fmax = (commercial_fertilizer_app + manures) / (Constants['crop_area'] * Fmax)

    # Fertilizer factor
    # linear
    fertilizer_factor = b + np.minimum(Fapp_Fmax, 1.) * (1. - b)
    # convex
    # fertilizer_factor_rel = b + np.minimum(Fapp_Fmax,1.)*(1.-b) 
    # fertilizer_factor=(np.exp(fertilizer_factor_rel*f_shape_fert)-1)/(np.exp(f_shape_fert)-1)

    # print(commercial_fertilizer_app,manures,VarMatSoil)
    # fertilizer_factor=1

    # sukratis code
    # Ns = VarMatSoil
    # Nf = Constants['manures'] + commercial_fertilizer_app
    # Nt = Ns + Nf
    # water_limited_nutrient_uptake_efficency=1
    # fertilizer_factor = (water_limited_nutrient_uptake_efficency* Nt)/Yield

    # #crop water factor
    # Dennis: i think can be taken out since it is calculated in wb yield already
    # crop_water_factor = np.maximum(0.,np.minimum(1.,1.-Constants['Ky']*(1.-Constants['Ta_Tm'])))    

    # updating certain parameters

    ######## Labour dispersion if earnings are low #######
    # old:
    # Constants['MV_lab'] = Constants['Price_of_crop'] * fertilizer_factor * Constants['max_Crop_yield'] * crop_water_factor * Constants['crop_area'] / Constants['max_crop_labour']
    Constants['MV_lab'] = Constants['Price_of_crop'] * fertilizer_factor * Constants['max_Crop_yield'] * Constants[
        'crop_area'] / Constants['max_crop_labour']

    if Constants['Price_per_manday'] >= Constants['MV_lab']:  # half of labour goes to off farm labour or non
        Constants['crop_labour'] = 0
        Constants['Off_farm_agric_labour'] = Constants['Off_farm_agric_labour'] + Constants['crop_labour'] / 2
        Constants['Off_farm_non_agric_labour'] = Constants['Off_farm_non_agric_labour'] + Constants['crop_labour'] / 2

    if Constants['Price_per_manday'] < Constants['MV_lab']:  # all labour returns to farm
        Constants['crop_labour'] = Constants['crop_labour'] + Constants['Off_farm_agric_labour'] + Constants[
            'Off_farm_non_agric_labour']
        Constants['Off_farm_agric_labour'] = 0
        Constants['Off_farm_non_agric_labour'] = 0

        # linear labour factor
    # labour_factor_FI = np.maximum(0.,np.minimum(Constants['crop_labour']/Constants['max_crop_labour'],1))

    # convex shape labour factor
    labour_factor_FI_rel = np.maximum(0., np.minimum(Constants['crop_labour'] / Constants['max_crop_labour'], 1))
    labour_factor_FI = (np.exp(labour_factor_FI_rel * f_shape_lab) - 1) / (np.exp(f_shape_lab) - 1)

    # labour_factor_FI = 1.

    # Crop yield
    # #Dennis: i think the crop water factor can be taken out since it is calculated in wb yield already
    # Constants['Crop_yield'] = fertilizer_factor*labour_factor_FI*Constants['max_Crop_yield']*crop_water_factor 

    # using yield from wb yield, having taken into account water stress
    # print("Fertilizer factor: {}".format(fertilizer_factor))
    # print("Labour factor: {}".format(labour_factor_FI))
    Constants['Crop_yield'] = fertilizer_factor * labour_factor_FI * Yield

    Constants['crop_for_home_consumption'] = np.minimum((Constants['Crop_yield'] * Constants['crop_area']),
                                                        Constants['crop_for_home_consumption'])
    Quantity_of_crop_sold = np.maximum(
        (Constants['Crop_yield'] * Constants['crop_area']) - Constants['crop_for_home_consumption'], 0.)

    Constants['Crop_sales'] = Constants['Price_of_crop'] * Quantity_of_crop_sold

    # Livestock_sales
    Livestock_sales = Constants['Price_of_livestock'] * Constants['livestock_sold']

    Constants['Farm_Sales'] = Constants['Crop_sales'] + Livestock_sales

    # Labour sales: Assuming outside price per manday is some factor < 1 of price per manday, i.e.
    fac = 1
    Labour_sales = (Constants['Off_farm_agric_labour'] + Constants['Off_farm_non_agric_labour']) * fac * Constants[
        'Price_per_manday']
    # Income generation activities
    Income_generation_activities = Constants['Farm_Sales'] + Labour_sales + Constants['Non_agricultural_products']
    # print(Constants['Farm_Sales'],Labour_sales)
    # income
    N = Income_generation_activities + interest_on_savings + Private
    # print(N)
    Constants['incmat'] = np.array(
        [Constants['Crop_sales'], Livestock_sales, Labour_sales, Constants['Non_agricultural_products'],
         interest_on_savings, Private, Constants['Crop_yield']])

    return (Constants, N)
