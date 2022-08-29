=======================
How to run Simulator.py
=======================

Has 4 options to run the model:
- Normal run		: run the SH model with the defined parameters, leave all bool values to 0
- Sensitivity analysis	: run the model with random parameter values (MCS), set Sens_bool=1
- Bootstrap analysis	: run the model with bootstrap climate data, set Bootstrap_bool=1
- Range analysis	: run the model with the optimal parameter sets from the MCS, Range_bool=1

Determine starting year using		: start_year
Determine simulation time (year) using	: Tsimul

Normal run
Uses parameters set defined in WB_Yield
Runs the model twice, once with existing irrigation and another with new ponds
Output:
- Various daily state and flux variables
- Yield
- Benefit

Sensitivity analysis
Run the model for many iterations with random parameter sets within defined range in: ParMin, ParMax
Define the desired number of iterations in: nmax
Depending on what parameter is tested, define it accordingly at the start of WB_Yield.py
Output:
- Model scores r-squared, NS, NS log, MAE and their associated parameter sets
- Yield

Bootstrap analysis
Run the model for many iterations with bootstrap climate data to check model stability or effects of different climate to yield
Define the desired number of iterations in: nbool
Output:
- Bootstrap climate data (precipitation, ET, irrigation)
- Yield
- Benefit

Range analysis
Run the model with the most optimal parameter sets
Define the thresholds for each model scores at the start of the range analysis in: OptPar_Range
Output:
- Range of yield predictions
- Range of benefit predictions

=======================
How to run KPCA.py
=======================
Documentations:
- https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
- Schölkopf, B., Smola, A., & Müller, K.-R. (1998). Nonlinear Component Analysis as a Kernel Eigenvalue Problem. Neural Computation, 10(5).
- KPCA_Results word doc for the cross validation steps and results

Define the desired kernel at the start and their hyperparameters, see scikit-learn doc for details
The first half of the code is used to obtain the regression model in kernel space to predict total error
The second half is to find the adjusted predicted yield i.e. SH model + structural error model
The benefit of the irrigation ponds are calculated at the end of this code as well