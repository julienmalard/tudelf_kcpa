import os

import pandas as pd

from KPCA import KPCAModel, remove_outliers, save_model_comparison_table

DATA_DIR = "../input_data"
OUT_DIR = "../out"
PLOT_DIR = os.path.join(OUT_DIR, 'plots')
TABLE_DIR = os.path.join(OUT_DIR, 'tables')

if __name__ == "__main__":
    dat = pd.read_excel(f"{DATA_DIR}/KPCA/KPCA_dat_adj_withCosts.xlsx", header=0)
    dat_irr = pd.read_excel(f"{DATA_DIR}/KPCA/KPCA_dat_adj_irr.xlsx", header=0)

    default_x_vars = [
        "AreaCot", "ChildrenHelp", "Evap", "FertAmount", "Irr",
        "Lat", "Long",
        "Prec", "SeedsCost", "SoilDepth"
    ]

    hybrid_model_x_vars = default_x_vars + ["YieldModel"]
    hybrid_model = KPCAModel(hybrid_model_x_vars, var_y="YieldDiff", data=remove_outliers(dat))
    hybrid_model.train_and_plot(f"{PLOT_DIR}/hybridModel")
    hybrid_model.save_tables(os.path.join(TABLE_DIR, "hybrid model"))

    no_sdm_model = KPCAModel(default_x_vars, var_y="YieldObs", data=remove_outliers(dat))
    no_sdm_model.train_and_plot(f"{PLOT_DIR}/noSDM")

    save_model_comparison_table(
        {
            "SH model": {
                "mae_test": 489,
                "ns_test": -0.624,
                "nslog_test": -1.145,
                "r2_test": 0.005
            },
            "SH + structural error model": hybrid_model,
            "KPCA alone": no_sdm_model
        },
        file=os.path.join(TABLE_DIR, "model comparison.txt")
    )

    # %% evaluating benefit
    survey_dat = pd.read_csv(f"{DATA_DIR}/Baseline/Final_Analysis_345_nrh_TijmenData_v4.csv", header=0)

    Irr_exist = survey_dat["water/area_irrig"] > 0
    CottonArea = survey_dat['financial_information/area_cotton']

    Yield_adj = remove_outliers(dat)["YieldModel"] + modelyield_model.y_pred_all
    Yield_adj_irr = remove_outliers(dat_irr)["YieldModel"] + modelyield_model_irr.y_pred_all
    Benefit = (Yield_adj_irr - Yield_adj) * 67. * 3.5

    Ben_extIrr = round(Benefit[Irr_exist].mean())
    Ben_noIrr = round(Benefit[~Irr_exist].mean())
    IrrBenefit = np.array((Ben_extIrr, Ben_noIrr))

    colors_ben = np.array(['g'] * len(IrrBenefit))
    colors_ben[IrrBenefit < 0] = 'r'
    x = np.array(['had irrigation', 'no irrigation'])
    plt.figure(figsize=(3, 7))
    bars2 = plt.bar(x, IrrBenefit, color=colors_ben)
    plt.ylabel("Mean Benefit per farmer [INR/y/ha]", fontsize=16)
    plt.title("Benefit", fontsize=16)
    plt.text(bars2[0].get_x() + 0.2, bars2[0].get_height() + 50, bars2[0].get_height())
    plt.text(bars2[1].get_x() + 0.2, bars2[1].get_height() - 300, bars2[1].get_height())

    plt.savefig(f'{PLOT_DIR}/BenefitBarIrr.png', dpi=300, bbox_inches="tight")
    plt.close()

    # %% bar plot
    Village = survey_dat["General/village_name"]
    Villages = Village.unique()
    IrrBenefit_vil = np.zeros(len(Villages))
    for i in range(len(Villages)):
        tmp = Village == Villages[i]
        IrrBenefit_vil[i] = np.round(Benefit[tmp].mean())

    colors = np.array(['g'] * len(Villages))
    colors[IrrBenefit_vil < 0] = 'r'
    plt.figure(figsize=(10, 20))
    bars = plt.barh(Villages, IrrBenefit_vil, color=colors)
    plt.ylabel("Mean Benefit per farmer [Rs/y]")
    plt.title("Benefit per village")

    plt.savefig(f'{PLOT_DIR}/BenefitBarVillage.png', dpi=300, bbox_inches="tight")
    plt.close()

    # %% hist of yield increase
    a = Yield_adj_irr - Yield_adj

    bins = np.linspace(-200, 200, 100)
    labels = ['Farmers with existing irrigation', 'Farmers without prior irrigation']

    plt.figure(figsize=(12, 7))
    plt.hist(a, bins, rwidth=0.8, label=labels, density=True)
    plt.title('Histogram of yield increase', fontsize=18)
    plt.xlabel('Yield increase [kg/ha]', fontsize=18)
    plt.ylabel('Probability', fontsize=18)
    plt.legend()

    plt.savefig(f'{PLOT_DIR}/histYieldIncrease.png', dpi=300, bbox_inches="tight")
    plt.close()


    # %% hist of yield
    def plot_hist_yield(a, b, title, file):
        bins = np.linspace(0, 5000, 50)
        labels = ['Old yield', 'Yield with new pond']

        plt.figure(figsize=(12, 7))
        plt.hist([a, b], bins, rwidth=0.8, label=labels, density=True)
        plt.title(title, fontsize=18)

        plt.xlabel('Yield [kg/ha]', fontsize=18)
        plt.ylabel('Probability', fontsize=18)
        plt.legend()

        plt.savefig(file, dpi=300, bbox_inches="tight")
        plt.close()


    plot_hist_yield(Yield_adj, Yield_adj_irr, 'Histogram of yield', f'{PLOT_DIR}/histYield.png')
    plot_hist_yield(Yield_adj[Irr_exist], Yield_adj_irr[Irr_exist],
                    'Histogram of yield (farmers with existing irrigation)', f'{PLOT_DIR}/histYield_existingIrr.png')

    # %%
    Xplot_irr = Yield_adj_irr
    Xplot_noIrr = Yield_adj
    Yplot = remove_outliers(dat_irr)['YieldObs']

    Xplot2 = sm.add_constant(Xplot_irr)
    estPlot = sm.RLM(Yplot, Xplot2, M=sm.robust.norms.HuberT())
    estPlot2 = estPlot.fit()

    plt.figure(figsize=(10, 8))
    plt.fill_between(np.unique(Xplot_irr), np.unique(estPlot2.predict(sm.add_constant(Xplot_irr))) + 450,
                     np.unique(estPlot2.predict(sm.add_constant(Xplot_irr))) - 450, color='yellow', alpha=0.5)
    plt.scatter(Xplot_irr, remove_outliers(dat_irr)['YieldObs'], c="red", s=20, edgecolor='k',
                label='Predicted yield with new ponds')
    plt.scatter(Xplot_noIrr, remove_outliers(dat)['YieldObs'], c="blue", s=20, edgecolor='k', label='Predicted yield')

    Xplot2 = sm.add_constant(Xplot_irr)
    estPlot = sm.RLM(Yplot, Xplot2, M=sm.robust.norms.HuberT())
    estPlot2 = estPlot.fit()
    betaPlot = estPlot2.params
    rsquared_adj = sm.OLS(Yplot, Xplot2).fit().rsquared_adj
    adj_r2Plot = round(rsquared_adj, 3)
    plt.text(300, 4000, '$r^2$ = {}'.format(adj_r2Plot), ha='left', va='center', fontsize=18)
    plt.text(300, 3600, 'µ = {}'.format(round(estPlot2.params[estPlot2.params.index[1]], 3)), ha='left', va='center',
             fontsize=18)
    # plt.errorbar(Xplot, Yplot,  0, 450, fmt='r^',label='Annual yield per farmer', elinewidth =0.5,
    #               marker='x', markersize='5',markeredgecolor='blue', ecolor=['red'],barsabove=False)
    plt.plot(np.arange(0, 5000), np.arange(0, 5000), color="black")
    plt.plot(np.unique(Xplot_irr), np.unique(estPlot2.predict(sm.add_constant(Xplot_irr))), color="black",
             linestyle='--')
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)
    plt.title("Predicted Yield vs. Observed Yield", fontsize=18)
    plt.xlabel("Predicted Yield [kg/ha]", fontsize=18)
    plt.ylabel("Observed Yield [kg/ha]", fontsize=18)
    plt.legend()
    plt.savefig(f'{PLOT_DIR}/PredictedVsObservedYield.png', dpi=300, bbox_inches="tight")
    plt.close()

    # %% linear regressions of pc results from spss
    SPSS_dat = pd.read_excel(f"{DATA_DIR}/KPCA/PCs_TotalError.xlsx", header=0)

    plotfit = np.arange(-2.2, 3.5, 0.5)

    namePC = 'PC3'
    Xplot = np.array(SPSS_dat[namePC])
    Yplot = np.array(SPSS_dat.iloc[:, -1])

    plt.figure(figsize=(10, 8))
    plt.scatter(Xplot, Yplot, c="red", s=20, edgecolor='k')

    fit = np.poly1d(np.polyfit(Xplot, Yplot, 1))
    model = HuberRegressor().fit(Xplot.reshape(-1, 1), Yplot.reshape(-1, 1))
    adj_r2Plot = round(model.score(Xplot.reshape(-1, 1), Yplot.reshape(-1, 1)), 3)

    plt.text(min(Xplot) + 6, max(Yplot), '$r^2$ = {}'.format(adj_r2Plot), ha='left', va='center', fontsize=18)
    # plt.plot(np.unique(Xplot), np.unique(estPlot2.predict(sm.add_constant(Xplot))), color="black",linestyle='--')
    plt.plot(plotfit, fit(plotfit), color="black", linestyle='--', linewidth=2)
    plt.xlim(-3.5, 6)
    # plt.ylim(0,5000)
    plt.grid(color='k', linestyle='-', linewidth=0.1)
    plt.title("Total Error by Perceived Fertilizer Cost, Pesticide Cost, and Crop Price ({})".format(namePC),
              fontsize=18)
    plt.xlabel("Perceived Fertilizer Cost, Pesticide Cost, and Crop Price ({})".format(namePC), fontsize=18)
    plt.ylabel("Total Error [kg/ha]", fontsize=18)

    plt.savefig(f'{PLOT_DIR}/Regression{namePC}.png', dpi=300, bbox_inches="tight")
    plt.close()
