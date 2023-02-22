# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:35:17 2021

@author: denni
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({'font.size': 12})

DATA_DIR = "../input_data"
PLOT_DIR = "../plots"


def mae(pred, obs):
    return np.sum(abs(pred - obs)) / len(pred)


def ns(pred, obs):
    return 1 - (np.sum((pred - obs) ** 2) / np.sum((obs.mean() - obs) ** 2))


def ns_log(pred, obs):
    return 1 - (np.sum((np.log10(pred) - np.log10(obs)) ** 2) / np.sum((np.log10(obs.mean()) - np.log10(obs)) ** 2))


class KPCAModel(object):
    def __init__(self, vars_x, var_y, data, kernels=None):
        self.vars_X = vars_x
        self.var_Y = var_y
        self.data = data
        self.kernels = kernels or ["rbf", "sigmoid", "poly"]

        self.best = {}
        self.y_pred_all = None

    def train_and_plot(self, folder=PLOT_DIR):
        if not os.path.isdir(folder):
            os.makedirs(folder)
        vars_x_with_model_yield = self.vars_X if "YieldModel" in self.vars_X else [*self.vars_X, "YieldModel"]
        x = StandardScaler().fit_transform(self.data[vars_x_with_model_yield])
        y = self.data[self.var_Y]

        # split training and test data
        x_train_initial, x_test_initial, y_train_initial, y_test_initial = train_test_split(x, y, test_size=0.25, random_state=1)
        yield_model_index = vars_x_with_model_yield.index("YieldModel")
        x_test_model_yield = x_test_initial[:, yield_model_index]
        if "YieldModel" not in self.vars_X:
            x_test_initial = np.delete(x_test_initial, yield_model_index, axis=1)
            x_train_initial = np.delete(x_train_initial, yield_model_index, axis=1)
            x = np.delete(x, yield_model_index, axis=1)


        # Running KPCA
        for kernel in self.kernels:
            degs = range(1, 6) if kernel == "poly" else [None]
            for deg in degs:
                print("kernel", kernel, "deg", deg)


                kpca = KernelPCA(kernel=kernel, fit_inverse_transform=True, n_components=None, degree=deg)
                x_train = kpca.fit_transform(x_train_initial)
                x_test = kpca.transform(x_test_initial)

                d = kpca.lambdas_
                d_cumsum = np.cumsum(d)
                var_explained = d_cumsum / np.sum(d)
                n_count = sum(var_explained <= 0.90) + 1

                # fitting model with significant pval
                x_model = x_train[:, 1:n_count + 1]
                X2 = sm.add_constant(x_model)
                est = sm.RLM(y_train_initial, X2, M=sm.robust.norms.HuberT())
                fitted_est = est.fit()
                pval = fitted_est.pvalues
                idx_est = tuple(np.where(pval < 0.05)[0])
                idx_est = idx_est[1:]

                x_model_significant = x_train[:, idx_est]
                x2_significant = sm.add_constant(x_model_significant)
                est_significant = sm.RLM(y_train_initial, x2_significant, M=sm.robust.norms.HuberT())
                fitted_est_significant = est_significant.fit()

                y_pred_train = fitted_est_significant.predict(x2_significant)

                # predict
                x2_test = sm.add_constant(x_test[:, idx_est])  # sigmoid
                y_pred_test = fitted_est_significant.predict(x2_test)

                # evaluate
                mae_train = mae(y_pred_train, y_train_initial)
                ns_train = ns(y_pred_train, y_train_initial)
                nslog_train = ns_log(y_pred_train, y_train_initial)

                mae_test = mae(y_pred_test, y_test_initial)
                ns_test = ns(y_pred_test, y_test_initial)
                nslog_test = ns_log(y_pred_test, y_test_initial)

                better = not self.best or ns_test > self.best["ns_test"]

                if better:
                    self.best = {
                        "kpca": kpca,

                        "kernel": kernel,
                        "deg": deg,
                        "n_count": n_count,
                        "idx_est": idx_est,
                        "var_explained": var_explained,
                        "fitted_est_significant": fitted_est_significant,
                        "y_pred_test": y_pred_test,

                        "pc_train": x_model_significant,
                        "pc_test": x2_test,

                        "mae_train": mae_train,
                        "ns_train": ns_train,
                        "nslog_train": nslog_train,
                        "mae_test": mae_test,
                        "ns_test": ns_test,
                        "nslog_test": nslog_test,
                        "x_test"
                        "x_train": x_train
                    }
        print(self.best["fitted_est_significant"].summary())
        self._plot_variance_explained(self.best["var_explained"], self.best['n_count'], folder)
        self._plot_kpca_summary(self.best['fitted_est_significant'].summary(), folder)

        print(f'MAE of {self.best["kernel"]} Kernel PCA Train: {self.best["mae_train"]} [kg/ha]')
        print(f'NS of {self.best["kernel"]} Kernel PCA Train: {self.best["ns_train"]} [kg/ha]')
        print(f'NS log of {self.best["kernel"]} Kernel PCA Train: {self.best["nslog_train"]} [kg/ha]')

        print(f'MAE of {self.best["kernel"]} Kernel PCA Test: {self.best["mae_test"]} [kg/ha]')
        print(f'NS of {self.best["kernel"]} Kernel PCA Test: {self.best["ns_test"]} [kg/ha]')
        print(f'NS log of {self.best["kernel"]} Kernel PCA Test: {self.best["nslog_test"]} [kg/ha]')

        self._plot_testdata_kpca(self.best["y_pred_test"], y_test_initial, folder)

        # ################################## KPCA ALL DATA ##############################################
        # transform all data and estimate them
        x_all = self.best["kpca"].transform(x)
        x2_all = sm.add_constant(
            x_all[:, self.best["idx_est"]]
        )  # components cumulative to >90% variance and significant in regression model
        self.y_pred_all = y_pred_all = self.best["fitted_est_significant"].predict(x2_all)

        self._plot_pc_vs_error(x_all[:, self.best["idx_est"]], self.y_pred_all, y, folder=folder)

        print(f'MAE of {self.best["kernel"]} Kernel PCA Test: {mae(y_pred_all, y)} [kg/ha]')
        print(f'NS of {self.best["kernel"]} Kernel PCA Test: {ns(y_pred_all, y)} [-]')
        print(f'NS log of {self.best["kernel"]} Kernel PCA Test: {ns_log(y_pred_all, y)} [-]')
        self._plot_alldata(y_pred_all, y, folder=folder)

        # Hack for also predicting final yield when predictand is yield difference
        if self.var_Y in ["YieldDiff", "YieldObs"]:
            yield_adj = self.best["y_pred_test"]
            if self.var_Y == "YieldDiff":
                yield_adj += x_test_model_yield

            yield_obs = y_test_initial

            print(f'MAE of {self.best["kernel"]} Predicted Yield: {mae(yield_adj, yield_obs)} [kg/ha]')
            print(f'NS of {self.best["kernel"]} Predicted Yield: {ns(yield_adj, yield_obs)} [-]')
            print(f'NS log of {self.best["kernel"]} Predicted Yield: {ns_log(yield_adj, yield_obs)} [-]')
            self._plot_adjusted_predicted_yield(yield_adj, yield_obs, folder=folder)

        self._plot_histogram(y_pred_all, y, folder=folder)

    def _plot_variance_explained(self, var_explained, n_count, folder):
        x_plot = np.arange(1, len(var_explained) + 1)
        y_plot = var_explained

        plt.figure(figsize=(10, 7))
        plt.plot(x_plot, y_plot, c="blue")
        plt.axhline(y=0.90, color='r', linestyle='--')
        plt.axvline(x=n_count, color='k', linestyle='--')
        plt.xlim(0, 50)
        plt.ylim(0, 1)
        plt.grid(color='k', linestyle='-', linewidth=0.1)
        plt.text(n_count + 1, 0.5, 'Number of PCs to explain >90% variance: {}'.format(n_count), ha='left', va='center',
                 fontsize=18)
        plt.text(-5, 0.9, 'var=0.9', ha='left', va='center')
        plt.title("Variance Explained, Kernel: {}".format(self.best["kernel"]), fontsize=18)
        plt.xlabel("Principal Components in Feature Space", fontsize=18)
        plt.ylabel("Variance Explained [-]", fontsize=18)

        plt.savefig(f'{folder}/varExplained_{self.best["kernel"]}_deg{self.best["deg"]}.png', dpi=300,
                    bbox_inches="tight")
        plt.close()

    def _plot_kpca_summary(self, summary, folder):
        plt.rc('figure', figsize=(8, 5))
        plt.text(0.01, 0.05, str(summary), {'fontsize': 10},
                 fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        plt.tight_layout()

        plt.savefig(f'{folder}/KPCA_Summary_{self.best["kernel"]}_deg{self.best["deg"]}.png', dpi=300,
                    bbox_inches="tight")
        plt.close()

    def _plot_testdata_kpca(self, y_pred, y_test, folder):
        # plots test data KPCA
        plt.figure(figsize=(10, 8))
        x_plot = y_pred
        y_plot = y_test

        plt.scatter(x_plot, y_plot, c="red", s=20, edgecolor='k')
        min_ = min(x_plot.min(), y_plot.min())
        max_ = max(x_plot.max(), y_plot.max())

        fitted_estplot = sm.RLM(y_plot, sm.add_constant(x_plot), M=sm.robust.norms.HuberT()).fit()
        rsquared_adj = sm.OLS(y_plot, sm.add_constant(x_plot)).fit().rsquared_adj
        adj_r2Plot = round(rsquared_adj, 3)
        plt.text(min_ + 300, max_ - 300, '$r^2$ = {}'.format(adj_r2Plot), ha='left', va='center', fontsize=18)
        plt.text(min_ + 300, max_ - 500,
                 'µ = {}'.format(round(fitted_estplot.params[fitted_estplot.params.index[1]], 3)), ha='left',
                 va='center', fontsize=18)

        plt.plot(np.arange(min_, max_), np.arange(min_, max_), color="black")
        plt.plot(np.unique(x_plot), np.unique(fitted_estplot.predict(sm.add_constant(x_plot))), color="black",
                 linestyle='--')
        plt.xlim(min_, max_)
        plt.ylim(min_, max_)
        plt.title(f"Predicted vs. Observed {self.var_Y}, Kernel: {self.best['kernel']} (test data)", fontsize=18)
        plt.xlabel(f"Predicted {self.var_Y} [kg/ha]", fontsize=18)
        plt.ylabel(f"Observed {self.var_Y} [kg/ha]", fontsize=18)
        plt.grid(color='k', linestyle='-', linewidth=0.1)
        plt.savefig(f'{folder}/KPCA_{self.best["kernel"]}_validate_{self.var_Y}_deg{self.best["deg"]}.png', dpi=300,
                    bbox_inches="tight")
        plt.close()

    def _plot_alldata(self, y_pred, y, folder):
        # plots
        plt.figure(figsize=(10, 8))
        x_plot = y_pred
        y_plot = y

        plt.scatter(x_plot, y_plot, c="red", s=20, edgecolor='k')
        min_ = min(x_plot.min(), y_plot.min())
        max_ = max(x_plot.max(), y_plot.max())

        fitted_est_plot = sm.RLM(y_plot, sm.add_constant(x_plot), M=sm.robust.norms.HuberT()).fit()
        rsquared_adj = sm.OLS(y_plot, sm.add_constant(x_plot)).fit().rsquared_adj
        adj_r2_plot = round(rsquared_adj, 3)

        plt.text(min_ + 300, max_ - 100, '$r^2$ = {}'.format(adj_r2_plot), ha='left', va='center', fontsize=18)
        plt.text(min_ + 300, max_ - 500,
                 'µ = {}'.format(round(fitted_est_plot.params[fitted_est_plot.params.index[1]], 3)),
                 ha='left', va='center', fontsize=18)

        plt.plot(np.arange(min_, max_), np.arange(min_, max_), color="black")
        plt.plot(np.unique(x_plot), np.unique(fitted_est_plot.predict(sm.add_constant(x_plot))), color="black",
                 linestyle='--')
        plt.xlim(min_, max_)
        plt.ylim(min_, max_)
        plt.grid(color='k', linestyle='-', linewidth=0.1)
        plt.title(f"Predicted vs. Observed {self.var_Y}, Kernel: {self.best['kernel']}", fontsize=18)
        plt.xlabel(f"Predicted {self.var_Y} [kg/ha]", fontsize=18)
        plt.ylabel(f"Observed {self.var_Y} [kg/ha]", fontsize=18)

        plt.savefig(f'{folder}/KPCA_{self.best["kernel"]}_alldata_{self.var_Y}_deg{self.best["deg"]}.png', dpi=300,
                    bbox_inches="tight")
        plt.close()

    def _plot_adjusted_predicted_yield(self, y_pred, y_obs, folder):
        print("range yield obs: ", y_obs.min(), y_obs.max())
        # plot adjusted yield
        fitted_estplot = sm.RLM(y_obs, sm.add_constant(y_pred), M=sm.robust.norms.HuberT()).fit()

        plt.figure(figsize=(10, 8))
        plt.fill_between(np.unique(y_pred), np.unique(fitted_estplot.predict(sm.add_constant(y_pred))) + 450,
                         np.unique(fitted_estplot.predict(sm.add_constant(y_pred))) - 450, color='yellow', alpha=0.5)
        plt.scatter(y_pred, y_obs, c="red", s=20, edgecolor='k')
        min_ = min(y_pred.min(), y_obs.min())
        max_ = max(y_pred.max(), y_obs.max())

        xplot2_sm = sm.add_constant(y_pred)
        est_plot_sm = sm.RLM(y_obs, xplot2_sm, M=sm.robust.norms.HuberT())
        fitted_estplot = est_plot_sm.fit()
        rsquared_adj = sm.OLS(y_obs, xplot2_sm).fit().rsquared_adj
        adj_r2Plot = round(rsquared_adj, 3)

        plt.text(min_ + 300, max_ - 200, '$r^2$ = {}'.format(adj_r2Plot), ha='left', va='center', fontsize=18)
        plt.text(min_ + 300, max_ - 500,
                 'µ = {}'.format(round(fitted_estplot.params[fitted_estplot.params.index[1]], 3)), ha='left',
                 va='center', fontsize=18)

        plt.plot(np.arange(min_, max_), np.arange(min_, max_), color="black")
        plt.plot(np.unique(y_pred), np.unique(fitted_estplot.predict(sm.add_constant(y_pred))), color="black",
                 linestyle='--')
        plt.xlim(min_, max_)
        plt.ylim(min_, max_)
        plt.title("Predicted Yield vs. Observed Yield", fontsize=18)
        plt.xlabel("Predicted Yield [kg/ha]", fontsize=18)
        plt.ylabel("Observed Yield [kg/ha]", fontsize=18)
        plt.savefig(f'{folder}/KPCA_{self.best["kernel"]}_finalPredictedYield_deg{self.best["deg"]}.png', dpi=300,
                    bbox_inches="tight")
        plt.close()

    def _plot_pc_vs_error(self, pcs, pred, obs, folder: str):
        self._plot_pc_vs_obs(pcs, obs, folder=folder)
        n_pcs = pcs.shape[1]
        error = (pred - obs).values
        fig, axes = plt.subplots(math.ceil(n_pcs / 2), 2, figsize=(15, 17))
        for i in range(n_pcs):
            pc = pcs[..., i]
            ax = axes.flatten()[i]
            ax.scatter(pc, error, c="red", s=20, edgecolor='k')
            plotfit = np.arange(pc.min(), pc.max(), 0.5)
            fit = np.poly1d(np.polyfit(pc, error, 1))
            model = HuberRegressor().fit(pc.reshape(-1, 1), error.reshape(-1, 1))
            adj_r2Plot = round(model.score(pc.reshape(-1, 1), error.reshape(-1, 1)), 3)

            ax.plot(plotfit, fit(plotfit), color="black", linestyle='--', linewidth=2)

            ax.set_title('PC {} $r^2$ = {}'.format(i + 1, adj_r2Plot))

        fig.suptitle('Total Error by Principal Components')
        fig.supylabel('Total error')
        plt.savefig(f'{folder}/PC_vs_error_{self.best["kernel"]}_deg{self.best["deg"]}.png', dpi=300,
                    bbox_inches="tight")
        plt.close()

    def _plot_pc_vs_obs(self, pcs, obs, folder: str):
        n_pcs = pcs.shape[1]
        y = (obs).values
        fig, axes = plt.subplots(math.ceil(n_pcs / 2), 2, figsize=(15, 17))
        for i in range(n_pcs):
            pc = pcs[..., i]
            ax = axes.flatten()[i]
            print(ax)
            ax.scatter(pc, y, c="red", s=20, edgecolor='k')
            plotfit = np.arange(pc.min(), pc.max(), 0.5)
            fit = np.poly1d(np.polyfit(pc, y, 1))
            model = HuberRegressor().fit(pc.reshape(-1, 1), y.reshape(-1, 1))
            adj_r2Plot = round(model.score(pc.reshape(-1, 1), y.reshape(-1, 1)), 3)

            ax.plot(plotfit, fit(plotfit), color="black", linestyle='--', linewidth=2)

            ax.set_title('PC {} $r^2$ = {}'.format(i + 1, adj_r2Plot))

        fig.suptitle('Observed value by Principal Components')
        fig.supylabel('Observed value')
        plt.savefig(f'{folder}/PC_vs_obs_{self.best["kernel"]}_deg{self.best["deg"]}.png', dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_histogram(self, y_pred, y, folder):
        # %% histogram of er
        a = y - y_pred

        mean, std = norm.fit(a)
        print("mean", mean, a.mean())
        x = np.linspace(mean - 3 * std, mean + 3 * std, 100)

        bins = np.linspace(min(a.min(), mean - 3 * std), max(a.max(), mean + 3 * std), 100)
        labels = ['$\epsilon_r$']

        plt.figure(figsize=(12, 7))
        plt.hist(a, bins, rwidth=0.8, label=labels, density=True)
        plt.plot(x, stats.norm.pdf(x, mean, std), label=f'Assumed distribution of $\epsilon_r$ ($\mu$={round(mean)}, $\sigma$={round(std)})')
        plt.title('Histogram of residual error ($\epsilon_r$)', fontsize=18)
        plt.xlabel('Residual error ($\epsilon_r$) [kg/ha]', fontsize=18)
        plt.ylabel('Probability', fontsize=18)
        plt.legend()
        plt.savefig(f'{folder}/HistEr_{self.best["kernel"]}_deg{self.best["deg"]}.png', dpi=300, bbox_inches="tight")
        plt.close()


def remove_outliers(dataframe):
    to_keep = ((dataframe - dataframe.mean()) / dataframe.std()).abs() <= 3.5
    return dataframe[to_keep].dropna()


if __name__ == "__main__":
    dat = pd.read_excel(f"{DATA_DIR}/KPCA/KPCA_dat_adj_withCosts.xlsx", header=0)
    dat_irr = pd.read_excel(f"{DATA_DIR}/KPCA/KPCA_dat_adj_irr.xlsx", header=0)

    default_vars_X = [
        "AreaCot", "ChildrenHelp", "Evap", "FertAmount", "Irr",
        "Lat", "Long",
        "Prec", "SeedsCost", "SoilDepth"
    ]

    modelyield_vars_X = default_vars_X + ["YieldModel"]
    modelyield_model = KPCAModel(modelyield_vars_X, var_y="YieldDiff",
                                 data=remove_outliers(dat))
    modelyield_model.train_and_plot(f"{PLOT_DIR}/modelYield")

    modelyield_model_irr = KPCAModel(modelyield_vars_X, var_y="YieldDiff", data=remove_outliers(dat_irr))
    modelyield_model_irr.train_and_plot(f"{PLOT_DIR}/modelYield_irr")

    obsYield_vars_X = default_vars_X + ["YieldObs"]
    obsYield_model = KPCAModel(obsYield_vars_X, var_y="YieldDiff",
                               data=remove_outliers(dat))
    obsYield_model.train_and_plot(f"{PLOT_DIR}/base")

    obsYield_model_irr = KPCAModel(obsYield_vars_X, var_y="YieldDiff", data=remove_outliers(dat_irr))
    obsYield_model_irr.train_and_plot(f"{PLOT_DIR}/base_irr")

    no_sdm_model = KPCAModel(default_vars_X, var_y="YieldObs", data=remove_outliers(dat))
    no_sdm_model.train_and_plot(f"{PLOT_DIR}/noSDM")

    no_sdm_model_irr = KPCAModel(default_vars_X, var_y="YieldObs", data=remove_outliers(dat_irr))
    no_sdm_model_irr.train_and_plot(f"{PLOT_DIR}/noSDM_irr")

    noyield_model = KPCAModel(default_vars_X, var_y="YieldDiff",
                              data=remove_outliers(dat))
    noyield_model.train_and_plot(f"{PLOT_DIR}/noYield")

    noyield_model_irr = KPCAModel(default_vars_X, var_y="YieldDiff", data=remove_outliers(dat_irr))
    noyield_model_irr.train_and_plot(f"{PLOT_DIR}/noYield_irr")

    emulator_vars_X = default_vars_X + ["PestCost", "FertCost"]
    emulator_model = KPCAModel(emulator_vars_X, var_y="YieldModel",
                               data=remove_outliers(dat))
    emulator_model.train_and_plot(f"{PLOT_DIR}/emulator")

    emulator_model_irr = KPCAModel(default_vars_X, var_y="YieldModel", data=remove_outliers(dat_irr))
    emulator_model_irr.train_and_plot(f"{PLOT_DIR}/emulator_irr")

    avg_obs_yield_model_vars_X = default_vars_X + ["averageObsYield"]
    dat["averageObsYield"] = dat["YieldObs"].mean()
    avg_obs_yield_model = KPCAModel(avg_obs_yield_model_vars_X, var_y="YieldObs",
                                    data=remove_outliers(dat))
    avg_obs_yield_model.train_and_plot(f"{PLOT_DIR}/averageObsYield")

    dat_irr["averageObsYield"] = dat_irr["YieldObs"].mean()
    avg_obs_yield_model_irr = KPCAModel(avg_obs_yield_model_vars_X, var_y="YieldObs", data=remove_outliers(dat_irr))
    avg_obs_yield_model_irr.train_and_plot(f"{PLOT_DIR}/averageObsYield_irr")

    WithCostsModelYieldNoPriceObsYield_vars_X = default_vars_X + ["PestCost", "FertCost", "YieldModel"]
    WithCostsModelYieldNoPriceObsYield_model = KPCAModel(WithCostsModelYieldNoPriceObsYield_vars_X, var_y="YieldDiff",
                                                         data=remove_outliers(dat))
    WithCostsModelYieldNoPriceObsYield_model.train_and_plot(f"{PLOT_DIR}/WithCostsModelYieldNoPriceObsYield")

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
    Yplot = dat_irr['YieldObs']

    Xplot2 = sm.add_constant(Xplot_irr)
    estPlot = sm.RLM(Yplot, Xplot2, M=sm.robust.norms.HuberT())
    estPlot2 = estPlot.fit()

    plt.figure(figsize=(10, 8))
    plt.fill_between(np.unique(Xplot_irr), np.unique(estPlot2.predict(sm.add_constant(Xplot_irr))) + 450,
                     np.unique(estPlot2.predict(sm.add_constant(Xplot_irr))) - 450, color='yellow', alpha=0.5)
    plt.scatter(Xplot_irr, Yplot, c="red", s=20, edgecolor='k', label='Predicted yield with new ponds')
    plt.scatter(Xplot_noIrr, Yplot, c="blue", s=20, edgecolor='k', label='Predicted yield')

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
