# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:35:17 2021

@author: denni
"""
import math
import os
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.decomposition import KernelPCA, PCA
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({'font.size': 12})


def mae(pred, obs):
    return np.sum(abs(pred - obs)) / len(pred)


def ns(pred, obs):
    return 1 - (np.sum((pred - obs) ** 2) / np.sum((obs.mean() - obs) ** 2))


def ns_log(pred, obs):
    return 1 - (np.sum((np.log10(pred) - np.log10(obs)) ** 2) / np.sum((np.log10(obs.mean()) - np.log10(obs)) ** 2))


def r2(pred, obs):
    return r2_score(obs, pred)


class KPCAModel(object):
    def __init__(self, vars_x, var_y, data, kernels=None):
        self.vars_X = vars_x
        self.var_Y = var_y
        self.data = data
        self.kernels = kernels or ["rbf", "sigmoid", "poly", "cosine"]

        self.best = {}
        self.y_pred_all = None
        self.saved_statistics = {}

    def train_and_plot(self, folder):
        if not os.path.isdir(folder):
            os.makedirs(folder)
        vars_x_with_model_yield = self.vars_X if "YieldModel" in self.vars_X else [*self.vars_X, "YieldModel"]
        vars_x_with_model_yield_and_error = vars_x_with_model_yield if "YieldDiff" in vars_x_with_model_yield else [*vars_x_with_model_yield, "YieldDiff"]
        x = StandardScaler().fit_transform(self.data[vars_x_with_model_yield_and_error])
        y = self.data[self.var_Y]

        # split training and test data
        x_train_initial, x_test_initial, y_train_initial, y_test_initial = train_test_split(
            x, y, test_size=0.25,
            random_state=1)
        yield_model_index = vars_x_with_model_yield_and_error.index("YieldDiff")
        x_test_model_yield = x_test_initial[:, yield_model_index]

        error_model_index = vars_x_with_model_yield_and_error.index("YieldDiff")
        x_train_model_error = x_train_initial[:, error_model_index]
        x_model_error = x[:, error_model_index]

        if "YieldModel" not in self.vars_X:
            x_test_initial = np.delete(x_test_initial, yield_model_index, axis=1)
            x_train_initial = np.delete(x_train_initial, yield_model_index, axis=1)
            x = np.delete(x, yield_model_index, axis=1)

        if "YieldDiff" not in self.vars_X:
            x_test_initial = np.delete(x_test_initial, error_model_index, axis=1)
            x_train_initial = np.delete(x_train_initial, error_model_index, axis=1)
            x = np.delete(x, error_model_index, axis=1)

        # Run PCA
        pca = PCA()
        x_train = pca.fit_transform(x_train_initial)
        d = pca.explained_variance_ratio_
        var_explained = np.cumsum(d)
        n_count = sum(var_explained <= 0.90) + 1

        # fitting model with significant pval
        x_model = x_train[:, 1:n_count + 1]
        X2 = sm.add_constant(x_model)
        est = sm.RLM(y_train_initial, X2, M=sm.robust.norms.HuberT())
        fitted_est = est.fit()
        pval = fitted_est.pvalues
        idx_est = tuple(np.where(pval < 0.05)[0])
        idx_est = idx_est[1:]
        most_important = [np.array(self.vars_X)[np.abs(pca.components_[i]) >= 0.5] for i in idx_est]

        x_model_significant = x_train[:, idx_est]

        model = sm.OLS(x_model_error, x).fit()
        rows = []
        for i, var_name in enumerate(self.vars_X):
            rows.append(
                "\\textbf{" + var_name + "} & " + f"{round_(model.params[i])} & {round_(model.pvalues[i])} \\\\")

        tableLaTEX = """
        \\begin{tabular}{lrr}
        \\textbf{} & \\textbf{Coef} & \\textbf{P-value} \\\\ \hline
        """ + "\n".join(rows) + """\hline
        \end{tabular}
        """
        with open(f'{folder}/linregTable.txt', encoding="utf8", mode='w') as f:
            f.write(tableLaTEX)

        error = x_train_model_error

        plt.clf()
        plt.figure(figsize=(15, 17))
        plt.title('Total Error by Prediction by Each PC')
        plt.xlabel('Month')
        plt.ylabel('Total Error [kg/ha]')
        plt.rcParams.update({'font.size': 18})
        plotfit = np.arange(-3, 5.5, 0.5)

        for i in range(len(idx_est)):
            plt.subplot(len(idx_est) // 2 + 1, 2, i + 1)
            plt.scatter(x_model_significant[:, i], error, c="red", s=20, edgecolor='k')
            fit = np.poly1d(np.polyfit(x_model_significant[:, i], error, 1))
            model = LinearRegression().fit(x_model_significant[:, i].reshape(-1, 1), error.reshape(-1, 1))
            adj_r2Plot = round(model.score(x_model_significant[:, i].reshape(-1, 1), error.reshape(-1, 1)), 3)

            plt.text(1.2, 0, 'PC {}'.format(i + 1), ha='center', va='center', fontsize=22)
            plt.text(5.8, 0, '$r^2$ = {}'.format(adj_r2Plot), ha='right', va='center', fontsize=22)
            plt.plot(plotfit, fit(plotfit), color="black", linestyle='--', linewidth=2)
            plt.xlim(-3.5, 6)
            plt.grid(color='k', linestyle='-', linewidth=0.1)
            plt.tight_layout(pad=1.0)

        plt.text(7, 10, 'Total Error by Principal Components', ha='center', fontsize=30)
        plt.text(7, -10, 'Principal components ', ha='center', fontsize=30)
        plt.text(-6, 10, 'Total error [kg/ha]', va='center', rotation='vertical', fontsize=30)
        plt.savefig(f'{folder}/lPCA.png', dpi=300, bbox_inches="tight")
        plt.close()

        # Running KPCA
        for kernel in self.kernels:
            degs = range(2, 6) if kernel == "poly" else [None]
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
                r2_test = r2(y_pred_test, y_test_initial)
                print("here", ns_test, r2_test)

                y_pred_all = fitted_est_significant.predict(sm.add_constant(kpca.transform(x)[:, idx_est]))
                mae_all = mae(y_pred_all, y)
                ns_all = ns(y_pred_all, y)
                kernel_name = f"Poly deg {deg}" if kernel == 'poly' else \
                    {"sigmoid": "Sigmoid", "rbf": "RBF", "cosine": "Cosine"}[kernel]
                self.saved_statistics[kernel_name] = {
                    "MAE_train": mae_train, "MAE_test": mae_test, "MAE_all": mae_all, "NS_test": ns_test,
                    "NS_train": ns_train, "NS_all": ns_all,
                    "R2_test": r2_test,
                    "NSlog_test": nslog_test
                }

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
                        "x_test": x_test,
                        "x_train": x_train,
                        "r2_test": r2_test
                    }
        print(self.best["fitted_est_significant"].summary())
        self._plot_variance_explained(self.best["var_explained"], self.best['n_count'], folder)
        self._plot_kpca_summary(self.best['fitted_est_significant'].summary(), folder)

        print(f'MAE of {self.best["kernel"]} Kernel PCA Train: {self.best["mae_train"]} [kg/ha]')
        print(f'NS of {self.best["kernel"]} Kernel PCA Train: {self.best["ns_train"]} [kg/ha]')
        print(f'NS log of {self.best["kernel"]} Kernel PCA Train: {self.best["nslog_train"]} [kg/ha]')

        print(f'MAE of {self.best["kernel"]} Kernel PCA Test: {self.best["mae_test"]} [kg/ha]')
        print(f'NS of {self.best["kernel"]} Kernel PCA Test: {self.best["ns_test"]}')
        print(f'NS log of {self.best["kernel"]} Kernel PCA Test: {self.best["nslog_test"]}')
        print(f'R2 log of {self.best["kernel"]} Kernel PCA Test: {self.best["r2_test"]}')

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

    def save_tables(self, directory: str):
        self.save_kernel_validation_table(file=os.path.join(directory, "kernel validation.txt"))

    def save_kernel_validation_table(self, file: str):
        if not self.saved_statistics:
            raise ValueError("Run KPCAModel.train_and_plot first.")

        rows = []
        for kernel, st in self.saved_statistics.items():
            rows.append(
                "\\textbf{" + kernel + "} & " + f"{round(st['MAE_train'])} & {round_(st['NS_train'])} & {round(st['MAE_test'])} & {round_(st['NS_test'])} & {round(st['MAE_all'])} & {round_(st['NS_all'])} \\\\")

        tableLaTEX = """
\\begin{tabular}{lrrrrrr}
\\textbf{} & \multicolumn{2}{c}{\\textbf{Train Data}} & \multicolumn{2}{c}{\\textbf{Test Data}} & \multicolumn{2}{c}{\\textbf{All Data}} \\\\
\\textbf{} & \\textbf{MAE {[}$\\frac{kg}{ha}${]}} & \\textbf{NS {[}-{]}} & \\textbf{MAE {[}$\\frac{kg}{ha}${]}} & \\textbf{NS {[}-{]}} & \\textbf{MAE {[}$\\frac{kg}{ha}${]}} & \\textbf{NS {[}-{]}} \\\\ \hline
""" + "\n".join(rows) + """\hline
\end{tabular}
"""
        folder = os.path.dirname(file)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(file, encoding="utf8", mode='w') as f:
            f.write(tableLaTEX)

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
        plt.savefig(f'{folder}/KPCA_validate.png', dpi=300,
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

        plt.savefig(f'{folder}/KPCA_allData_predicted_vs_obs.png', dpi=300,
                    bbox_inches="tight")
        plt.close()

    def _plot_adjusted_predicted_yield(self, y_pred, y_obs, folder):
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
        plt.savefig(f'{folder}/KPCA_FinalPredictedYield.png', dpi=300,
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
        plt.savefig(f'{folder}/PC_vs_error.png', dpi=300,
                    bbox_inches="tight")
        plt.close()

    def _plot_pc_vs_obs(self, pcs, obs, folder: str):
        n_pcs = pcs.shape[1]
        y = (obs).values
        fig, axes = plt.subplots(math.ceil(n_pcs / 2), 2, figsize=(15, 17))
        for i in range(n_pcs):
            pc = pcs[..., i]
            ax = axes.flatten()[i]
            ax.scatter(pc, y, c="red", s=20, edgecolor='k')
            plotfit = np.arange(pc.min(), pc.max(), 0.5)
            fit = np.poly1d(np.polyfit(pc, y, 1))
            model = HuberRegressor().fit(pc.reshape(-1, 1), y.reshape(-1, 1))
            adj_r2Plot = round(model.score(pc.reshape(-1, 1), y.reshape(-1, 1)), 3)

            ax.plot(plotfit, fit(plotfit), color="black", linestyle='--', linewidth=2)

            ax.set_title('PC {} $r^2$ = {}'.format(i + 1, adj_r2Plot))

        fig.suptitle('Observed value by Principal Components')
        fig.supylabel('Observed value')
        plt.savefig(f'{folder}/PC_vs_obs.png', dpi=300, bbox_inches="tight")
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
        plt.plot(x, stats.norm.pdf(x, mean, std),
                 label=f'Assumed distribution of $\epsilon_r$ ($\mu$={round(mean)}, $\sigma$={round(std)})')
        plt.title('Histogram of residual error ($\epsilon_r$)', fontsize=18)
        plt.xlabel('Residual error ($\epsilon_r$) [kg/ha]', fontsize=18)
        plt.ylabel('Probability', fontsize=18)
        plt.legend()
        plt.savefig(f'{folder}/Error histogram.png', dpi=300, bbox_inches="tight")
        plt.close()


def remove_outliers(dataframe):
    to_keep = ((dataframe - dataframe.mean()) / dataframe.std()).abs() <= 3.5
    return dataframe[to_keep].dropna()


def round_(x):
    return float('%.3g' % x)


def save_model_comparison_table(models: Dict[str, Union[KPCAModel, Dict]], file: str):
    rows = []
    for label, model in models.items():
        if isinstance(model, dict):
            st = model
        else:
            st = model.best
        rows.append(
            f"{label} & {round(st['mae_test'])} & {round(st['ns_test'], 4)} & {round(st['nslog_test'], 4)} & {round(st['r2_test'], 3)} \\\\"
        )

    tableLaTEX = """
\\begin{tabular}{lllll}
\hline
 & $MAE$ [kg/ha] & $NS$ [-] & $NS_{log}$ [-] & $r^2$ [-] \\\\ \hline
""" + "\n".join(rows) + """\hline
\end{tabular}
    """
    folder = os.path.dirname(file)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(file, encoding="utf8", mode='w') as f:
        f.write(tableLaTEX)
