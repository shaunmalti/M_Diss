import os
import sys

import pandas as pd
import pandas_datareader.data as web
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model

from numpy import cumsum, log, polyfit, sqrt, std, subtract
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
# %matplotlib inline
import csv

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        # mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
        plt.show()
    return

def hurst(ts):
    # calculate standard deviation of differenced series using various lags
    lags = range(2, 20)
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    m = polyfit(log(lags), log(tau), 1)
    return m[0] * 2.0

data = pd.read_csv('C_generated_unnormed.csv', header=None)
vals = data.values
vals = np.ndarray.flatten(vals[0:50])

test2 = np.ndarray.flatten(vals[50:100])
print(hurst(cumsum(test2)))

C_data = pd.read_csv('C.txt')["Close"].values
x = np.r_[C_data[0], np.ndarray.flatten(vals)].cumsum()
# another thing to do - do not do differencing on data when training
# print(hurst(vals))
# tsplot(vals,lags=30)
plt.plot(vals)
plt.show()

#


exit()

"""PARAMS - alphas = np.array([0.5, -0.25])
betas = np.array([0.5, -0.3])"""
"""this is a check for move avg autoreg"""
data = pd.read_csv('Move_Avg_AutoReg_1000.csv', header=None)
vals = data.values
for i in range(int(len(vals)/10)):
    try:
        mdl = smt.ARMA(np.ndarray.flatten(vals[(i*10):(i*10)+10]), order=(2, 2)).fit(
            maxlag=30, method='mle', trend='nc')
        print(mdl.summary())
    except:
        pass
# mdl = smt.ARMA(np.ndarray.flatten(vals), order=(2, 2)).fit(
#     maxlag=30, method='mle', trend='nc')
# print(mdl.summary())
exit()

"""This is a check for moving average x1000"""
data = pd.read_csv('Move_Avg_1000.csv', header=None)
vals = data.values
for i in range(int(len(vals)/10)):
    mdl = smt.ARMA(np.ndarray.flatten(vals[(i*10):(i*10)+10]), order=(0, 1)).fit(
                maxlag=30, method='mle', trend='nc')
    print(mdl.summary())
mdl = smt.ARMA(np.ndarray.flatten(vals), order=(0, 1)).fit(
                maxlag=30, method='mle', trend='nc')
print(mdl.summary())


exit()

option = 2

if option == 1:
    data = pd.read_csv('Regressive_2.csv',header=None)
    vals = data.values
    for i in range(0,len(vals)):
        # tsplot(vals,lags=30)
        # Fit an AR(p) model to simulated AR(1) model with alpha = 0.6
        mdl = smt.AR(vals[i]).fit(maxlag=30, ic='aic', trend='nc')
        # % time
        est_order = smt.AR(vals[i]).select_order(
            maxlag=30, ic='aic', trend='nc')

        print('\nalpha estimate: {:3.5f} | best lag order = {}'
          .format(mdl.params[0], est_order))
elif option == 2:
    data = pd.read_csv('Move_Avg.csv', header=None)
    vals = data.values
    for i in range(0, len(vals)):
        max_lag = 30
        # tsplot(vals[i],lags=30)
        mdl = smt.ARMA(vals[i], order=(0, 1)).fit(
            maxlag=max_lag, method='mle', trend='nc')
        print(mdl.summary())
        print("************************************************************")
elif option == 3:
    """PARAMS - alphas = np.array([0.5, -0.25])
    betas = np.array([0.5, -0.3])"""
    data = pd.read_csv('Move_Avg_AutoReg.csv', header=None)
    vals = data.values
    for i in range(0, len(vals)):
        try:
            mdl = smt.ARMA(vals[i], order=(2, 2)).fit(
                maxlag=30, method='mle', trend='nc')
            print(mdl.summary())
        except:
            pass
