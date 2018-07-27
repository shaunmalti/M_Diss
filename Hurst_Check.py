import matplotlib
matplotlib.use('TkAgg')
from numpy import cumsum, log, polyfit, sqrt, std, subtract
import numpy as np
import os
import pandas as pd



def import_csv(filename, dir):
    # import, set header names and change columns depending on content
    data = pd.read_csv(dir + filename,sep=',',names=['Date','Open','High','Low','Close','Volume'])

    # redefine data as being 0/1. 1 when close > open and 0 in all other cases
    data_np = data.values
    new_array = []

    for i in range(0,len(data_np)):
        if data_np[i][4] > data_np[i][1]:
            new_array.append(1)
        else:
            new_array.append(-1)

    for i in range(1, len(new_array)):
        new_array[i] = new_array[i] + new_array[i - 1]

    return np.array(new_array)

def hurst(ts):
    # calculate standard deviation of differenced series using various lags
    lags = range(2, 20)
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    m = polyfit(log(lags), log(tau), 1)
    return m[0] * 2.0

dir = "./Data/daily_tickbot_data/"

for filename in os.listdir(dir):
    if filename.endswith(".txt"):
        data = import_csv(filename, dir)
        print(hurst(data))