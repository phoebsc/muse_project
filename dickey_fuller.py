import statsmodels
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

"""
Perform an augmented dickey-fuller test on data to assess stationarity. Based on code tutorial 
from: http://www.insightsbot.com/blog/1MH61d/augmented-dickey-fuller-test-in-python.
"""

class StationarityTests:
    @staticmethod
    def ADF_Stationarity_Test(timeseries, significance_level = 0.05, verbose=True):

        # Dickey-Fuller test:
        adf_test = adfuller(timeseries, autolag='AIC') # Use Akaiki Information Criterion to determine lag

        p_value = adf_test[1]
        is_stationary = None

        # Is the p-value less than our desired significance value? If so, reject the null hypothesis that the
        # time series has a unit root (i.e., reject the null hypothesis that the timeseries is non-stationary).
        if (p_value < significance_level):
            is_stationary = True
        else:
            is_stationary = False

        if verbose:
            df_results = pd.Series(adf_test[0:4],
                                  index=['ADF Test Statistic', 'P-Value', '# Lags Used', '# Observations Used'])

            # Add Critical Values
            for key, value in adf_test[4].items():
                df_results['Critical Value (%s)' % key] = value

            print('Augmented Dickey-Fuller Test Results:')
            print(df_results)

        return is_stationary




