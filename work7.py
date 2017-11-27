# -*- coding:utf-8 -*-
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from matplotlib import pyplot
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
####Task 1: Loading the data – 5 pts
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
volume = pd.read_csv('volume_per_year.csv',index_col='Month',date_parser=dateparse)#read data
print(volume.head())
print(volume.head().index)

#####Task 2: Stationarity – 5 pts
#QA
volume.plot()
pyplot.show()
#QC
from statsmodels.tsa.stattools import adfuller

# def test_stationarity(timeseries):
#     # Determing rolling statistics
#     ma = pd.rolling_mean(timeseries, window=12)
#     rolstd = pd.rolling_std(timeseries, window=12)

#     # Plot rolling statistics:
#     orig = plt.plot(timeseries, color='blue', label='Original')
#     mean = plt.plot(ma, color='green', label='Rolling Mean')
#     std = plt.plot(rolstd, color='red', label='Rolling Std')
#     plt.legend(loc='best')
#     plt.title('Rolling Mean & Standard Deviation')
#     plt.show(block=False)
#     pyplot.show(block=False)

#     # Perform Dickey-Fuller test:
#     print 'Results of Dickey-Fuller Test:'
#     dftest = adfuller(timeseries, autolag='AIC')
#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
#     for key, value in dftest[4].items():
#         dfoutput['Critical Value (%s)' % key] = value
#     print dfoutput

# test_stationarity(volume)

ma = volume.rolling(window=12,center=False).mean()
msd = volume.rolling(window=12,center=False).std()
#msd = pd.rolling_std(volume,window=12,center=False)
plt.plot(volume,'blue')
plt.plot(ma,'green')
plt.plot(msd,'red')
plt.show()
# The moving average and moving deviation is increasing over year
adtestoutput = adfuller(volume.volume)
print('Test Statistic:      %.6f' % adtestoutput[0])
print('p-value:      %.6f' % adtestoutput[1])
print('#Lags Used:      %.6f' % adtestoutput[2])
print('Number of Observations Used      %.6f' % adtestoutput[3])
print('Critical Value (1%%)      %.6f' % adtestoutput[4]['1%'])
print('Critical Value (10%%      %.6f' % adtestoutput[4]['10%'])
print('Critical Value (5%%)      %.6f' % adtestoutput[4]['5%'])


#####Task 3:Make a Time Series stationary – 5pts
#QA plot log
logvolume = np.log(volume)
logvolume.plot()
pyplot.show()
#QC log data plot
mavolume = pd.rolling_mean(logvolume,window=12)
plt.plot(logvolume)
plt.plot(mavolume,color='red')
pyplot.show()
#QD test
volume_without_trend = logvolume - mavolume
adtestoutput2 = adfuller(volume_without_trend[11:])
print('Test Statistic:      %.6f' % adtestoutput2[0])
print('p-value:      %.6f' % adtestoutput2[1])
print('#Lags Used:      %.6f' % adtestoutput2[2])
print('Number of Observations Used      %.6f' % adtestoutput2[3])
print('Critical Value (1%%)      %.6f' % adtestoutput2[4]['1%'])
print('Critical Value (10%%      %.6f' % adtestoutput2[4]['10%'])
print('Critical Value (5%%)      %.6f' % adtestoutput2[4]['5%'])
# The series volume_without_trend is stationary with confidence level 95%
#QE、F test ewma
ewma = pd.ewma(logvolume,halflife=12)
volume_without_trend_ewma = logvolume - ewma
adtestoutput3 = adfuller(volume_without_trend_ewma)
print('Test Statistic:      %.6f' % adtestoutput3[0])
print('p-value:      %.6f' % adtestoutput3[1])
print('#Lags Used:      %.6f' % adtestoutput3[2])
print('Number of Observations Used      %.6f' % adtestoutput3[3])
print('Critical Value (1%%)      %.6f' % adtestoutput3[4]['1%'])
print('Critical Value (10%%      %.6f' % adtestoutput3[4]['10%'])
print('Critical Value (5%%)      %.6f' % adtestoutput3[4]['5%'])

#####Task 4: Removing trend and seasonality with differencing – 5pts
logvol_ma_diff = logvolume - mavolume
logvol_ma_diff.dropna(inplace=True)
logvol_ma_diff.plot()
pyplot.show()

adtestoutput4 = adfuller(dif_logvolume)
print('Test Statistic:      %.6f' % adtestoutput4[0])
print('p-value:      %.6f' % adtestoutput4[1])
print('#Lags Used:      %.6f' % adtestoutput4[2])
print('Number of Observations Used      %.6f' % adtestoutput4[3])
print('Critical Value (1%%)      %.6f' % adtestoutput4[4]['1%'])
print('Critical Value (10%%      %.6f' % adtestoutput4[4]['10%'])
print('Critical Value (5%%)      %.6f' % adtestoutput4[4]['5%'])

#####Task5: Forecast Time Series – 5pts
#QA acf
from statsmodels.tsa.stattools import acf, pacf

logvol_ma_diff.dropna(inplace=True)
lag_acf = acf(logvol_ma_diff)
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(logvol_ma_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(logvol_ma_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
pyplot.show()

#QB、C、D、E arima
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(logvolume, order=(2, 1, 2))
#plt.plot(logvol_ma_diff)
results_ARIMA=model.fit(disp=-1)
plt.plot(results_ARIMA.fittedvalues, color='red')
pyplot.show()
#QF convert the predicted values into the original scale one
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

#QG Apply exponential to go back to the initial scale
predictions_ARIMA_log = pd.Series(logvolume.ix[0], index=logvolume.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
print(predictions_ARIMA_log.head())
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(predictions_ARIMA,color='red')
plt.plot(volume,'blue')
pyplot.show()
