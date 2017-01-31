
# coding: utf-8

# # Project Intro
# Forecast sales using store, promotion, and competitor data
# 
# Rossmann operates over 3,000 drug stores in 7 European countries. Currently, 
# Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.
# 
# In their first Kaggle competition, Rossmann is challenging you to predict 6 weeks of daily sales for 1,115 stores located across Germany. Reliable sales forecasts enable store managers to create effective staff schedules that increase productivity and motivation. 
# 
# ---
# You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. Note that some stores in the dataset were temporarily closed for refurbishment.
# 
# ### Files
# 
# 1. train.csv - historical data including Sales
# 1. test.csv - historical data excluding Sales
# 1. sample_submission.csv - a sample submission file in the correct format
# 1. store.csv - supplemental information about the stores
# ### Data fields
# 
# Most of the fields are self-explanatory. The following are descriptions for those that aren't.
# 
# - Id - an Id that represents a (Store, Date) duple within the test set
# - Store - a unique Id for each store
# - Sales - the turnover for any given day (this is what you are predicting)
# - Customers - the number of customers on a given day
# - Open - an indicator for whether the store was open: 0 = closed, 1 = open
# - StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
# - SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
# - StoreType - differentiates between 4 different store models: a, b, c, d
# - Assortment - describes an assortment level: a = basic, b = extra, c = extended
# - CompetitionDistance - distance in meters to the nearest competitor store
# - CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
# - Promo - indicates whether a store is running a promo on that day
# - Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
# - Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
# - PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
# 
# [Link to Kaggle](https://www.kaggle.com/c/rossmann-store-sales)

# # Import Data and Libraries

# In[673]:

import numba

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import math
from datetime import datetime
from datetime import timedelta

from pandas.tools.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error


# In[244]:

plt.rcParams['agg.path.chunksize'] = 100000


# In[245]:

store_raw = pd.read_csv('store.csv')
test_raw = pd.read_csv('test.csv')
train_raw = pd.read_csv('train.csv')


# In[246]:

store_raw.head(20)


# In[247]:

store_raw.info()


# In[248]:

train_raw.head(20)


# In[249]:

train_raw.tail()


# In[250]:

train_raw[train_raw['Store']==1]


# In[251]:

train_raw.info()


# # Data Wrangling

# ## merge two tables

# In[252]:

train_store_raw = train_raw.merge(store_raw, on='Store')


# In[253]:

train_store_raw.head(20)


# In[254]:

train_store_raw.info()


# ## change column types

# In[255]:

train_store_raw['Date_formated'] = pd.to_datetime(train_store_raw['Date'], format='%Y-%m-%d')


# In[256]:

train_store_raw.head()


# In[257]:

train_store_raw.info()


# ## convert StateHolidays
# a = public holiday, b = Easter holiday, c = Christmas, 0 = None

# In[258]:

state_holidays = {'a':1, 'b':2, 'c':3, '0':0}


# In[259]:

train_store_raw['StateHolidy_formatted'] = train_store_raw['StateHoliday'].replace(state_holidays)


# In[260]:

train_store_raw.head()


# In[261]:

train_store_raw.info()


# ## replace StoreType
#  differentiates between 4 different store models: a, b, c, d

# In[262]:

store_types = {'a':0, 'b':1, 'c':2, 'd':3}
train_store_raw['StoreType_formatted'] = train_store_raw['StoreType'].replace(store_types)


# In[263]:

# replace Assortment
# describes an assortment level: a = basic, b = extra, c = extended


# In[264]:

assortments = {'a':0, 'b':1, 'c':2}
train_store_raw['Assortment_formatted'] = train_store_raw['Assortment'].replace(assortments)


# In[265]:

train_store_raw.head()


# In[266]:

train_store_raw.info()


# ## convert PromoInteval

# In[267]:

train_store_raw['PromoInterval']


# In[268]:

train_store_raw['PromoInterval'].iloc[0] == np.nan


# In[269]:

train_store_raw['PromoInterval'].value_counts().plot(kind='bar')


# In[270]:

train_store_raw['PromoInterval'].iloc[len(train_store_raw['PromoInterval'])-1]


# In[271]:

# replace them into number


# In[272]:

def formatPromoInterval(string):
    intervals={'Jan,Apr,Jul,Oct':1, 'Feb,May,Aug,Nov':2, 'Mar,Jun,Sept,Dec':3}
    if pd.notnull(string):
        months = intervals[string]
    else:
        months = 0
    return months


# In[273]:

formatPromoInterval(train_store_raw['PromoInterval'].iloc[0])


# In[274]:

train_store_raw['PromoInterval_formatted'] = train_store_raw['PromoInterval'].apply(lambda x: formatPromoInterval(x))


# In[275]:

pd.set_option('display.max_columns', None)


# In[276]:

train_store_raw.tail()


# In[277]:

train_ready = train_store_raw[['Store', 'DayOfWeek','Date_formated', 'Sales', 'Customers', 'Open', 'Promo', 'StateHolidy_formatted', 'SchoolHoliday', 'StoreType_formatted', 'Assortment_formatted', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval_formatted']]


# In[278]:

train_ready.tail()


# In[279]:

train_ready.info()


# In[280]:

train_ready['EpochTime'] = train_ready['Date_formated'].astype('int64')


# In[281]:

train_ready.tail()


# In[282]:

train_ready['DayOfYear'] = train_ready['Date_formated'].dt.dayofyear


# In[283]:

train_ready.head()


# In[284]:

df = pd.DataFrame({'year': train_ready['CompetitionOpenSinceYear'],'month': train_ready['CompetitionOpenSinceMonth']})
df['day']=1


# In[285]:

df_1 = pd.to_datetime(df)


# In[286]:

df2 = train_ready['Date_formated'] - df_1


# In[287]:

train_ready['CompetitionOpenSinceDays'] = df2.astype('timedelta64[D]')


# In[288]:

train_ready.tail()


# In[289]:

def calPromo2SinceDate(df):
    r = float('nan')
    if not math.isnan(df['Promo2SinceWeek']) and not math.isnan(df['Promo2SinceYear']):
        d = "{0}-{1}".format(int(df['Promo2SinceYear']), int(df['Promo2SinceWeek']))
        r = datetime.strptime(d + '-0', "%Y-%W-%w")
    return r


# In[290]:

s = train_ready[['Promo2SinceWeek', 'Promo2SinceYear']].apply(calPromo2SinceDate, axis=1)


# In[291]:

df = s.to_frame()


# In[292]:

df = pd.to_datetime(s)


# In[293]:

df2 = train_ready['Date_formated'] - df


# In[294]:

train_ready['Promo2SinceDays'] = df2.astype('timedelta64[D]')


# In[295]:

# train_ready.head(1000)


# In[296]:

avg = np.average(train_ready['CompetitionOpenSinceDays'][train_ready['CompetitionOpenSinceDays'].isnull() == False])


# In[297]:

train_ready['CompetitionOpenSinceDays'].fillna(avg, inplace=True)


# In[298]:

avg = np.average(train_ready['Promo2SinceDays'][train_ready['Promo2SinceDays'].isnull() == False])


# In[299]:

train_ready['Promo2SinceDays'].fillna(avg, inplace=True)


# In[300]:

train_ready.head()


# In[301]:

avg = np.average(train_ready['CompetitionOpenSinceMonth'][train_ready['CompetitionOpenSinceMonth'].isnull() == False])


# In[302]:

train_ready['CompetitionOpenSinceMonth'].fillna(avg, inplace=True)


# In[303]:

avg = np.average(train_ready['CompetitionOpenSinceYear'][train_ready['CompetitionOpenSinceYear'].isnull() == False])


# In[304]:

train_ready['CompetitionOpenSinceYear'].fillna(avg, inplace=True)


# In[305]:

train_ready.head()


# In[306]:

avg = np.average(train_ready['CompetitionDistance'][train_ready['CompetitionDistance'].isnull() == False])


# In[307]:

train_ready['CompetitionDistance'].fillna(avg, inplace=True)


# In[308]:

avg = np.average(train_ready['Promo2SinceWeek'][train_ready['Promo2SinceWeek'].isnull() == False])


# In[309]:

train_ready['Promo2SinceWeek'].fillna(avg, inplace=True)


# In[310]:

avg = np.average(train_ready['Promo2SinceYear'][train_ready['Promo2SinceYear'].isnull() == False])


# In[311]:

train_ready['Promo2SinceYear'].fillna(avg, inplace=True)


# In[312]:

print train_ready['Store'].isnull().values.any()
print train_ready['DayOfWeek'].isnull().values.any()
print train_ready['DayOfYear'].isnull().values.any()
print train_ready['EpochTime'].isnull().values.any()
print train_ready['Customers'].isnull().values.any()
print train_ready['Open'].isnull().values.any()
print train_ready['Promo'].isnull().values.any()
print train_ready['StateHolidy_formatted'].isnull().values.any()
print train_ready['SchoolHoliday'].isnull().values.any()
print train_ready['StoreType_formatted'].isnull().values.any()
print train_ready['Assortment_formatted'].isnull().values.any()
print train_ready['CompetitionDistance'].isnull().values.any()
print train_ready['CompetitionOpenSinceMonth'].isnull().values.any()
print train_ready['CompetitionOpenSinceYear'].isnull().values.any()
print train_ready['Promo2'].isnull().values.any()
print train_ready['Promo2SinceWeek'].isnull().values.any()
print train_ready['Promo2SinceYear'].isnull().values.any()
print train_ready['PromoInterval_formatted'].isnull().values.any()


# # Data Explore

# In[643]:

feature_columns = ['Store', 'DayOfWeek', 'DayOfYear','Date_formated', 'Open', 'Promo', 'StateHolidy_formatted', 'SchoolHoliday', 'StoreType_formatted', 'Assortment_formatted', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear','PromoInterval_formatted','CompetitionOpenSinceDays','Promo2SinceDays']
target_coumn = ['Sales']


# In[120]:

## use randomForestRegressor to determine feature importances
# x = train_ready[feature_columns]
# y = train_ready[target_coumn]

# rfr = RandomForestRegressor()
# rfr.fit(x,y)


# # In[121]:

# print "Features sorted by their score:"
# print sorted(zip(map(lambda x: round(x, 4), rfr.feature_importances_), feature_columns), 
#              reverse=True)


# In[72]:

feature_columns = ['Date_formated','Open', 'CompetitionDistance', 'Store', 'Promo', 'DayOfWeek', 'CompetitionOpenSinceMonth', 'DayOfYear', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceDays', 'StoreType_formatted', 'Assortment_formatted', 'EpochTime', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval_formatted', 'Promo2SinceDays', 'Promo2', 'SchoolHoliday', 'StateHolidy_formatted']
target_coumn = ['Sales']


# In[314]:

# result seems not good, try to use TimeSeries analysis


# In[315]:

train_ready[feature_columns].head()


# In[316]:

train_ready.info()


# In[577]:

train_ready_ts_train = train_ready[['Date_formated', 'Sales']]
train_ready_ts_train.set_index('Date_formated', inplace=True)
train_ready_ts_train = train_ready_ts_train.iloc[:,0]


# In[579]:

type(train_ready_ts_train)


# In[580]:

train_ready_ts_train_g = train_ready_ts_train.groupby(train_ready_ts_train.index).mean()


# In[581]:

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput


# In[322]:

# train_ready_ts_train_g.plot(figsize=(100,30))


# In[582]:

# Predict using ARIMA


# In[583]:

ts_log = np.log(train_ready_ts_train_g)
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.plot()


# In[593]:

ts_log = ts_log.replace([np.inf, -np.inf], np.mean(ts_log))
ts_log.isnull().any()


# # In[602]:

# ts_log_diff = ts_log_diff.replace([np.inf, -np.inf], np.mean(ts_log_diff))
# ts_log_diff.fillna(np.mean(ts_log_diff), inplace=True)
# ts_log_diff.isnull().any()


# # In[603]:

# ts_log.head()


# # In[604]:

# ts_log_diff.head()


# # In[605]:

# test_stationarity(ts_log_diff)


# In[606]:

moving_avg = pd.rolling_mean(ts_log, 7)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')


# In[612]:

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(20)


# In[608]:

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)


# In[350]:

# expwighted_avg = pd.ewma(ts_log, halflife=7)
# plt.plot(ts_log)
# plt.plot(expwighted_avg, color='red')


# # In[351]:

# ts_log_ewma_diff = ts_log - expwighted_avg
# test_stationarity(ts_log_ewma_diff)


# # In[421]:

# decomposition = seasonal_decompose(ts_log.values, freq=30)

# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid

# plt.subplot(411)
# plt.plot(ts_log, label='Original')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(trend, label='Trend')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(seasonal,label='Seasonality')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(residual, label='Residuals')
# plt.legend(loc='best')
# plt.tight_layout()


# # In[445]:

# ts_log_decompose = residual
# ts_log_decompose = pd.DataFrame(ts_log_decompose, index=ts_log.index)
# ts_log_decompose.dropna(inplace=True)
# ts_log_decompose = ts_log_decompose.iloc[:,0]
# test_stationarity(ts_log_decompose)


# # In[446]:

# ts_log_decompose


# In[609]:

lag_acf = acf(ts_log_moving_avg_diff, nlags=10)
lag_pacf = pacf(ts_log_moving_avg_diff, nlags=10, method='ols')


# # In[610]:

# #Plot ACF: 
# plt.subplot(121) 
# plt.plot(lag_acf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_log_moving_avg_diff)),linestyle='--',color='red')
# plt.axhline(y=1.96/np.sqrt(len(ts_log_moving_avg_diff)),linestyle='--',color='green')
# plt.title('Autocorrelation Function')


# # In[611]:

# #Plot PACF:
# plt.subplot(122)
# plt.plot(lag_pacf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_log_moving_avg_diff)),linestyle='--',color='red')
# plt.axhline(y=1.96/np.sqrt(len(ts_log_moving_avg_diff)),linestyle='--',color='green')
# plt.title('Partial Autocorrelation Function')
# plt.tight_layout()


# **from there we could see p in PACF < 1, q in ACF < 1**
# [understand p,q,d](https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/)
# * may need to make the series more stationary

# In[613]:

# Test Predict


# In[618]:

#AR Model
model = ARIMA(ts_log, order=(1, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_moving_avg_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_moving_avg_diff)**2))


# In[524]:

# #MA Model
# model = ARIMA(ts_log, order=(0, 1,1))  
# results_MA = model.fit(disp=-1)  
# plt.plot(ts_log_moving_avg_diff)
# plt.plot(results_MA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_moving_avg_diff)**2))


# # In[532]:

# #ARIMA model
# model = ARIMA(ts_log, order=(1, 1, 1))  
# results_ARIMA = model.fit(disp=-1)  
# plt.plot(ts_log_moving_avg_diff)
# plt.plot(results_ARIMA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_moving_avg_diff)**2))


# *seems the AR approach is pretty good*

# In[619]:

predictions_ARIMA_diff = pd.Series(results_AR.fittedvalues, copy=True)
print predictions_ARIMA_diff.head(20)


# In[621]:

predictions_ARIMA_log = ts_log.add(predictions_ARIMA_diff,fill_value=0)
predictions_ARIMA_log.tail(20)


# In[622]:

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(train_ready_ts_train_g, alpha = 1)
plt.plot(predictions_ARIMA, alpha = 0.3)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-train_ready_ts_train_g)**2)/len(train_ready_ts_train_g)))


# In[623]:

train_ready_ts_train_g.head(30)


# In[624]:

predictions_ARIMA.head(30)


# In[625]:

# y_true = train_ready_ts_train_g
# y_pred = predictions_ARIMA
# mean_squared_error(y_true, y_pred)


# In[626]:

# merge the time series predict with the dataframe


# In[631]:

predictions_ARIMA.describe()


# In[639]:

predictions_ARIMA_df = pd.DataFrame(predictions_ARIMA, columns=['Sales'])


# In[645]:

predictions_ARIMA_df = predictions_ARIMA_df.reset_index()


# In[647]:

train_ready_with_ts = train_ready.merge(predictions_ARIMA_df, on='Date_formated')


# In[649]:

train_ready_with_ts.head()


# In[650]:

train_ready_with_ts['sales_diff'] = train_ready_with_ts['Sales_x'] - train_ready_with_ts['Sales_y']


# In[651]:

train_ready_with_ts.head()


# In[654]:

# plt.plot(train_ready_with_ts['Sales_x'], alpha=.3)
# plt.plot(train_ready_with_ts['Sales_y'], alpha=.7)
# plt.plot(train_ready_with_ts['sales_diff'], alpha=.2)


# In[656]:

train_ready_with_ts.columns


# In[664]:

feature_columns = [u'Open', u'sales_diff', u'Sales_y', u'Promo']
target_coumn = ['Sales_x']


# In[665]:

## use randomForestRegressor to determine feature importances
x = train_ready_with_ts[feature_columns]
y = train_ready_with_ts[target_coumn]

rfr = RandomForestRegressor(n_estimators=100, n_jobs=4)
rfr.fit(x, y)


# In[666]:

print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rfr.feature_importances_), feature_columns), 
             reverse=True)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# # Predict

# In[667]:

test_raw.tail()


# In[542]:

train_raw.head()


# In[543]:

test_store_raw = test_raw.merge(store_raw, on='Store')


# In[544]:

test_store_raw.head()


# In[545]:

test_store_raw['Date_formated'] = pd.to_datetime(test_store_raw['Date'], format='%Y-%m-%d')
test_store_raw['StateHolidy_formatted'] = test_store_raw['StateHoliday'].replace(state_holidays)
test_store_raw['StoreType_formatted'] = test_store_raw['StoreType'].replace(store_types)
test_store_raw['Assortment_formatted'] = test_store_raw['Assortment'].replace(assortments)
test_store_raw['PromoInterval_formatted'] = test_store_raw['PromoInterval'].apply(lambda x: formatPromoInterval(x))
test_ready = test_store_raw[['Store', 'DayOfWeek','Date_formated', 'Open', 'Promo', 'StateHolidy_formatted', 'SchoolHoliday', 'StoreType_formatted', 'Assortment_formatted', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval_formatted']]
test_ready['EpochTime'] = test_ready['Date_formated'].astype('int64')
test_ready['DayOfYear'] = test_ready['Date_formated'].dt.dayofyear
df = pd.DataFrame({'year': test_ready['CompetitionOpenSinceYear'],'month': test_ready['CompetitionOpenSinceMonth']})
df['day']=1
df_1 = pd.to_datetime(df)
df2 = test_ready['Date_formated'] - df_1
test_ready['CompetitionOpenSinceDays'] = df2.astype('timedelta64[D]')
s = test_ready[['Promo2SinceWeek', 'Promo2SinceYear']].apply(calPromo2SinceDate, axis=1)
df = s.to_frame()
df = pd.to_datetime(s)
df2 = test_ready['Date_formated'] - df
test_ready['Promo2SinceDays'] = df2.astype('timedelta64[D]')
avg = np.average(test_ready['CompetitionOpenSinceDays'][test_ready['CompetitionOpenSinceDays'].isnull() == False])
test_ready['CompetitionOpenSinceDays'].fillna(avg, inplace=True)
avg = np.average(test_ready['Promo2SinceDays'][test_ready['Promo2SinceDays'].isnull() == False])
test_ready['Promo2SinceDays'].fillna(avg, inplace=True)
avg = np.average(test_ready['CompetitionOpenSinceMonth'][test_ready['CompetitionOpenSinceMonth'].isnull() == False])
test_ready['CompetitionOpenSinceMonth'].fillna(avg, inplace=True)
avg = np.average(test_ready['CompetitionOpenSinceYear'][test_ready['CompetitionOpenSinceYear'].isnull() == False])
test_ready['CompetitionOpenSinceYear'].fillna(avg, inplace=True)
avg = np.average(test_ready['CompetitionDistance'][test_ready['CompetitionDistance'].isnull() == False])
test_ready['CompetitionDistance'].fillna(avg, inplace=True)
avg = np.average(test_ready['Promo2SinceWeek'][test_ready['Promo2SinceWeek'].isnull() == False])
test_ready['Promo2SinceWeek'].fillna(avg, inplace=True)
avg = np.average(test_ready['Promo2SinceYear'][test_ready['Promo2SinceYear'].isnull() == False])
test_ready['Promo2SinceYear'].fillna(avg, inplace=True)

print ('Store', test_ready['Store'].isnull().values.any())
print ('DayOfWeek',test_ready['DayOfWeek'].isnull().values.any())
print ('DayOfYear',test_ready['DayOfYear'].isnull().values.any())
print ('EpochTime',test_ready['EpochTime'].isnull().values.any())
# print test_ready['Customers'].isnull().values.any()
print ('Open',test_ready['Open'].isnull().values.any())
print ('Promo',test_ready['Promo'].isnull().values.any())
print ('StateHolidy_formatted',test_ready['StateHolidy_formatted'].isnull().values.any())
print ('SchoolHoliday',test_ready['SchoolHoliday'].isnull().values.any())
print ('StoreType_formatted',test_ready['StoreType_formatted'].isnull().values.any())
print ('Assortment_formatted',test_ready['Assortment_formatted'].isnull().values.any())
print ('CompetitionDistance',test_ready['CompetitionDistance'].isnull().values.any())
print ('CompetitionOpenSinceMonth',test_ready['CompetitionOpenSinceMonth'].isnull().values.any())
print ('CompetitionOpenSinceYear',test_ready['CompetitionOpenSinceYear'].isnull().values.any())
print ('Promo2',test_ready['Promo2'].isnull().values.any())
print ('Promo2SinceWeek',test_ready['Promo2SinceWeek'].isnull().values.any())
print ('Promo2SinceYear',test_ready['Promo2SinceYear'].isnull().values.any())
print ('PromoInterval_formatted',test_ready['PromoInterval_formatted'].isnull().values.any())


# In[671]:

predictions_ARIMA_df.head()


# In[672]:

test_ready.head()


# In[677]:

test_ready_with_ts = test_ready
test_ready_with_ts['Date_formated'] = test_ready_with_ts['Date_formated'] - timedelta(days=365*2)


# In[678]:

test_ready_with_ts.head()


# In[682]:

test_ready_with_ts = test_ready_with_ts.merge(train_ready_with_ts[['Date_formated','Sales_y', 'Sales_x', 'sales_diff']], on='Date_formated')


# In[ ]:




# In[683]:

test_ready_with_ts[test_ready_with_ts['Open'].isnull()]


# In[684]:

test_ready_with_ts['Open'].isnull().any()


# In[687]:

feature_columns = [u'Open', u'sales_diff', u'Sales_y', u'Promo']
target_coumn = ['Sales_x']


# In[ ]:




# In[ ]:

pred = rfr.predict(test_ready_with_ts[feature_columns])


# In[ ]:

df = pd.DataFrame(pred, columns=['Sales'])


# In[124]:

df['Id'] = range(1, len(df)+1)


# In[129]:

df = df[['Id','Sales']]


# In[130]:

df.to_csv('pred.csv', index=False)


# In[ ]:



