
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

# In[1]:


import numba

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import math
from datetime import datetime

from pandas.tools.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA


# In[2]:

plt.rcParams['agg.path.chunksize'] = 100000


# In[3]:

store_raw = pd.read_csv('store.csv')
test_raw = pd.read_csv('test.csv')
train_raw = pd.read_csv('train.csv')


# In[4]:

store_raw.head(20)


# In[5]:

store_raw.info()


# In[6]:

train_raw.head(20)


# In[7]:

train_raw.tail()


# In[8]:

train_raw[train_raw['Store']==1]


# In[9]:

train_raw.info()


# # Data Wrangling

# ## merge two tables

# In[10]:

train_store_raw = train_raw.merge(store_raw, on='Store')


# In[11]:

train_store_raw.head(20)


# In[12]:

train_store_raw.info()


# ## change column types

# In[13]:

train_store_raw['Date_formated'] = pd.to_datetime(train_store_raw['Date'], format='%Y-%m-%d')


# In[14]:

train_store_raw.head()


# In[15]:

train_store_raw.info()


# ## convert StateHolidays
# a = public holiday, b = Easter holiday, c = Christmas, 0 = None

# In[16]:

state_holidays = {'a':1, 'b':2, 'c':3, '0':0}


# In[17]:

train_store_raw['StateHolidy_formatted'] = train_store_raw['StateHoliday'].replace(state_holidays)


# In[18]:

train_store_raw.head()


# In[19]:

train_store_raw.info()


# ## replace StoreType
#  differentiates between 4 different store models: a, b, c, d

# In[20]:

store_types = {'a':0, 'b':1, 'c':2, 'd':3}
train_store_raw['StoreType_formatted'] = train_store_raw['StoreType'].replace(store_types)


# In[21]:

# replace Assortment
# describes an assortment level: a = basic, b = extra, c = extended


# In[22]:

assortments = {'a':0, 'b':1, 'c':2}
train_store_raw['Assortment_formatted'] = train_store_raw['Assortment'].replace(assortments)


# In[23]:

train_store_raw.head()


# In[24]:

train_store_raw.info()


# ## convert PromoInteval

# In[25]:

train_store_raw['PromoInterval']


# In[26]:

train_store_raw['PromoInterval'].iloc[0] == np.nan


# In[27]:

train_store_raw['PromoInterval'].value_counts().plot(kind='bar')


# In[28]:

train_store_raw['PromoInterval'].iloc[len(train_store_raw['PromoInterval'])-1]


# In[29]:

# replace them into number


# In[30]:

def formatPromoInterval(string):
    intervals={'Jan,Apr,Jul,Oct':1, 'Feb,May,Aug,Nov':2, 'Mar,Jun,Sept,Dec':3}
    if pd.notnull(string):
        months = intervals[string]
    else:
        months = 0
    return months


# In[31]:

formatPromoInterval(train_store_raw['PromoInterval'].iloc[0])


# In[32]:

train_store_raw['PromoInterval_formatted'] = train_store_raw['PromoInterval'].apply(lambda x: formatPromoInterval(x))


# In[33]:

pd.set_option('display.max_columns', None)


# In[34]:

train_store_raw.tail()


# In[35]:

train_ready = train_store_raw[['Store', 'DayOfWeek','Date_formated', 'Sales', 'Customers', 'Open', 'Promo', 'StateHolidy_formatted', 'SchoolHoliday', 'StoreType_formatted', 'Assortment_formatted', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval_formatted']]


# In[36]:

train_ready.tail()


# In[37]:

train_ready.info()


# In[38]:

train_ready['EpochTime'] = train_ready['Date_formated'].astype('int64')


# In[39]:

train_ready.tail()


# In[40]:

train_ready['DayOfYear'] = train_ready['Date_formated'].dt.dayofyear


# In[41]:

train_ready.head()


# In[42]:

df = pd.DataFrame({'year': train_ready['CompetitionOpenSinceYear'],'month': train_ready['CompetitionOpenSinceMonth']})
df['day']=1


# In[43]:

df_1 = pd.to_datetime(df)


# In[44]:

df2 = train_ready['Date_formated'] - df_1


# In[45]:

train_ready['CompetitionOpenSinceDays'] = df2.astype('timedelta64[D]')


# In[46]:

train_ready.tail()


# In[47]:

def calPromo2SinceDate(df):
    r = float('nan')
    if not math.isnan(df['Promo2SinceWeek']) and not math.isnan(df['Promo2SinceYear']):
        d = "{0}-{1}".format(int(df['Promo2SinceYear']), int(df['Promo2SinceWeek']))
        r = datetime.strptime(d + '-0', "%Y-%W-%w")
    return r


# In[48]:

s = train_ready[['Promo2SinceWeek', 'Promo2SinceYear']].apply(calPromo2SinceDate, axis=1)


# In[49]:

df = s.to_frame()


# In[50]:

df = pd.to_datetime(s)


# In[51]:

df2 = train_ready['Date_formated'] - df


# In[52]:

train_ready['Promo2SinceDays'] = df2.astype('timedelta64[D]')


# In[53]:

train_ready.head(1000)


# In[54]:

avg = np.average(train_ready['CompetitionOpenSinceDays'][train_ready['CompetitionOpenSinceDays'].isnull() == False])


# In[55]:

train_ready['CompetitionOpenSinceDays'].fillna(avg, inplace=True)


# In[56]:

avg = np.average(train_ready['Promo2SinceDays'][train_ready['Promo2SinceDays'].isnull() == False])


# In[57]:

train_ready['Promo2SinceDays'].fillna(avg, inplace=True)


# In[58]:

train_ready.head()


# In[59]:

avg = np.average(train_ready['CompetitionOpenSinceMonth'][train_ready['CompetitionOpenSinceMonth'].isnull() == False])


# In[60]:

train_ready['CompetitionOpenSinceMonth'].fillna(avg, inplace=True)


# In[61]:

avg = np.average(train_ready['CompetitionOpenSinceYear'][train_ready['CompetitionOpenSinceYear'].isnull() == False])


# In[62]:

train_ready['CompetitionOpenSinceYear'].fillna(avg, inplace=True)


# In[63]:

train_ready.head()


# In[64]:

avg = np.average(train_ready['CompetitionDistance'][train_ready['CompetitionDistance'].isnull() == False])


# In[65]:

train_ready['CompetitionDistance'].fillna(avg, inplace=True)


# In[66]:

avg = np.average(train_ready['Promo2SinceWeek'][train_ready['Promo2SinceWeek'].isnull() == False])


# In[67]:

train_ready['Promo2SinceWeek'].fillna(avg, inplace=True)


# In[68]:

avg = np.average(train_ready['Promo2SinceYear'][train_ready['Promo2SinceYear'].isnull() == False])


# In[69]:

train_ready['Promo2SinceYear'].fillna(avg, inplace=True)


# In[70]:

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

# In[85]:

# scatter = scatter_matrix(train_ready, figsize=(60,60))


# * from the scatter matrix we see sales is highly related to: customers, competitionInDistance and competitionOpenInYear, let's focus on these three
# * or we could run a random forest to tell what are the important features

# In[71]:

feature_columns = ['Store', 'DayOfWeek', 'DayOfYear','EpochTime', 'Open', 'Promo', 'StateHolidy_formatted', 'SchoolHoliday', 'StoreType_formatted', 'Assortment_formatted', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear','PromoInterval_formatted','CompetitionOpenSinceDays','Promo2SinceDays']
target_coumn = ['Sales']


# In[120]:

## use randomForestRegressor to determine feature importances
# x = train_ready[feature_columns]
# y = train_ready[target_coumn]

# rfr = RandomForestRegressor()
# rfr.fit(x,y)


# In[121]:

# print "Features sorted by their score:"
# print sorted(zip(map(lambda x: round(x, 4), rfr.feature_importances_), feature_columns), 
#              reverse=True)


# In[130]:

feature_columns = ['Date_formated','Open', 'CompetitionDistance', 'Store', 'Promo', 'DayOfWeek', 'CompetitionOpenSinceMonth', 'DayOfYear', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceDays', 'StoreType_formatted', 'Assortment_formatted', 'EpochTime', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval_formatted', 'Promo2SinceDays', 'Promo2', 'SchoolHoliday', 'StateHolidy_formatted']
target_coumn = ['Sales']


# In[123]:

# x = train_ready[feature_columns]
# y = train_ready[target_coumn]

# rfr = RandomForestRegressor()
# rfr.fit(x,y)


# In[72]:

# result seems not good, try to use TimeSeries analysis


# In[73]:

train_ready[feature_columns].head()


# In[74]:

train_ready.info()


# In[75]:

train_ready_ts = train_ready[['Date_formated', 'Sales']]
train_ready_ts.set_index('Date_formated', inplace=True)
train_ready_ts


# In[76]:

print(train_ready_ts.info())


# In[77]:

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


# In[169]:

# train_ready_ts.plot(figsize=(300,30))


# In[78]:

# Predict using ARIMA


# In[80]:

ts_log = np.log(train_ready_ts)
ts_log_diff = ts_log - ts_log.shift()
# ts_log_diff.plot()


# In[81]:

ts_log = ts_log.replace([np.inf, -np.inf], np.nan)
ts_log.dropna(inplace=True)


# In[82]:

ts_log_diff = ts_log_diff.replace([np.inf, -np.inf], np.nan)


# In[83]:

ts_log_diff.dropna(inplace=True)


# In[193]:

# test_stationarity(ts_log_diff)


# In[ ]:

lag_acf = acf(ts_log_diff, nlags=10)
lag_pacf = pacf(ts_log_diff, nlags=10, method='ols')


# In[203]:

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='red')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='green')
plt.title('Autocorrelation Function')
print('ploting acf.....')
plt.savefig('Plot_acf.png')

# In[202]:

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='red')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='green')
plt.title('Partial Autocorrelation Function')
print('ploting pacf.....')
plt.tight_layout()
plt.savefig('Plot_pcaf.png')


# **from there we could see p in PACF is 1, q in ACF is 1**
# [understand p,q,d](https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/)

# In[84]:

# Test Predict


# In[ ]:
print('Building models.....')
#AR Model
model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
plt.savefig('AR.png')

# In[ ]:

#MA Model
model = ARIMA(ts_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))
plt.savefig('MA.png')

# In[ ]:

#ARIMA model
model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
plt.savefig('Combined.png')