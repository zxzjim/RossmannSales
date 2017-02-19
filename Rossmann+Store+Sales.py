
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

# In[52]:

get_ipython().magic(u'matplotlib inline')
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
from pandas.tools.plotting import autocorrelation_plot
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm  
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

import gc


# In[53]:

pd.set_option('display.max_columns', None)


# In[54]:

plt.rcParams['agg.path.chunksize'] = 100000


# In[55]:

store_raw = pd.read_csv('store.csv')
test_raw = pd.read_csv('test.csv')
train_raw = pd.read_csv('train.csv')


# In[56]:

store_raw.head()


# In[57]:

store_raw.info()


# In[58]:

train_raw.head()


# In[59]:

test_raw.head()


# In[60]:

train_raw.tail()


# In[61]:

test_raw.tail()


# In[62]:

train_raw.info()


# In[63]:

test_raw.info()


# # Data Wrangling

# ## merge two tables

# In[64]:

train_store_raw = train_raw.merge(store_raw, on='Store', how='left')
test_store_raw = test_raw.merge(store_raw, on='Store',how='left')


# ## change column types

# In[65]:

train_store_raw['Date_formatted'] = pd.to_datetime(train_store_raw['Date'], format='%Y-%m-%d')
test_store_raw['Date_formatted'] = pd.to_datetime(test_store_raw['Date'], format='%Y-%m-%d')


# ## convert StateHolidays
# a = public holiday, b = Easter holiday, c = Christmas, 0 = None

# In[66]:

state_holidays = {'a':1, 'b':2, 'c':3, '0':0}

train_store_raw['StateHolidy_formatted'] = train_store_raw['StateHoliday'].replace(state_holidays)
test_store_raw['StateHolidy_formatted'] = test_store_raw['StateHoliday'].replace(state_holidays)


# ## replace StoreType
#  differentiates between 4 different store models: a, b, c, d

# In[67]:

store_types = {'a':0, 'b':1, 'c':2, 'd':3}
train_store_raw['StoreType_formatted'] = train_store_raw['StoreType'].replace(store_types)
test_store_raw['StoreType_formatted'] = test_store_raw['StoreType'].replace(store_types)

# replace Assortment
# describes an assortment level: a = basic, b = extra, c = extended

assortments = {'a':0, 'b':1, 'c':2}
train_store_raw['Assortment_formatted'] = train_store_raw['Assortment'].replace(assortments)
test_store_raw['Assortment_formatted'] = test_store_raw['Assortment'].replace(assortments)


# ## convert PromoInteval

# In[68]:

def formatPromoInterval(string):
    intervals={'Jan,Apr,Jul,Oct':1, 'Feb,May,Aug,Nov':2, 'Mar,Jun,Sept,Dec':3}
    if pd.notnull(string):
        months = intervals[string]
    else:
        months = 0
    return months


# In[69]:

train_store_raw['PromoInterval_formatted'] = train_store_raw['PromoInterval'].apply(lambda x: formatPromoInterval(x))
test_store_raw['PromoInterval_formatted'] = test_store_raw['PromoInterval'].apply(lambda x: formatPromoInterval(x))


# In[70]:

train_store_raw.tail()


# In[71]:

test_store_raw.tail()


# In[72]:

train_ready = train_store_raw[['Store', 'DayOfWeek','Date_formatted', 'Sales', 'Customers', 'Open', 'Promo', 'StateHolidy_formatted', 'SchoolHoliday', 'StoreType_formatted', 'Assortment_formatted', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval_formatted']]
test_ready = test_store_raw[['Id','Store', 'DayOfWeek','Date_formatted', 'Open', 'Promo', 'StateHolidy_formatted', 'SchoolHoliday', 'StoreType_formatted', 'Assortment_formatted', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval_formatted']]


# In[73]:

avg = np.average(train_ready['CompetitionOpenSinceMonth'][train_ready['CompetitionOpenSinceMonth'].isnull() == False])

train_ready['CompetitionOpenSinceMonth'].fillna(avg, inplace=True)

avg = np.average(train_ready['CompetitionOpenSinceYear'][train_ready['CompetitionOpenSinceYear'].isnull() == False])

train_ready['CompetitionOpenSinceYear'].fillna(avg, inplace=True)

avg = np.average(train_ready['CompetitionDistance'][train_ready['CompetitionDistance'].isnull() == False])

train_ready['CompetitionDistance'].fillna(avg, inplace=True)

avg = np.average(train_ready['Promo2SinceWeek'][train_ready['Promo2SinceWeek'].isnull() == False])

train_ready['Promo2SinceWeek'].fillna(avg, inplace=True)

avg = np.average(train_ready['Promo2SinceYear'][train_ready['Promo2SinceYear'].isnull() == False])

train_ready['Promo2SinceYear'].fillna(avg, inplace=True)

print train_ready['Store'].isnull().values.any()
print train_ready['DayOfWeek'].isnull().values.any()
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


# In[74]:

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

print test_ready['Store'].isnull().values.any()
print test_ready['DayOfWeek'].isnull().values.any()
print test_ready['Open'].isnull().values.any()
print test_ready['Promo'].isnull().values.any()
print test_ready['StateHolidy_formatted'].isnull().values.any()
print test_ready['SchoolHoliday'].isnull().values.any()
print test_ready['StoreType_formatted'].isnull().values.any()
print test_ready['Assortment_formatted'].isnull().values.any()
print test_ready['CompetitionDistance'].isnull().values.any()
print test_ready['CompetitionOpenSinceMonth'].isnull().values.any()
print test_ready['CompetitionOpenSinceYear'].isnull().values.any()
print test_ready['Promo2'].isnull().values.any()
print test_ready['Promo2SinceWeek'].isnull().values.any()
print test_ready['Promo2SinceYear'].isnull().values.any()
print test_ready['PromoInterval_formatted'].isnull().values.any()


# In[75]:

train_ready.head()


# In[76]:

train_ready.info()


# In[78]:

test_ready.head()


# In[79]:

test_ready.info()


# # Data Explore

# In[81]:

train_ready.columns


# In[82]:

feature_columns = [u'Store', u'DayOfWeek', u'Customers',
       u'Open', u'Promo', u'StateHolidy_formatted', u'SchoolHoliday',
       u'StoreType_formatted', u'Assortment_formatted', u'CompetitionDistance',
       u'CompetitionOpenSinceMonth', u'CompetitionOpenSinceYear', u'Promo2',
       u'Promo2SinceWeek', u'Promo2SinceYear', u'PromoInterval_formatted',
       u'EpochTime', u'DayOfYear', u'CompetitionOpenSinceDays',
       u'Promo2SinceDays']
target_coumn = ['Sales']


# In[83]:

train_ready.head()


# In[84]:

len(train_ready['Store'].unique())


# In[91]:

train_ready_g = train_ready.groupby(['Store'])


# In[92]:

len(train_ready_g.groups)


# In[643]:

g1 = train_ready_g.get_group(10).sort('Date_formatted')


# In[644]:

# let's run it store by store


# In[ ]:




# # Predict

# In[80]:

test_ready_g = test_ready.groupby('Store')


# In[81]:

test_ready_g.get_group(1)


# In[82]:

test_ready.info()


# In[343]:

pred1 = rfr.predict(test_ready_with_ts1[feature_columns])


# In[344]:

df1 = pd.DataFrame(pred1, columns=['Sales'])


# In[345]:

df1['Id'] = range(1, len(df1)+1)


# In[346]:

df1 = df1[['Id','Sales']]


# In[347]:

df1.columns


# In[348]:

df1.to_csv('pred1.csv', index=False)


# In[ ]:



