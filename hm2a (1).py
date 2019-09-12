#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas_datareader.data as web
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn import preprocessing
start = datetime.datetime(2010,1,1)
end = datetime.datetime(2018,1,1)
df = web.DataReader("IBM","yahoo",start,end)
df.head()




# In[2]:


dfreg = df.loc[:,['Adj Close','Volume']]       # extract only specified 2 columns of data
dfreg['HL_PCT'] = (df['High']-df['Low'])/df['Close']*100.0       # add hi/lo %'g as feature
dfreg['PCT_change'] = (df['Close']-df['Open'])/df['Open']*100.0  # add %'g chg as feature
dfreg.head()


# In[3]:


#help(df.loc)


# In[4]:


dfreg.fillna(value=-9999,inplace=True)
dfreg.head()


# In[5]:


#help(dfreg.drop)




# In[6]:


train_siz = int(math.ceil(0.20*len(dfreg)))  # 20% of data assigned to training set
#print (train_siz)                            #  math.ceil just rounds up to nxt integer
forecast_col = 'Adj Close'
#dfreg.headfreg.head()d()
dfreg['future_price'] = dfreg[forecast_col].shift(-train_siz)
dfreg.head()
#print (dfreg.shape)
#print (dfreg[0:10])
x = np.array(dfreg.drop(['future_price'],1))
#print (x.shape)
#print (x[0:10])
x = preprocessing.scale(x)              # re-scale all data so that stdev is 1
#print (x[0:10])
#y = x[:,0]
#print (y.shape)
#print (y[0:10])
#print (len(x))
#print (x[0:10])


# In[7]:


#help(dfreg.shift)t_s


# In[8]:


#print len(df)
x_valid = x[train_siz:len(df)]
x_train = x[0:train_siz-1]
#print (len(x_train))
#print (len(x_valid))
#print  x[0:10]
#y = np.array(dfreg.drop(['future_price',1))
#dfreg.fillna(value=-9999,inply = np.array(dfreg['future_price']ace=True)
y = np.array(dfreg['future_price'])
#y = np.array(dfreg['future_price'].fillna(value=-9999,inplace=True))
y_valid = y[train_siz:len(df)]
y_train = y[0:train_siz-1]
#print (len(y_train))
#print (len(y_valid))
#print (y[0:10])
#print ("y_valid")
#print (y_valid[0:100])
#print ("y_train")
#print (y_train[0:100])
#print ("y")
#print (y[2000:2014])


# In[ ]:





# In[9]:


# Linear Regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(x_train, y_train)

#Lasso Regression
clflasso = linear_model.Lasso(alpha=0.1)
clflasso.fit(x_train,y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(x_train,y_train)


# In[10]:


clfreg_forecast_set = clfreg.predict(x_valid)
clflasso_forecast_set = clflasso.predict(x_valid)
clfknn_forecast_set = clfknn.predict(x_valid)
print clfreg_forecast_set
print clflasso_forecast_set
print clfknn_forecast_set


# In[11]:


x_axis = list(range(len(clfreg_forecast_set)))
y_prediction = clfreg_forecast_set
print (y_prediction)
print (y_valid)
print y_valid.shape
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=100
fig_size[1]=100
plt.rcParams["figure.figsize"] = fig_size
plt.plot(x_axis,map(float, y_valid),'r--',x_axis,y_prediction,'g')


# In[12]:


x_axis = list(range(len(clflasso_forecast_set)))
y_prediction = clflasso_forecast_set
print (y_prediction)
print (y_valid)
print y_valid.shape
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=100
fig_size[1]=100
plt.rcParams["figure.figsize"] = fig_size
plt.plot(x_axis,map(float, y_valid),'r--',x_axis,y_prediction,'g')


# In[13]:


x_axis = list(range(len(clfknn_forecast_set)))
y_prediction = clfknn_forecast_set
print (y_prediction)
print (y_valid)
print y_valid.shape
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=100
fig_size[1]=100
plt.rcParams["figure.figsize"] = fig_size
plt.plot(x_axis,map(float, y_valid),'r--',x_axis,y_prediction,'g')


# In[14]:


clfreg_forecast_set


# In[15]:


#clfreg.score(x_valid,y_valid)


# In[16]:


clflasso_forecast_set


# In[17]:


#clflasso.score(x_valid,y_valid)


# In[18]:


clfknn_forecast_set


# In[19]:


#clfknn.score(x_valid,y_valid)


# In[ ]:
