#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## source code referenced from :
## https://github.com/kprakhar27/Financial-Trading-in-Python/blob/main/Technical%20Indicators.ipynb


# In[2]:


import talib


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df=pd.read_csv('EUR_STOCK_DATA.csv')


# In[5]:


df.head(10)


# In[6]:


## selecting subset of the above dataframe for clear plots and to observe clear trends
df1=df.iloc[:240,:]


# In[7]:


df1.head(10)


# In[8]:


# Calculate 20-day EMA
df1['EMA_20'] = talib.EMA(df1['Close'], timeperiod=20)
# Calculate 40-day EMA
df1['EMA_40'] = talib.EMA(df1['Close'], timeperiod=40)

plt.figure(figsize=(16, 8))

# Plot the EMAs with price
plt.plot(df1['EMA_20'].iloc[40:], label='EMA_20')
plt.plot(df1['EMA_40'], label='EMA_40')
plt.plot(df1['Close'].iloc[40:], label='Close')

# Customize and show the plot
plt.legend()
plt.title('EMAs')
plt.show()


# In[9]:


# Calculate the SMA
df1['SMA'] = talib.SMA(df1['Close'], timeperiod=40)
# Calculate the EMA
df1['EMA'] = talib.EMA(df1['Close'], timeperiod=40)

plt.figure(figsize=(16, 8))

# Plot the SMA, EMA with price
plt.plot(df1['SMA'], label='SMA')
plt.plot(df1['EMA'], label='EMA')
plt.plot(df1['Close'].iloc[40:], label='Close')

# Customize and show the plot
plt.legend()
plt.title('SMA vs EMA')
plt.show()


# In[10]:


# Calculate the ADX with the time period set to 21
df1['ADX_40'] = talib.ADX(df1['High'], 
                           df1['Low'],
                            df1['Close'], timeperiod = 40)

# Print the last five rows
print(df1.head(25))


# In[11]:


# Create subplots
fig, (plt1, plt2) = plt.subplots(2, figsize=(15,15))

# Plot ADX with the price
plt1.set_ylabel('Price')
plt1.plot(df1['Close'][80:])
plt2.set_ylabel('ADX')
plt2.plot(df1['ADX_40'], color='red')

plt1.set_title('Price and ADX')
plt.show()


# In[12]:


# Calculate RSI with a time period of 21
df1['RSI_20'] = talib.RSI(df1['Close'], timeperiod = 20)

# Print the last five rows
print(df1.head(5))


# In[13]:


fig, (plt1, plt2) = plt.subplots(2, figsize=(15,15))
# Plot RSI with the price
plt1.set_ylabel('Price')
plt1.plot(df1['Close'][20:])
plt2.set_ylabel('RSI')
plt2.plot(df1['RSI_20'], color='orangered')

plt1.set_title('Price and RSI')
plt.show()


# In[14]:


# Define the Bollinger Bands
up_band, mid_band, low_band = talib.BBANDS(df1['Close'],
                                 nbdevup=2,
                                 nbdevdn=2,
                                 timeperiod=20)

plt.figure(figsize=(16, 8))

# Plot the Bollinger Bands 
plt.plot(df1['Close'][20:], label='Price')
plt.plot(up_band, color='Red' , label='Upper band')
plt.plot(mid_band, color='Green', label='Middle band')
plt.plot(low_band, color='Red', label='Lower band')

# Customize and show the plot
plt.title('Bollinger Bands')
plt.legend()
plt.show()


# In[15]:


label=[]
for i in range((len(df.index)-1)):
  if (df.Close[i+1]) > (df.Close[i]):
    label.append(1)
  else:
    label.append(0)

#dropping the last row of the dataset
df_new=df.drop(12889, axis=0)


# In[16]:


df_new['label']=label
df_new.drop(['Gmt time','Close'], axis=1, inplace=True)
df_new.head(10)


# In[19]:


df_new['EMA_20'] = talib.EMA(df['Close'][:-1], timeperiod=20)
df_new['SMA_20'] = talib.SMA(df['Close'][:-1], timeperiod=20)
df_new['RSI_20'] = talib.RSI(df['Close'][:-1], timeperiod=20)
# df_new['CMA_20'] = talib.CMA(df['Close'][:-1], timeperiod=20)


# In[20]:


df_new.head(5)


# In[23]:


up_band, mid_band, low_band = talib.BBANDS(df['Close'][:-1],
                                 nbdevup=2,
                                 nbdevdn=2,
                                 timeperiod=20)
up_band


# In[24]:


df_new['up_band_20']=up_band
df_new['low_band_20']=low_band


# In[25]:


df_new.head(10)


# In[26]:


df_new.head(50)


# In[27]:


df_new.to_csv('new_Euro_Data.csv',index=False)


# In[ ]:




