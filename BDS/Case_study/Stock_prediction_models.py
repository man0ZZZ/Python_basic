#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


##knitting the data frame
start_date=str(input('please enter the start date: '))
end_date=str(input('please eneter the end date: '))
start=start_date+' 00:00:00.000'
end=end_date+' 20:00:00.000'
for i in range(len(df.index)):
    if df['Gmt time'][i]==start:
        s=i
    elif df['Gmt time'][i]==end:
        c=i
try:
    print('start date',df['Gmt time'][s])
    print('end date', df['Gmt time'][c])
    df1=df.iloc[(s):(c+1),:]
except:
    print('please enter valid date')


# In[7]:


df1.reset_index(drop=True, inplace=True)
df1.head(10)


# In[8]:


print(len(df1.index))


# In[9]:


label=[]
for i in range(len(df1.index)):
    if i==(len(df1.index)-1):
        label.append(np.nan)
    else:
        if (df1.Close[i+1]) > (df1.Close[i]):
            label.append(1)
        else:
            label.append(0)


# In[10]:


len(label)


# In[11]:


df1['label']=label
df1=df1.drop(['Gmt time'], axis=1)
df1.tail(10)


# In[12]:


df1=df1.drop(df1.index[-1], axis=0)


# In[13]:


df1['EMA_20'] = talib.EMA(df1['Close'], timeperiod=20)
df1['SMA_20'] = talib.SMA(df1['Close'], timeperiod=20)
df1['RSI_20'] = talib.RSI(df1['Close'], timeperiod=20)
df1['EMA_30'] = talib.EMA(df1['Close'], timeperiod=30)
df1['SMA_30'] = talib.SMA(df1['Close'], timeperiod=30)
df1['RSI_30'] = talib.RSI(df1['Close'], timeperiod=30)
df1['EMA_40'] = talib.EMA(df1['Close'], timeperiod=40)
df1['SMA_40'] = talib.SMA(df1['Close'], timeperiod=40)
df1['RSI_40'] = talib.RSI(df1['Close'], timeperiod=40)

# df_new['CMA_20'] = talib.CMA(df['Close'][:-1], timeperiod=20)


# In[14]:


df1.head(5)


# In[21]:


len(df1.Close)


# In[18]:


up_band20, mid_band20, low_band20 = talib.BBANDS(df1['Close'],
                                 nbdevup=2,
                                 nbdevdn=2,
                                 timeperiod=20)
up_band30, mid_band30, low_band30 = talib.BBANDS(df1['Close'],
                                 nbdevup=2,
                                 nbdevdn=2,
                                 timeperiod=30)
up_band40, mid_band40, low_band40 = talib.BBANDS(df1['Close'],
                                 nbdevup=2,
                                 nbdevdn=2,
                                 timeperiod=40)


# In[25]:


len(up_band40)


# In[26]:


df1['up_band_20']=up_band20


# In[27]:


df1['low_band_20']=low_band20
df1['up_band_30']=up_band30
df1['low_band_20']=low_band30
df1['up_band_40']=up_band40
df1['low_band_40']=low_band40


# In[29]:


df1.head(50)


# In[30]:


df1=df1.drop(['Close'], axis=1)


# In[35]:


##dropping rows with nan values till row 39..
df1=df1.iloc[40:,:]


# In[36]:


y=df1.label.values
X=df1.drop(['label'], axis=1).values


# In[37]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=1, train_size=0.75)


# In[38]:


##applying random forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=50, random_state=2)
rf_model.fit(X_train, y_train)


# In[39]:


y_pred=rf_model.predict(X_test)


# In[45]:


from sklearn.metrics import f1_score,accuracy_score,precision_score,confusion_matrix,recall_score
print(f1_score(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))


# In[41]:


from sklearn.naive_bayes import GaussianNB
nba=GaussianNB()
nba.fit(X_train,y_train)


# In[42]:


y_nb_pred=nba.predict(X_test)


# In[46]:


print(f1_score(y_test,y_nb_pred))
print(accuracy_score(y_test,y_nb_pred))
print(precision_score(y_test,y_nb_pred))
print(recall_score(y_test,y_nb_pred))
print(confusion_matrix(y_test, y_nb_pred))


# In[63]:


#applying SVM
#standadizing the values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
Xs=sc.fit_transform(X)
ys=y
#ys=sc.fit_transform(y.reshape(-1,1))## do not need to scale the y label,, we are going to use classification model


# In[64]:


Xs_train, Xs_test, ys_train, ys_test=train_test_split(Xs, ys, random_state=2, train_size=0.75)


# In[65]:


#applying the model
from sklearn.svm import SVC
svm_model = SVC(kernel = 'rbf')
svm_model.fit(Xs_train, ys_train.ravel())


# In[66]:


y_svm_pred=svm_model.predict(Xs_test)


# In[67]:


y_svm_pred[:10]


# In[69]:


print(f1_score(y_test,y_svm_pred))
print(accuracy_score(y_test,y_svm_pred))
print(precision_score(y_test,y_svm_pred))
print(recall_score(y_test,y_svm_pred))
print(confusion_matrix(y_test, y_svm_pred))


# In[ ]:




