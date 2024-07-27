#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime


# In[7]:


dataset=pd.read_csv(r"C:\Users\mohan\OneDrive\Desktop\Google_Stock_price_Train.csv",index_col="Date",parse_dates=True)
dataset.head()


# In[8]:


dataset.isna().any()


# In[9]:


dataset["Open"].plot(figsize=(16,6))


# In[11]:


dataset["Close"]=dataset["Close"].str.replace(',','').astype(float)
dataset["Volume"]=dataset["Volume"].str.replace(',','').astype(float)


# In[12]:


dataset.info()


# In[13]:


datast=pd.read_csv(r"C:\Users\mohan\Downloads\Google_Stock_price_Train.csv",index_col="Date",parse_dates=True)
datast.head()


# In[14]:


dataset.isna().any()


# In[15]:


dataset["Open"].plot(figsize=(16,6))


# In[16]:


datast["Open"].plot(figsize=(16,6))


# In[17]:


datast["Open"].plot(figsize=(16,6))


# In[18]:


datast.info()


# In[19]:


dataset["Close"]=dataset["Close"].str.replace(',','').astype(float)
dataset["Volume"]=dataset["Volume"].str.replace(',','').astype(float)


# In[20]:


datast["Close"]=dataset["Close"].str.replace(',','').astype(float)
datast["Volume"]=dataset["Volume"].str.replace(',','').astype(float)


# In[22]:


datast["Close"]=datast["Close"].str.replace(',','').astype(float)
datast["Volume"]=datast["Volume"].str.replace(',','').astype(float)


# In[23]:


datast.info()


# In[24]:


datast.rolling(7).mean().head(20)
datast['Open'].plot(figsize=(16,6))
datast.rolling(window=30).mean()['Close'].plot()


# In[25]:


dataset['Close: 30 Day Mean']=dataset['Close'].rolling(window=30).mean()
dataset[['Close','Close: 30 Day Mean']].plot(figsize=(16,6))
#dataset['Close'].expanding(min_periods=1).mean().plot(figsize=(16,6))


# In[26]:


dataset['Close'].expanding(min_periods=1).mean().plot(figsize=(16,6))


# In[27]:


training_set=dataset['Open']
training_set=pd.DataFrame(training_set)


# In[29]:


from sklearn.preprocessing import MinMaxScaler
sc =MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)
#Creating a data structure with 60 timesteps and 1 output
X_train=[]
y_train=[]
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train,y_train=np.array(X_train), np.array(y_train)


# In[30]:


X_train=np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))


# In[39]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


# In[40]:


regressor=Sequential()


# In[41]:


#adding 1st LSTM layer and some dropout regularisation

regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

#adding 2nd LSTM layer and some dropout regularisation

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

#adding 3rd LSTM layer and some dropout regularisation

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

#adding 4th LSTM layer and some dropout regularisation

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#Adding output layer
regressor.add(Dense(units = 1))


# In[42]:


#compiling RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')
regressor.fit(X_train,y_train,epochs=100,batch_size=32)


# In[50]:


dataset_test=pd.read_csv(r'C:\Users\mohan\Downloads\Google_Stock_Price_Test.csv',index_col="Date")


# In[61]:


real_stock_price=dataset_test.iloc[:,1:2].values


# In[62]:


dataset_test.info()


# In[63]:


dataset_test["Volume"]=dataset_test["Volume"].str.replace(',','').astype(float)


# In[64]:


test_set=dataset_test['Open']


# In[65]:


test_set=pd.DataFrame(test_set)


# In[66]:


test_set.info()


# In[56]:


dataset_total=pd.concat((datast['Open'],dataset_test['Open']),axis=0)


# In[67]:


dataset_total=pd.concat((datast['Open'],dataset_test['Open']),axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
X_test=[]
for i in range(60,80):
    X_test.append(inputs[i-60:i, 0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price=regressor.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)


# In[68]:


predicted_stock_price=pd.DataFrame(predicted_stock_price)
predicted_stock_price.info()


# In[70]:


#Visualising the result
plt.plot(real_stock_price, color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price,color='blue',label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

