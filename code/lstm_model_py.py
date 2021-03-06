# -*- coding: utf-8 -*-
"""lstm_model.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VIN72l-a46w3qmAZPmmb_w2Lofh7hgUV
"""

!pip install --upgrade pandas-datareader

!pip install --upgrade pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.model_selection import train_test_split

start = '2010-01-01'
end = '2022-12-31'

df = data.DataReader('AAPL','yahoo',start,end)

df.tail()

df = df.reset_index()

df = df.drop(['Date','Adj Close'], axis=1)

plt.plot(df.Close)

ma_100 = df.Close.rolling(100).mean()
ma_200 = df.Close.rolling(200).mean()

plt.figure(figsize=(12,10))
plt.plot(df.Close)
plt.plot(ma_100,'r')
plt.plot(ma_200, 'black')

df.columns

#Data splitting
data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

data_test.head()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_array = scaler.fit_transform(data_train)
data_test_array = scaler.fit_transform(data_test)

data_train_array

X_train = []
y_train = []

for i in range(100, data_train_array.shape[0]):
  X_train.append(data_train_array[i-100:i])
  y_train.append(data_train_array[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train.shape

data_train_array[1]

X_train[1]

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units=50, activation = 'relu', return_sequences= True,
               input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=70, activation = 'relu', return_sequences= True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation = 'relu', return_sequences= True))
model.add(Dropout(0.4))

model.add(LSTM(units=50, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.summary()

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train,y_train, epochs=50)

!mkdir -p saved_model
#model.save('saved_model/my_model') 

model.save('final_project/model')

# my_model directory
!ls final_project

# Contains an assets folder, saved_model.pb, and variables folder.
!ls final_project/model

data_test.head()

data_train.head()

past_100_days = data_train.tail(100)

past_100_days

final_df = past_100_days.append(data_test, ignore_index= True)

final_df.head()

input_data = scaler.fit_transform(final_df)

input_data.shape

X_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  X_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

X_test , y_test = np.array(X_test), np.array(y_test)

X_test.shape

y_predict = model.predict(X_test)

y_test

scaler.scale_

scale = 1/0.00682769
y_predict = y_predict*scale
y_test = y_test*scale

plt.figure(figsize=(12,8))
plt.plot(y_predict,'r',label='predict')
plt.plot(y_test,'b',label='org')
plt.legend()

import tensorflow as tf
from tensorflow import keras

