import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.model_selection import train_test_split
import keras
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2021-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker','AAPL')
df = data.DataReader(user_input,'yahoo',start,end)

#Displaying Data
st.subheader('Data from 2010-2022')
st.write(df.head())

#Graphs
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(11,8))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('100 Moving AVG')
ma_100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(11,8))
plt.plot(df.Close)
plt.plot(ma_100)
st.pyplot(fig)

st.subheader('200 Moving AVG And 100 Moving AVG')
ma_200 = df.Close.rolling(200).mean()
ma_100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(11,8))
plt.plot(df.Close,'g')
plt.plot(ma_100,'b')
plt.plot(ma_200,'r')
st.pyplot(fig)

#Splitting Data

data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_array = scaler.fit_transform(data_train)
data_test_array = scaler.fit_transform(data_test)


#load the trained model

model = load_model('keras_model.h5')

#testing
past_100_days = data_train.tail(100)
final_df = past_100_days.append(data_test, ignore_index= True)
input_data = scaler.fit_transform(final_df)

X_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  X_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

X_test , y_test = np.array(X_test), np.array(y_test)
y_predict = model.predict(X_test)
scaler = scaler.scale_

scale = 1/scaler[0]
y_predict = y_predict*scale
y_test = y_test*scale

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,8))
plt.plot(y_predict,'r',label='predict')
plt.plot(y_test,'b',label='org')
plt.legend()
st.pyplot(fig2)

#NOTES
#1. Add Drop down button
#2. Beautify GUI
#3. Display Vairag graph's
# Explore other models