#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:07:47 2021

@author: yanetgomez
"""

# Import Libraries
import pandas as pd
#pip install plotly
import plotly.express as px
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go 
import os

# Set directory
os.chdir('/Users/yanetgomez/Desktop/TIME SERIES/TS Project/archive')

# Import Datasets
# source: https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data

temperature_df = pd.read_csv('GlobalLandTemperaturesByCountry.csv') #[577462 rows x 4 columns]

#Print the first and last 10 rows in the dataframe
first_10=temperature_df.head(10)
last_10=temperature_df.tail(10)
first_10
last_10
# Total number of samples
temperature_df.shape[0]

#Exploratory Data Analysis
temperature_df['Country'].unique()

#Check missing values
temperature_df.isnull().sum()

# Check dataframe information
temperature_df.info()

#Average, minimum, and maximum temperatures across entire dataset
temperature_df.describe()

# Data Cleaning
# Use Groupby country to see the count by country
country_group_df=temperature_df.groupby(by="Country").count().reset_index('Country').rename(columns={'AverageTemperature':'AverageTemperatureCount', 'AverageTemperatureUncertainty': 'AverageTemperatureUncertaintyCount'})
country_group_df

country_group_df["Country"]

from plotly.offline import plot
fig=px.bar(country_group_df, x = "Country", y = "AverageTemperatureCount")
plot(fig)

fig=px.bar(country_group_df, x = "Country", y = "AverageTemperatureUncertaintyCount")
plot(fig)

fig=px.histogram(country_group_df, x = "AverageTemperatureCount")
plot(fig)

#Remove countries with many missing values 
country_group_df[(country_group_df['AverageTemperatureCount']<1500)|(country_group_df['AverageTemperatureUncertaintyCount']<1500)]
countries_with_less_data=country_group_df[(country_group_df['AverageTemperatureCount']<1500)|(country_group_df['AverageTemperatureUncertaintyCount']<1500)]['Country'].tolist()
countries_with_less_data


temperature_df=temperature_df[~temperature_df['Country'].isin(countries_with_less_data)]

temperature_df.reset_index(inplace=True, drop=True)

# Fill in NA with rolling average on past 730 days


temperature_df['AverageTemperature'] = temperature_df['AverageTemperature'].fillna(temperature_df['AverageTemperature'].rolling(730, min_periods =1).mean())

temperature_df['AverageTemperatureUncertainty'] = temperature_df['AverageTemperatureUncertainty'].fillna(temperature_df['AverageTemperatureUncertainty'].rolling(730, min_periods =1).mean())

temperature_df.isna().sum()

# Check for different versions of the same country

temperature_df['Country'].unique()

duplicates=[]
for i in temperature_df['Country'].unique():
    if '(' in i:
        duplicates.append(i)
        
duplicates

# Replace Duplicates

temperature_df=temperature_df.replace(duplicates, ['Congo', 'Denmark', 'Falkland Islands','France', 'Netherlands', 'United Kingdom' ])

temperature_df['Country'].unique()

# Data Visualization

countries= temperature_df['Country'].unique().tolist()
mean_temperature=[]
for i in countries:
    mean_temperature.append(temperature_df[temperature_df['Country']==i]['AverageTemperature'].mean())

#Plot mean temperatures of countries
data = [dict(type ='choropleth', 
             locations = countries, 
             z= mean_temperature, 
             locationmode= 'country names')
        ] 

    
layout = dict (title = 'Average Global Land Temperatures', 
               geo =dict(showframe =False, 
                         showocean =True, 
                         oceancolor='aqua', 
                         projection = dict(type = 'orthographic')))

fig =go.Figure(dict(data=data, layout=layout))
plot(fig)
py.iplot(fig, validate =False, filename='worldmap')

fig.write_html('worldmap.html', auto_open=True)

# Animation of global temperature change

temperature_df['year'] = temperature_df['dt'].apply(lambda x: x.split('-')[0])
temperature_df

fig = px.choropleth(temperature_df, locations = 'Country',
                    locationmode = 'country names', 
                    color = 'AverageTemperature', 
                    hover_name = 'Country', 
                    animation_frame ='year', 
                    color_continuous_scale = px.colors.sequential.deep_r)

fig.show()
                                            
                            
plot(fig)
fig.write_html('tempchange.html', auto_open=True)



#Global Average
df_global = temperature_df.groupby('year').mean().reset_index()

df_global['year'] = df_global['year'].apply(lambda x: int(x))

df_global= df_global[df_global['year']> 1850]

#Uncertainty upper bound
trace1 = go.Scatter(
    x= df_global ['year'], 
    y= np.array (df_global['AverageTemperature']) + np.array(df_global['AverageTemperatureUncertainty']), 
    name = 'Uncertainty top', 
    line = dict(color ='green'))


#Uncertainty lower bound
trace2 = go.Scatter(
    x= df_global ['year'], 
    y= np.array (df_global['AverageTemperature']) - np.array(df_global['AverageTemperatureUncertainty']), 
    fill = 'tonexty', 
    name = 'Uncertainty bottom', 
    line = dict(color ='green'))



# Recorded Temperature

trace3 = go.Scatter(
    x= df_global['year'], 
    y= df_global['AverageTemperature'], 
    name= 'Average Temperature', 
    line = dict(color='red'))

data= [trace1, trace2, trace3]

layout = go.Layout(
    xaxis = dict(title = 'year'), 
    yaxis =dict (title = 'Average Temperature, ˚C'), 
    title= 'Average Land Temperatures Globally', 
    showlegend = False)


fig1 = go.Figure(data = data, layout = layout)
py.iplot(fig1)


plot(fig1)
fig1.write_html('templine.html', auto_open=True)



# us Data
US_df = temperature_df[temperature_df['Country']== 'United States'].reset_index(drop=True)

US_df

fig = px.line(title = 'US Temeprature Data')
US_df_updated = US_df[US_df['year'] > '1813']
fig.add_scatter(x = US_df_updated['dt'], y = US_df_updated['AverageTemperature'], name = 'US Temperature')

fig.show()
plot(fig)


# Prepare data for modeling

temperature_df['Month']=temperature_df['dt'].apply(lambda x: int(x.split('-')[1]))
df_global_monthly = temperature_df.groupby(['dt']).mean().reset_index()
df_global_monthly


# Create function that creates teh data for training the Time Series

def prepare_data (df, feature_range):
    #Get the columns
    columns = df.columns
    #For the given range, create lagged input feature for the given columns
    for i in range (1, (feature_range +1)):
        for j in columns[1:]:
            name = j + '_t-' + str(i)
            df[name]= df[j].shift((i))
            
    #Create the target by using next value as the target
    df['Target']=df['AverageTemperature'].shift(-1)
    return df

df_global_monthly = prepare_data(df_global_monthly,3)

df_global_monthly

df_global_monthly= df_global_monthly.dropna().reset_index(drop=True)

df_global_monthly

# Split Data
train = df_global_monthly[:int(0.9 * len(df_global_monthly))].drop(columns = 'dt').values
test = df_global_monthly[int(0.9 * len(df_global_monthly)):].drop(columns = 'dt').values

# Scale the data

scaler = MinMaxScaler(feature_range = (0,1))
train = scaler.fit_transform(train)
test = scaler.transform(test)


# Split data into input features and output targets

train_x, train_y = train[:,:-1], train [:, -1]
test_x, test_y = test[:, :-1], test [:, -1]

#Reshape inputs to be 3d (samples, timestamps, and features)

train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))

test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# Build and Train LSTM Model to predict global temperatures
def create_model(train_x):
    #Create the model
    inputs =keras.layers.Input(shape = (train_x.shape[1], train_x.shape[2]))
    x= keras.layers.LSTM(50, return_sequences = True)(inputs)
    x= keras.layers.Dropout(0.3)(x)
    x= keras.layers.LSTM(50, return_sequences = True)(x)
    x= keras.layers.Dropout(0.3)(x)
    x= keras.layers.LSTM(50)(x)
    outputs = keras.layers.Dense(1, activation = 'linear')(x)
    
    model = keras.Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = 'adam', loss= 'mse')
    return model


model = create_model(train_x)

model.summary()

# Fit the network
history = model.fit(train_x, train_y, epochs=50, batch_size = 72, validation_data= (test_x, test_y), shuffle = False)


#PLot history
def plot_history(history):
    #plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label ='test')
    plt.grid()
    plt.legend()
    plt.show()
    
plot_history(history)
    
    
# Assess Model Performance    

def prediction(model,test_x,train_x, df):
    
    #Predict unsing the model
    predict = model.predict(test_x)
    
    #Reshape test_x and traxin x for visualization and reverse scaling 
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[2]))
    
    #Concatenate test_x with predicted value
    predict_ = np.concatenate((test_x, predict), axis = 1)
    
    # Reverse scaling
    predict_ = scaler.inverse_transform(predict_)
    original_ = scaler.inverse_transform(test)
    
    #Create dataframe to store predicted and original values
    pred = pd.DataFrame()
    pred['dt'] = df ['dt'][-test_x.shape[0]:]
    pred['Original'] = original_[:, -1]
    pred['Predicted'] = predict_[:, -1]
    
    #Calculate the erorr
    pred['Error'] = pred['Original']- pred['Predicted']
    
    #Create dataframe for visualization
    df= df[['dt', 'AverageTemperature']][: -test_x.shape[0]]
    df.columns = ['dt', 'Original']
    original = df.append(pred[['dt', 'Original']])
    df.columns = ['dt', 'Predicted']
    predicted = df.append(pred[['dt', 'Predicted']])
    original = original.merge(predicted, left_on = 'dt', right_on ='dt')
    return pred, original 


pred, original = prediction(model, test_x, train_x, df_global_monthly)

def plot_error(df): 
    
    #Plotting the currenr and predicted values
    fig = px.line(title = "Prediction vs Actual")
    fig.add_scatter(x = df['dt'], y = df['Original'], name = 'Original', opacity= 0.7)
    fig.add_scatter(x = df['dt'], y = df['Predicted'], name = 'Predicted', opacity= 0.5)
    fig.show()
    fig.write_html('predvsActual.html', auto_open=True)

    
    fig = px.line(title = 'Error')
    fig = fig.add_scatter(x = df['dt'], y = df['Error'])
    fig.show()
    fig.write_html('predvsActual.html', auto_open=True)


plot_error(pred)


def plot (df):
    #Plotting the current and predicted values
    fig= px.line(title = "Prediction vs Actual")
    fig.add_scatter(x =df['dt'], y = df['Original'], name = 'Original', opacity= 0.7)
    fig.add_scatter(x =df['dt'], y = df['Predicted'], name = 'Predicted', opacity= 0.5)
    fig.show()
    fig.write_html('original.html', auto_open=True)
    
plot(original)    
    
    
# Model for US Data

US_df = temperature_df[temperature_df['Country']=='United States'].reset_index(drop=True)

US_df
    
US_df = US_df.drop(['Country', 'year'], axis=1)


US_df= prepare_data(US_df,3)

US_df

US_df= US_df.dropna().reset_index(drop=True)

# Split the data

train= US_df[:int(0.9 * len(US_df))].drop(columns='dt').values

test= US_df[int(0.9 * len(US_df)):].drop(columns = 'dt').values
            
            
train.shape
test.shape
# Scale the data

scaler = MinMaxScaler(feature_range = (0,1))
train = scaler.fit_transform(train)
test = scaler.transform(test)

# Split the data
train_x, train_y = train[:,:-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]

#Reshape inputs to be 3d (samples, timestamps, and features)

train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))

test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

train_x


# Prediction for US
model2 = create_model(train_x)
model2.summary()

# Fit the network
history2 = model2.fit(train_x, train_y, epochs =50, batch_size =72, validation_data = (test_x, test_y), shuffle = False)
plot_history(history2)


# Predict and Assess model performance 

pred, original = prediction(model2 , test_x, train_x, US_df)

plot(original)

# Model for Italy Data

Italy_df = temperature_df[temperature_df['Country']=='Italy'].reset_index(drop=True)

Italy_df
    
Italy_df = Italy_df.drop(['Country', 'year'], axis=1)


Italy_df= prepare_data(Italy_df,3)

Italy_df

Italy_df= Italy_df.dropna().reset_index(drop = True)

# Split the data

train_Italy =Italy_df[:int(0.9 * len(Italy_df))].drop(columns ='dt').values
test_Italy = Italy_df[int(0.9 * len(Italy_df)):].drop(columns = 'dt').values 

train_Italy.shape
test_Italy.shape

# Scale the data

scaler = MinMaxScaler(feature_range = (0,1))
train_Italy = scaler.fit_transform(train_Italy)
test_Italy = scaler.transform(test_Italy)

# Split the data
train_Italy_x, train_Italy_y = train_Italy[:,:-1], train_Italy[:, -1]
test_Italy_x, test_Italy_y = test_Italy[:, :-1], test_Italy[:, -1]


#Reshape inputs to be 3d (samples, timestamps, and features)

train_Italy_x = train_Italy_x.reshape((train_Italy_x.shape[0], 1, train_Italy_x.shape[1]))

test_Italy_x = test_Italy_x.reshape((test_Italy_x.shape[0], 1, test_Italy_x.shape[1]))

print(train_Italy_x.shape, train_Italy_y.shape, test_Italy_x.shape, test_Italy_y.shape)




# Prediction for Italy
model3 = create_model(train_Italy_x)
model3.summary()

# Fit the network
history3 = model3.fit(train_Italy_x, train_Italy_y, epochs =50, batch_size =72, validation_data = (test_Italy_x, test_Italy_y), shuffle = False)
plot_history(history3)


# Predict and Assess model performance 

pred, original = prediction(model3 , test_Italy_x, train_Italy_x, Italy_df)

test_Italy_x.shape 
train_Italy_x.shape

plot(original)


     