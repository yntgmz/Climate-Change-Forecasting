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



#git remote add origin https://github.com/yntgmz/Climate-Change-Forecasting.git
#git branch -M main
#git push -u origin main



