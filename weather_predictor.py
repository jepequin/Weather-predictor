import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

#Uses genetic programming to decide on best ML model for predicting country temperatures

#select country
country = input('Enter the name of a country: ')


#load the data
temperatures = pd.read_csv('TemperaturesByCountry.csv')

#Select rows corresponding to selected country 
country_temp = temperatures.loc[temperatures['Country']==country]

#Erase NaN values, columns having type 'object' and round temperature values
country_temp = country_temp.dropna().reset_index().drop(['dt','Country'],axis=1).round(decimals=0)

#Get rid of 'AverageTemperature' values occurring less than 2 times
country_temp = country_temp[country_temp.groupby('AverageTemperature').index.transform(len) > 1]

#clean the data
#temperature_shuffle = temperature.iloc[np.random.permutation(len(temperature))]
#temp = temperature_shuffle.reset_index(drop=True) 

#Split training and testing data
training_indices, testing_indices = train_test_split(country_temp.index, train_size=0.75, test_size=0.25)

#Let Generic Programming find best ML model and hyperparameters
tpot = TPOTClassifier(generations=2, verbosity=2)
tpot.fit(country_temp.drop('AverageTemperature', axis=1).loc[training_indices].values, country_temp.loc[training_indices,'AverageTemperature'].values)

#Export generated code
tpot.export('weather_predictor_pipeline.py')