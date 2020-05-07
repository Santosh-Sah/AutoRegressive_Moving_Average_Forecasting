# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:50:57 2020

@author: Santosh Sah
"""
import pandas as pd
import pickle
from statsmodels.tsa.stattools import adfuller
"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importAutoRegressiveMovingAverageForecastingDataset(autoRegressiveMovingAverageForecastingDatasetFileName):
    
    autoRegressiveMovingAverageForecastingDataset = pd.read_csv(autoRegressiveMovingAverageForecastingDatasetFileName,index_col='Date',parse_dates=True)
    
    #the dataset is minthly dataset. Hence setting its frequency as monthly.
    autoRegressiveMovingAverageForecastingDataset.index.freq = "D"
    
    #we only need first four month data
    autoRegressiveMovingAverageForecastingDataset[:120]
    
    return autoRegressiveMovingAverageForecastingDataset

#splitting dataset into training and testing set
def splitAutoRegressiveMovingAverageForecastingDataset(autoRegressiveMovingAverageForecastingDataset):
    
    #splitting the dataset into training and testing set.
    autoRegressiveMovingAverageForecastingTrainingSet = autoRegressiveMovingAverageForecastingDataset.iloc[:90]
    autoRegressiveMovingAverageForecastingTestingSet = autoRegressiveMovingAverageForecastingDataset.iloc[90:]
    
    return autoRegressiveMovingAverageForecastingTrainingSet, autoRegressiveMovingAverageForecastingTestingSet

#test dataset is stationary or non stationary
def agumentedDickeyFullerTest(series,title=''):
    
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        
"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)

"""
read X_train from pickle file
"""
def readAutoRegressiveMovingAverageForecastingXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readAutoRegressiveMovingAverageForecastingXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
Save AutoRegressiveMovingAverageForecasting as a pickle file.
"""
def saveAutoRegressiveMovingAverageForecastingModel(autoRegressiveMovingAverageForecastingModel):
    
    #Write AutoRegressiveMovingAverageForecastingModel as a picke file
    with open("AutoRegressiveMovingAverageForecastingModel.pkl",'wb') as AutoRegressiveMovingAverageForecastingModel_Pickle:
        pickle.dump(autoRegressiveMovingAverageForecastingModel, AutoRegressiveMovingAverageForecastingModel_Pickle, protocol = 2)

"""
read AutoRegressiveMovingAverageForecasting from pickle file
"""
def readAutoRegressiveMovingAverageForecastingModel():
    
    #load AutoRegressiveMovingAverageForecastingModel model
    with open("AutoRegressiveMovingAverageForecastingModel.pkl","rb") as AutoRegressiveMovingAverageForecastingModel:
        autoRegressiveMovingAverageForecastingModel = pickle.load(AutoRegressiveMovingAverageForecastingModel)
    
    return autoRegressiveMovingAverageForecastingModel

"""
Save AutoRegressiveMovingAverageForecasting as a pickle file.
"""
def saveAutoRegressiveMovingAverageForecastingModelForFullDataset(autoRegressiveMovingAverageForecastingModelForFullDataset):
    
    #Write AutoRegressiveMovingAverageForecastingModelForFullDataset as a picke file
    with open("AutoRegressiveMovingAverageForecastingModelForFullDataset.pkl",'wb') as AutoRegressiveMovingAverageForecastingModelForFullDataset_Pickle:
        pickle.dump(autoRegressiveMovingAverageForecastingModelForFullDataset, AutoRegressiveMovingAverageForecastingModelForFullDataset_Pickle, protocol = 2)

"""
read AutoRegressiveMovingAverageForecasting from pickle file
"""
def readAutoRegressiveMovingAverageForecastingModelForFullDataset():
    
    #load AutoRegressiveMovingAverageForecastingModelForFullDataset model
    with open("AutoRegressiveMovingAverageForecastingModelForFullDataset.pkl","rb") as AutoRegressiveMovingAverageForecastingModelForFullDataset:
        autoRegressiveMovingAverageForecastingModelForFullDataset = pickle.load(AutoRegressiveMovingAverageForecastingModelForFullDataset)
    
    return autoRegressiveMovingAverageForecastingModelForFullDataset

"""
save AutoRegressiveMovingAverageForecasting PredictedValues as a pickle file
"""

def saveAutoRegressiveMovingAverageForecastingPredictedValues(autoRegressiveMovingAverageForecastingPredictedValues):
    
    #Write AutoRegressiveMovingAverageForecastingPredictedValues in a picke file
    with open("AutoRegressiveMovingAverageForecastingPredictedValues.pkl",'wb') as autoRegressiveMovingAverageForecastingPredictedValues_Pickle:
        pickle.dump(autoRegressiveMovingAverageForecastingPredictedValues, autoRegressiveMovingAverageForecastingPredictedValues_Pickle, protocol = 2)

"""
read AutoRegressiveMovingAverageForecasting PredictedValues from pickle file
"""
def readAutoRegressiveMovingAverageForecastingPredictedValues():
    
    #load AutoRegressiveMovingAverageForecastingPredictedValues
    with open("AutoRegressiveMovingAverageForecastingPredictedValues.pkl","rb") as AutoRegressiveMovingAverageForecastingPredictedValues_pickle:
        autoRegressiveMovingAverageForecastingPredictedValues = pickle.load(AutoRegressiveMovingAverageForecastingPredictedValues_pickle)
    
    return autoRegressiveMovingAverageForecastingPredictedValues

"""
save AutoRegressiveMovingAverageForecasting ForecastedValues as a pickle file
"""

def saveAutoRegressiveMovingAverageForecastingForecastedValues(autoRegressiveMovingAverageForecastingForecastedValues):
    
    #Write AutoRegressiveMovingAverageForecastingForecastedValues in a picke file
    with open("AutoRegressiveMovingAverageForecastingForecastedValues.pkl",'wb') as autoRegressiveMovingAverageForecastingForecastedValues_Pickle:
        pickle.dump(autoRegressiveMovingAverageForecastingForecastedValues, autoRegressiveMovingAverageForecastingForecastedValues_Pickle, protocol = 2)

"""
read AutoRegressiveMovingAverageForecastingForecastedValues from pickle file
"""
def readAutoRegressiveMovingAverageForecastingForecastedValues():
    
    #load AutoRegressiveMovingAverageForecastingForecastedValues
    with open("AutoRegressiveMovingAverageForecastingForecastedValues.pkl","rb") as autoRegressiveMovingAverageForecastingForecastedValues_pickle:
        autoRegressiveMovingAverageForecastingForecastedValues = pickle.load(autoRegressiveMovingAverageForecastingForecastedValues_pickle)
    
    return autoRegressiveMovingAverageForecastingForecastedValues


