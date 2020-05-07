# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:52:57 2020

@author: Santosh Sah
"""

from AutoRegressiveMovingAverageForecastingUtils import (importAutoRegressiveMovingAverageForecastingDataset, saveAutoRegressiveMovingAverageForecastingForecastedValues,
                                               readAutoRegressiveMovingAverageForecastingForecastedValues, readAutoRegressiveMovingAverageForecastingModelForFullDataset)
from AutoRegressiveMovingAverageForecastingVisualization import visualizeAutoRegressiveMovingAverageForecastingForecastedValues

def forecastAutoRegressiveMovingAverageForecastingModel():
    
    #reading the dataset
    autoRegressiveMovingAverageForecastingDataset = importAutoRegressiveMovingAverageForecastingDataset("DailyTotalFemaleBirths.csv")
    
    #reading the model whichis trained on the whole dataset
    autoRegressiveMovingAverageForecastingModel = readAutoRegressiveMovingAverageForecastingModelForFullDataset()
    
    #forecasting for 11 months
    autoRegressiveMovingAverageForecastingForecastedValues = autoRegressiveMovingAverageForecastingModel.predict(len(autoRegressiveMovingAverageForecastingDataset),
                                                                          len(autoRegressiveMovingAverageForecastingDataset)+11,
                                                                          typ='levels').rename("ARMA(2,2) Prediction")
    
    #saving the forecasted values
    saveAutoRegressiveMovingAverageForecastingForecastedValues(autoRegressiveMovingAverageForecastingForecastedValues)

def plotAutoRegressiveMovingAverageForecastingForecastedValues():
    
    #reading the dataset
    autoRegressiveMovingAverageForecastingDataset = importAutoRegressiveMovingAverageForecastingDataset("DailyTotalFemaleBirths.csv")
    
    #reading the forecated values
    autoRegressiveMovingAverageForecastingForecastedValues = readAutoRegressiveMovingAverageForecastingForecastedValues()
    
    #visualizing the forecated values
    visualizeAutoRegressiveMovingAverageForecastingForecastedValues(autoRegressiveMovingAverageForecastingDataset, autoRegressiveMovingAverageForecastingForecastedValues)

if __name__ == "__main__":
    #forecastAutoRegressiveMovingAverageForecastingModel()
    plotAutoRegressiveMovingAverageForecastingForecastedValues()
    