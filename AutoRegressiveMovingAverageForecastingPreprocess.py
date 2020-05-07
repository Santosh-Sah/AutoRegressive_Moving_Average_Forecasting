# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:51:38 2020

@author: Santosh Sah
"""

from AutoRegressiveMovingAverageForecastingUtils import (importAutoRegressiveMovingAverageForecastingDataset, saveTrainingAndTestingDataset, 
                                                         splitAutoRegressiveMovingAverageForecastingDataset)

def preprocess():
    
    autoRegressiveMovingAverageForecastingDataset = importAutoRegressiveMovingAverageForecastingDataset("DailyTotalFemaleBirths.csv")
    
    X_train, X_test = splitAutoRegressiveMovingAverageForecastingDataset(autoRegressiveMovingAverageForecastingDataset)
    
    saveTrainingAndTestingDataset(X_train, X_test)
    

if __name__ == "__main__":
    preprocess()