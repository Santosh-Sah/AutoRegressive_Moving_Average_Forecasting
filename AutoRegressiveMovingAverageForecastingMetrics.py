# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:41:20 2020

@author: Santosh Sah
"""
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from AutoRegressiveMovingAverageForecastingUtils import (readAutoRegressiveMovingAverageForecastingXTest, readAutoRegressiveMovingAverageForecastingPredictedValues)

"""

calculating AutoRegressiveMovingAverageForecasting metrics

"""
def testAutoRegressiveMovingAverageForecastingMetrics():
    
    #reading testing set
    X_test = readAutoRegressiveMovingAverageForecastingXTest()
    
    #reading predicted value
    predictedValues = readAutoRegressiveMovingAverageForecastingPredictedValues()
    
    meanSquredError = mean_squared_error(X_test, predictedValues)
    
    meanAbsoluteError = mean_absolute_error(X_test, predictedValues)
    
    rootMeanSquaredError = np.sqrt(mean_squared_error(X_test, predictedValues))
    
    print(meanSquredError) #1143.4649378653387
    
    print(meanAbsoluteError) #30.24228895401259
    
    print(rootMeanSquaredError) #33.815158403670665
    
    
    
if __name__ == "__main__":
    testAutoRegressiveMovingAverageForecastingMetrics()