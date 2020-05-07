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
    
    print(meanSquredError) #59.91368652225238
    
    print(meanAbsoluteError) #5.892390823362883
    
    print(rootMeanSquaredError) #7.74039317620574
    
    
    
if __name__ == "__main__":
    testAutoRegressiveMovingAverageForecastingMetrics()