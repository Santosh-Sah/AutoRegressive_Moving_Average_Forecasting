# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:52:22 2020

@author: Santosh Sah
"""

from AutoRegressiveMovingAverageForecastingUtils import (readAutoRegressiveMovingAverageForecastingXTest, readAutoRegressiveMovingAverageForecastingModel, 
                                               saveAutoRegressiveMovingAverageForecastingPredictedValues, readAutoRegressiveMovingAverageForecastingXTrain,
                                               readAutoRegressiveMovingAverageForecastingPredictedValues)
from AutoRegressiveMovingAverageForecastingVisualization import visualizeAutoRegressiveMovingAverageForecastingPredictedValues

"""
test the model on testing dataset
"""
def testAutoRegressiveMovingAverageForecastingModel():
    
    #reading the training dataset
    X_train = readAutoRegressiveMovingAverageForecastingXTrain()
    
    #reading testing set
    X_test = readAutoRegressiveMovingAverageForecastingXTest()
    
    start = len(X_train)
    
    end = len(X_train) + len(X_test) - 1
    
    #reading model from pickle file
    autoRegressiveMovingAverageForecastingModel = readAutoRegressiveMovingAverageForecastingModel()
    
    #forecasting for 36 months
    predictedValues = autoRegressiveMovingAverageForecastingModel.predict(start = start, end = end).rename("ARMA(2,2) Prediction")
    
    #saving the foreasted values
    saveAutoRegressiveMovingAverageForecastingPredictedValues(predictedValues)

def plotAutoRegressiveMovingAverageForecastingPredictedValues():
    
    #reading testing set
    X_test = readAutoRegressiveMovingAverageForecastingXTest()
    
    #reading predicted value
    predictedValues = readAutoRegressiveMovingAverageForecastingPredictedValues()
    
    #visualizing the predicted values with training set and the testing set
    visualizeAutoRegressiveMovingAverageForecastingPredictedValues(X_test, predictedValues)
    
    
if __name__ == "__main__":
    #testAutoRegressiveMovingAverageForecastingModel()
    plotAutoRegressiveMovingAverageForecastingPredictedValues()