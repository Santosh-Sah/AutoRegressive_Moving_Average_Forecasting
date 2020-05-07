# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:51:54 2020

@author: Santosh Sah
"""
from statsmodels.tsa.arima_model import ARMA, ARMAResults
from pmdarima import auto_arima

from AutoRegressiveMovingAverageForecastingUtils import (saveAutoRegressiveMovingAverageForecastingModel, readAutoRegressiveMovingAverageForecastingXTrain, 
                                               importAutoRegressiveMovingAverageForecastingDataset, saveAutoRegressiveMovingAverageForecastingModelForFullDataset,
                                               agumentedDickeyFullerTest)

"""
Train AutoRegressiveMovingAverageForecasting model on training set
"""
def trainAutoRegressiveMovingAverageForecastingModel():
    
    X_train = readAutoRegressiveMovingAverageForecastingXTrain()
    
    #training model on the training set
    autoRegressiveMovingAverageForecastingModel = ARMA(X_train["Births"], order = (2, 2))
    
    autoRegressiveMovingAverageForecastingModelFitResult = autoRegressiveMovingAverageForecastingModel.fit()
    
    autoRegressiveMovingAverageForecastingModelFitResult.summary()
    
    saveAutoRegressiveMovingAverageForecastingModel(autoRegressiveMovingAverageForecastingModelFitResult)

"""
Train AutoRegressiveMovingAverageForecasting model on full dataset
"""
def trainAutoRegressiveMovingAverageForecastingModelOnFullDataset():
    
    autoRegressiveMovingAverageForecastingDataset = importAutoRegressiveMovingAverageForecastingDataset("DailyTotalFemaleBirths.csv")
    
    #training model on the whole dataset
    autoRegressiveMovingAverageForecastingModel = ARMA(autoRegressiveMovingAverageForecastingDataset["Births"], order = (2, 2))
    
    autoRegressiveMovingAverageForecastingModelFitResult = autoRegressiveMovingAverageForecastingModel.fit()
    
    autoRegressiveMovingAverageForecastingModelFitResult.summary()
    
    saveAutoRegressiveMovingAverageForecastingModelForFullDataset(autoRegressiveMovingAverageForecastingModelFitResult)

def testIsDatasetStationary():
    
    autoRegressiveMovingAverageForecastingDataset = importAutoRegressiveMovingAverageForecastingDataset("DailyTotalFemaleBirths.csv")
    
    agumentedDickeyFullerTest(autoRegressiveMovingAverageForecastingDataset["Births"])
    
# =============================================================================
#     Augmented Dickey-Fuller Test:
#     ADF test statistic       -4.808291
#     p-value                   0.000052
#     # lags used               6.000000
#     # observations          358.000000
#     critical value (1%)      -3.448749
#     critical value (5%)      -2.869647
#     critical value (10%)     -2.571089
#     Strong evidence against the null hypothesis
#     Reject the null hypothesis
#     Data has no unit root and is stationary
# =============================================================================
    
def determineARMAOrderOfPAndQ():
    
    autoRegressiveMovingAverageForecastingDataset = importAutoRegressiveMovingAverageForecastingDataset("DailyTotalFemaleBirths.csv")
    
    auto_arima(autoRegressiveMovingAverageForecastingDataset["Births"], seasonal = False, trace = True).summary()
    
# =============================================================================
#     Fit ARIMA(2,1,2)x(0,0,0,0) [intercept=True]; AIC=2463.038, BIC=2486.421, Time=0.631 seconds
#     Fit ARIMA(0,1,0)x(0,0,0,0) [intercept=True]; AIC=2650.760, BIC=2658.555, Time=0.025 seconds
#     Fit ARIMA(1,1,0)x(0,0,0,0) [intercept=True]; AIC=2565.234, BIC=2576.925, Time=0.105 seconds
#     Fit ARIMA(0,1,1)x(0,0,0,0) [intercept=True]; AIC=2463.584, BIC=2475.275, Time=0.140 seconds
#     Fit ARIMA(0,1,0)x(0,0,0,0) [intercept=False]; AIC=2648.768, BIC=2652.665, Time=0.019 seconds
# =============================================================================

if __name__ == "__main__":
    #testIsDatasetStationary()   
    #determineARMAOrderOfPAndQ()
    #trainAutoRegressiveMovingAverageForecastingModel()
    trainAutoRegressiveMovingAverageForecastingModelOnFullDataset()
