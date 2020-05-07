# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:53:28 2020

@author: Santosh Sah
"""
import pylab

def visualizeAutoRegressiveMovingAverageForecastingPredictedValues(autoRegressiveMovingAverageForecastingXTest, autoRegressiveMovingAverageForecastingPredictedValues):
    
    #plotting the predicted values, and testing set
    title = 'Daily Total Female Births'
    
    ylabel='Births'
    
    xlabel='' 

    ax = autoRegressiveMovingAverageForecastingXTest['Births'].plot(legend=True,figsize=(12,6),title=title)
    
    autoRegressiveMovingAverageForecastingPredictedValues.plot(legend=True)
    
    ax.autoscale(axis='x',tight=True)
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    pylab.savefig('PredeictedValues.png')

def visualizeAutoRegressiveMovingAverageForecastingForecastedValues(autoRegressiveMovingAverageForecastingDataset, autoRegressiveMovingAverageForecastingForecastedValues):
    
    #plotting the predicted values, and testing set
    title = 'Daily Total Female Births'
    
    ylabel='Births'
    
    xlabel='' 

    ax = autoRegressiveMovingAverageForecastingDataset['Births'].plot(legend=True,figsize=(12,6),title=title)
    
    autoRegressiveMovingAverageForecastingForecastedValues.plot(legend=True)
    
    ax.autoscale(axis='x',tight=True)
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    pylab.savefig('ForecastedValues.png')