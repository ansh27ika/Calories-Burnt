## PREDICTION MODEL 
## TO DETECT THE AMOUNT PF CALORIES BURNT

project idea: Calories in the foods we eat provide energy in the form of heat so that our bodies can function.
This means that we need to eat a certain amount of calories just to sustain life. But if we take in too many calories,
then we risk gaining weight.
So there is need to burn Calories, 
for burning calories we doing exercises and more.
for know how much calories we have burn 
Today we are going to buid a machine learning model
that predict calories based on some data.


>>importing dataset 
i combined to datasets
calories_burnt.csv and exercise.csv
taken as a dataset from kaggle


# importing impotant libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

>>defining libraries
1.matplot and seaborn used for datavisualization
2.XGBoost is an efficient implementation of gradient boosting 
that can be used for regression predictive modeling.
