## PREDICTION MODEL 
## TO DETECT THE AMOUNT OF CALORIES BURNT

project idea: Calories in the foods we eat provide energy in the form of heat so that our bodies can function.
This means that we need to eat a certain amount of calories just to sustain life. But if we take in too many calories,
then we risk gaining weight.
So there is need to burn Calories, 
for burning calories we doing exercises and more.
for know how much calories we have burn 
Today we are going to buid a machine learning model
that predict calories based on some data.


>>importing dataset 
i combined 2 datasets
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

>> observation 
>> 1. We can see that when our duration of exercise increase our burning of calories increase.
>> 2. Distribution on the basis of height shows NORMAL DISTRIBUTION
>> 3. Distribution on the basis of AGE shows RIGHT SKEWED DISTRIBUTION
 after encoding  gender column has changed data. 1 for male and 0 for female  because ml only take numerical data.
 ,in compare to linear model randomforest give more accuracy and  is a very good accuracy about 99 PERCENT 
