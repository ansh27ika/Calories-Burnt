
## TO DETECT THE AMOUNT OF CALORIES BURNT
###### REGRESSION MODEL 

### PROJECT OVERVIEW

Calories in the foods we eat provide energy in the form of heat so that our bodies can function.
This means that we need to eat a certain amount of calories just to sustain life. But if we take in too many calories,
then we risk gaining weight.
So there is need to burn Calories, 
for burning calories we doing exercises and more.
for know how much calories we have burn 
Today we are going to buid a machine learning model
that predict calories based on some data.




# importing impotant libraries 

1. `numpy`
2. ` pandas` 
3. `matplotlib`
4. `seaborn` 
5. `train_test_split`
6. `XGBRegressor`
7. `Metrics`

### Importing dataset 
combined 2 datasets
calories_burnt.csv and exercise.csv
taken as a dataset from kaggle


### Observation
 
1. We can see that when our duration of exercise increase our burning of calories increase.
2. Distribution on the basis of height shows :NORMAL DISTRIBUTION
3. Distribution on the basis of AGE shows : RIGHT SKEWED DISTRIBUTION
 after encoding  gender column has changed data. 1 for male and 0 for female  because ml only take numerical data.
###### In compare to linear model randomforest give more accuracy and  is a very good accuracy about 99 PERCENT 
