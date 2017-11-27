# -*- coding:utf-8 -*-
import pandas as pd
import pylab as plt
import numpy as np
import statsmodels.api as sm

#Create a dataframe df by loading the data using read_csv
df = pd.read_csv('logistic_trades.csv')
print(df.tail())

#Analyze the statistics of df
print(df.describe())

#Display the mean of each column separately
print(df.mean())

#Since the news sentiment has only 4 levels, draw the following table using crosstab
print(pd.crosstab(df['success'],df['sentiment'],rownames=['success']))

#Draw the histogram for each column
df.hist()
plt.show()

#Sentiment is a categorical variable. We are going to transform this variable into 4 dummy variables using the command get_dummies from pandas.
#Following the previous example create dummy variables for Sentiment using the function get_dummies. Store the result into data_dummy
data_dummy = pd.get_dummies(df.sentiment,prefix='sentiment')
print(data_dummy.head())

#Create a joint to keep success, news intensity, price, sentiment_2, sentiment_3 and sentiment_4
data = df[['success','news intensity','price']].join(data_dummy[['sentiment_2','sentiment_3','sentiment_4']])
print(data.head())

#Add the intercept manually (a column named intersect will only 1)
data['intersect'] = np.ones(len(data))
print(data.head())

#Perform logistic regression.
# Step 1: Remove the column name ‘success’
colnames = data.columns[1:]
print(colnames)

#Step 2: Create the logistic model
Logit_model = sm.Logit(data['success'],data[colnames])
result = Logit_model.fit()
print(result.summary())

#Step 3: Fi the model
print(result.conf_int())

#Interpret the following result:
print(np.exp(result.params))

#Calculate confidence interval with the function conf_int() associated to result
print(df.norm.interval(0.95))

#Display odds ration (just use np.exp in the params of result)
print(np.exp(df.coef_))