# -*- coding: utf-8 -*-
"""Salary_prediction.101ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Yff1j1iBFeMGb0DLix-e2_8gEgmRbV1E

# Salary Prediction
"""

#Importing all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Salary Prediction of Data Professions.csv")
print(df.shape)
df.head()

df.describe()

df['UNIT'].unique()

df.iloc[df['SALARY'].idxmax()]

sns.histplot(df['SALARY'],bins=20)
plt.title('Distribution of Salaries')
plt.show()

sns.scatterplot(x=df['PAST EXP'],y= df['SALARY'])
plt.title('Salary vs Past Exp')
plt.show()

df['DOJ'] = pd.to_datetime(df['DOJ'])
df['CURRENT DATE'] = pd.to_datetime(df['CURRENT DATE'])
df["TENURE"] = df["CURRENT DATE"] -df['DOJ']

df

sns.scatterplot(y=df['SALARY'], x=df['TENURE'])
plt.show()

df.iloc[df['TENURE'].idxmax()]

df.isnull().sum()

df.fillna(method='ffill',inplace=True)

df.isnull().sum()

Numerical = df.select_dtypes(include=['int','float']).columns
Numerical

categorical = df[['SEX','UNIT','DESIGNATION']]
categorical

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df.head()

# Encode categorical variables using one-hot encoding
df= pd.get_dummies(df, columns=['SEX', 'DESIGNATION', 'UNIT'], drop_first=True)

df.head()

from sklearn.model_selection import train_test_split
#models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import Pipeline

#metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score

X = df.drop(['FIRST NAME', 'LAST NAME', 'SALARY', 'DOJ', 'CURRENT DATE','TENURE'], axis=1)
y =df['SALARY']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#train
scores=[]
model_names = ['Linear regression', 'Random Forest', 'Gradient Boosting',
               'Adaboost']
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting':GradientBoostingRegressor(),
     'adaboost':AdaBoostRegressor()
}

for name, model in models.items():
  model.fit(X_train,y_train)
  y_pred = model.predict(X_test)

  mse = mean_squared_error(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)
  r2 =r2_score(y_test,y_pred)
  scores.append(r2)

  print(f"{name} Metrics:")
  print(f"Mean Squared Error (MSE): {mse}")
  print(f"Mean Absolute Error (MAE): {mae}")
  print(f"R-squared (R2): {r2}")
  print("\n")

sns.lineplot(x = model_names ,y= scores)
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()



X_train

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred1= lr.predict(X_test)
r2_score(y_test,y_pred1)

import pickle
#save the best model
pickle.dump(lr,open('model101.pkl','wb'))

model101= pickle.load(open('model101.pkl','rb'))

def salary_prediction(inputs):
  return model101.predict[[inputs]]
