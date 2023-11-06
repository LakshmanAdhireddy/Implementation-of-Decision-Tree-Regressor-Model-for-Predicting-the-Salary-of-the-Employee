# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare your data
Collect and clean data on employee salaries and features Split data into training and testing sets

2.Define your model
Use a Decision Tree Regressor to recursively partition data based on input features Determine maximum depth of tree and other hyperparameters

3.Train your model
Fit model to training data Calculate mean salary value for each subset

4.Evaluate your model
Use model to make predictions on testing data Calculate metrics such as MAE and MSE to evaluate performance

5.Tune hyperparameters
Experiment with different hyperparameters to improve performance

6.Deploy your model
Use model to make predictions on new data in real-world application.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Lakshman
RegisterNumber:  212222240001
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output:
### Initial dataset:
### Data Info:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707265/6715a95d-9eec-45fb-aee4-d4636fb8d3db)

### Optimization of null values:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707265/dba911fa-2845-4815-afd9-bb92b6760ae3)

### Converting string literals to numericl values using label encoder:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707265/6b1699be-f8ef-45af-bcaf-0ae572514f65)

### Assigning x and y values:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707265/03759d66-ce4a-4842-9a68-721479208a85)

### Mean Squared Error:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707265/2b958c8b-439c-434c-ad75-def8c8732f5e)

### R2 (variance):
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707265/541467a5-01f4-4c16-b731-5189ca7a3441)

### Prediction:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707265/adca78e6-7073-43b9-919c-46e7ccff35c8)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
