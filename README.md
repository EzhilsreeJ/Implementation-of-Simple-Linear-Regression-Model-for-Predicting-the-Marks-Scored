# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries for data handling, visualization, and model building.
2. Load the dataset and inspect the first and last few records to understand the data structure.
3. Prepare the data by separating the independent variable (hours studied) and the dependent variable (marks scored).
4. Split the dataset into training and testing sets to evaluate the model's performance.
5. Initialize and train a linear regression model using the training data.
6. Predict the marks for the test set using the trained model.
7. Evaluate the model by comparing the predicted marks with the actual marks from the test set.
8. Visualize the results for both the training and test sets by plotting the actual data points and the regression line. 

## Program:

### Program to implement the simple linear regression model for predicting the marks scored.
## Developed by: EZHIL SREE J
# RegisterNumber: 212223230056

```
import numpy as np
import pandas as pd
from sklearn.metrics import  mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
```
```
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/c7d96793-88a1-4219-8966-e0d9ca507923)
```
dataset.info()
```
# OUTPUT:
![image](https://github.com/user-attachments/assets/b65275c3-ad0a-4bf0-b4b7-ddb6b988ddce)

```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
# OUTPUT:
![image](https://github.com/user-attachments/assets/807f36ce-d4d2-4458-aa7a-d4e0ca1b800b)
![image](https://github.com/user-attachments/assets/3ce0e8bf-2868-42ea-8e30-8b2586eac026)

```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train.shape)
print(X_test.shape)
```
# OUTPUT:
![image](https://github.com/user-attachments/assets/c8ac9e21-9f80-4c26-a23f-b3c60dad1085)
```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
```
# OUTPUT :
![image](https://github.com/user-attachments/assets/26f385a7-fcb0-4f18-80bb-ae0ec80afb69)
```
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
# OUTPUT:
![image](https://github.com/user-attachments/assets/93a9a1bd-6550-4f40-8558-11daf09a25cc)
```
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,reg.predict(X_train),color="green")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
# OUTPUT:
![image](https://github.com/user-attachments/assets/84dc7a85-dbaf-4a5f-9fec-3f83019cbe14)

```
plt.scatter(X_test, Y_test,color="blue")
plt.plot(X_test, reg.predict(X_test), color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
# OUTPUT:
![image](https://github.com/user-attachments/assets/3e1b80f8-0385-4bc0-a1ab-de803997c83f)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
