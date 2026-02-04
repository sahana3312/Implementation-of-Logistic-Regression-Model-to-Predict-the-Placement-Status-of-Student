# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the students placement dataset from the CSV file and preprocess the data.
2. Select relevant features and convert the placement status into numerical form.
3. Split the dataset into training and testing sets and train the Logistic Regression model.
4. predict the placemesnt status of students and evaluate the model using using accuracy. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SAHANA S
RegisterNumber: 212225040356

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = {
    'CGPA': [6.5, 7.2, 8.1, 5.9, 9.0, 6.8, 7.9, 8.5],
    'Aptitude': [70, 75, 85, 60, 90, 72, 88, 92],
    'Placement': [0, 1, 1, 0, 1, 0, 1, 1]   # 1 = Placed, 0 = Not Placed
}

df = pd.DataFrame(data)
df

X = df[['CGPA', 'Aptitude']]   # Independent variables
y = df['Placement']           # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Example: CGPA = 7.5, Aptitude = 80
new_student = [[7.5, 80]]
result = model.predict(new_student)

if result[0] == 1:
    print("Student is Placed")
else:
    print("Student is Not Placed")
*/
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)

0<img width="558" height="202" alt="image" src="https://github.com/user-attachments/assets/8a236fc1-f479-4ac1-99b9-826b06204c4b" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
