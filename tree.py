import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score



df = pd.read_csv('data/hd_data.csv')
print((df['Heart Disease']).unique())
df = df.dropna()
df['Heart Disease Numeric'] = df['Heart Disease'].replace({'Absence': 0, 'Presence': 1})
X = df[['Age', 'BP', 'Cholesterol', 'Max HR']]
Y = df['Heart Disease Numeric'].astype(int)
print("Y unique is", Y.unique())
print("Y data type is", Y.dtype)

#X = df[['Age', 'BP', 'Cholesterol']]
#Y = df['Max HR']  # target variable

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state=42)
tree = DecisionTreeClassifier()

print("Y train is", Y_train.unique())
print("Y test is", Y_test.unique())

#Training the data on the training dataset
tree.fit(X_train, Y_train)

#Predictions using the test dataset
predictions = tree.predict(X_test)


#Calculating accuracy
accuracy = accuracy_score(Y_test, predictions)
recall = recall_score(Y_test, predictions)
precision = precision_score(Y_test, predictions)
f1 = f1_score(Y_test, predictions)

print("Accuracy: ", accuracy, "\n")
print("Recall: ", recall, "\n")
print("Precision: ", precision, "\n")
print("F1 Score:", f1)

