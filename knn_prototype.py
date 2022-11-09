# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, utils
import pandas as pd
import numpy as np

dataset = pd.read_csv('dengue_cases_log_data.csv')

X = np.array(dataset['Precip_log']).reshape(-1,1)
y = np.array(dataset['Dengue_class']).reshape(-1,1)

n_label = preprocessing.LabelEncoder()
y_trans = n_label.fit_transform(y)
print(y_trans)
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y_trans, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(X_train, y_train)

print("Training size:",len(X_train))

print("Train score:",knn.score(X_train,y_train))

# Predict on dataset which model has not seen before
print(X_test)
print(knn.predict(X_test))
print("Test Size: ",len(X_test))
print("TEST SCORE: ",knn.score(X_test,y_test))
# X_test
# print(knn.fit(X_test,y_test))
print()