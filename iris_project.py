"""
The goal is to classify iris flowers into one of three species (Setosa, Versicolor, Virginica) based on their petal and sepal measurements. This is a multi-class classification problem
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

"""Load the Dataset"""

# Load the dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target  # Target labels (0 = Setosa, 1 = Versicolor, 2 = Virginica)

# Display basic info
print(df.head())
print(df.info())

"""Visualizing the dataset to understand patterns"""

# Check class distribution
sns.countplot(x=df['species'])
plt.title("Class Distribution of Iris Species")
plt.show()

# Pairplot visualization
sns.pairplot(df, hue='species', palette='Set2')
plt.show()

"""The dataset is already well-structured, so no additional feature engineering is required.

Features (X): Sepal length, Sepal width, Petal length, Petal width.

Target (y): Species (0, 1, 2).
"""

X = df.drop('species', axis=1)
y = df['species']

"""Split the Data"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

"""Select the Model and Train the Model"""

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

"""FInd the Logistic Regression"""

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

"""Evaluate the Decision Tree Model"""

print("Decision Tree Metrics:")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("Classification Report:\n", classification_report(y_test, dt_pred))

"""Evaluate the Logistic Rgression"""

print("Logistic Regression Metrics:")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("Classification Report:\n", classification_report(y_test, lr_pred))

"""Improve the Model"""

from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [3, 5, 10], 'criterion': ['gini', 'entropy']}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters for Decision Tree:", grid_search.best_params_)
