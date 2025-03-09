#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# MULTI-LINEAR REGRESSION MODEL WITH EVALUATION METRICS (MSE, MRSE, R-SQUARE)


# In[4]:


# Import all required libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             mean_squared_error, r2_score)
from sklearn.preprocessing import label_binarize

# Load the dataset from the local directory as CSV
# The dataset is a CSV file
file_path = r"C:\Users\HP win10\Desktop\Course Moodle\COM7003 Artificial Intelligence\crop_data.csv"
crops = pd.read_csv(file_path)

# Use Label Encoding on target variable to make it suitable for training
# This converts the categorical target variable to numerical values 
label_encoder = LabelEncoder()
crops["label_encoded"] = label_encoder.fit_transform(crops["label"])

# Separate the 'Features' from the 'Target' variables
X = crops.drop(columns=["label", "label_encoded"]) # independent variable (features)
y = crops["label_encoded"] # dependent variable (Target in numerical values)
y2 = crops["label"] # dependent variable (Target in categorical values)

# Split the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train A Multiple Linear Regression Model
mul_reg = LinearRegression()  # Initialize the model
mul_reg.fit(X_train_scaled, y_train) # Train the model on the training data

# Display model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {mul_reg.intercept_}")
print(f"Feature Coefficients: {dict(zip(X.columns, mul_reg.coef_))}")

# MAKE PREDICTION
y_pred = mul_reg.predict(X_test_scaled)

# Compare actual vs predicted values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted Crop (First 5 rows):")
print(results.head())

# EVALUATE MODEL PERFORMANCE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")


# In[ ]:




