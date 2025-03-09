#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# LOGISTIC REGRESSION MODEL
# TO FIND THE BEST PERFORMING FEATURE IN THE DATASET


# In[19]:


# All required libraries are imported here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from the local directory as CSV
file_path = r"C:\Users\HP win10\Desktop\Course Moodle\COM7003 Artificial Intelligence\crop_data.csv"
crops = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("The first five rows of the dataset:")
print(crops.head())

# Identify missing values using .isnull() function
missing_values = crops.isnull().sum()
print("\nThe missing values for each column in the dataset:")
print(missing_values)

# Check for duplicates
print("\nDuplicates in Dataset:")
print(crops.duplicated().sum())

# Check how many crops are in the dataset
print("\nAll Unique Values in Target Column:")
print(crops.label.unique())
print(crops['label'].value_counts()) # Counts the occurrence of each unique value

# EXPLORATORY DATA ANALYSIS (EDA)
# Summary statistics
print("\nDataset Summary Statistics:")
print(crops.describe())

# Use Label Encoding on target variable to make it suitable for training
label_encoder = LabelEncoder()
crops["label_encoded"] = label_encoder.fit_transform(crops["label"])

# Separate the 'Features' from the 'Target' variables
X = crops.drop(columns=["label", "label_encoded"])  # independent variable (features)
y = crops["label_encoded"]  # dependent variable (Target in numerical values)
y2 = crops["label"]  # dependent variable (Target in categorical values)
print(f"\nThe Independent Variables (Features):\n{X} {X.shape}\n\n"
     f"The Dependent Variables (Numerical Target):\n{y} {y.shape}\n\n"
     f"The Dependent Variables (Categorical Target):\n{y2} {y2.shape}")

print("\nAll Categories and Encoded Values:")
categories = list(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
for category, encoded_value in categories:
    print(f"{category}: {encoded_value}")

# Group the crops based on mean values of the features
crops_grouped = crops.groupby('label').agg({
    'N': 'mean', 'P': 'mean', 'K': 'mean', 'temperature': 'mean',
    'humidity': 'mean', 'ph': 'mean', 'rainfall': 'mean',
}).reset_index()

print("\nGrouped Data by crop (Mean value):")
print(crops_grouped)

# Correlation matrix visualization
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# Distribution between Temperature and pH
plt.figure(figsize=(12, 5))
# First subplot: Temperature distribution
plt.subplot(1, 2, 1)
sns.histplot(crops['temperature'], color="purple", bins=15, kde=True, stat='density', alpha=0.2)
# Second subplot: pH distribution
plt.subplot(1, 2, 2)
sns.histplot(crops['ph'], color="green", bins=15, kde=True, stat='density', alpha=0.2)
plt.show()  # Display the plot

# Checks if the data is balanced using Countplot (Removed 'hue' to simply plot the count)
sns.countplot(y='label', data=crops, palette="plasma_r")
plt.title("Crop Distribution")
plt.show()

# Using Pairplot to show the relationship between features
# Combine X and y2 to create a new DataFrame for pairplot
X_with_y2 = X.copy()
X_with_y2['label'] = y2

# Using sns.pairplot() with the 'label' column for hue
sns.pairplot(X_with_y2, hue='label', palette="Set2")
plt.show()

# Using joinplot to highlight the relationship between rainfall and temperature 
sns.jointplot(x="rainfall", y="humidity", data=crops[(crops['temperature'] < 30) & (crops['rainfall'] > 120)], hue="label")
plt.show()

# Using Boxplot to visualize the effect of pH in crops
sns.set_palette("coolwarm")  # Set the color palette
sns.boxplot(x='ph', y='label', data=crops, hue='label', palette="coolwarm")
# Title and show plot
plt.title("Effect of pH on Crop Types", fontsize=16)
plt.show()


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt

# Using a colorful palette
sns.set_palette("coolwarm")  # Set the color palette

# Boxplot with 'hue' parameter to avoid warning
sns.boxplot(x='ph', y='label', data=crops, hue='label', palette="coolwarm")

# Title and show plot
plt.title("Effect of pH on Crop Types", fontsize=16)
plt.show()


# In[ ]:




