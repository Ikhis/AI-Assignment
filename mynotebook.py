#!/usr/bin/env python
# coding: utf-8

# # ARTIFICIAL INTELLIGENCE ASSESSMENT CODE: PREDICTIVE MODELING FOR CROP SELECTION BASED ON WEATHER AND SOIL CONTENT

# ## DATA EXPLORATION, PREPROCESSING AND VISUALIZATION

# In[66]:


# STEP 1: IMPORT NECESSARY LIBRARY 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

# STEP 2: LOAD THE CROP DATASET

# Load the dataset from the local directory as CSV
# The dataset is a CSV file
file_path = r"C:\Users\HP win10\Desktop\Course Moodle\COM7003 Artificial Intelligence\crop_data.csv"
crops = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("The first five rows of the dataset:")
print(crops.head())

# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# Identify missing values using .isnull() function
# Then count using .sum()
missing_values = crops.isnull().sum()
print("\nThe missing values for each column in the dataset:")
print(missing_values)

#Check for duplicates
print("\nDuplicates in Dataset:")
print(crops.duplicated().sum())

# Check how many crops are in the dataset
# This will confirm if the crop column is binary or multi-label feature
print("\nAll Unique Values in Target Column:")
print(crops.label.unique())
print(crops['label'].value_counts()) # Counts the occurence of each unique value

# Summary statistics
print("\nDataset Summary Statistics:")
print(crops.describe())

# Use Label Encoding on target variable to make it suitable for training
# This converts the categorical target variable to numerical values 
label_encoder = LabelEncoder()
crops["label_encoded"] = label_encoder.fit_transform(crops["label"])

# Separate the 'Features' from the 'Target' variables
X = crops.drop(columns=["label", "label_encoded"]) # independent variable (features)
y = crops["label_encoded"] # dependent variable (Target in numerical values) 
y2 = crops["label"] # dependent variable (Target in categorical values)
print(f"\nThe Independent Variables (Features):\n{X} {X.shape}\n\n"
     f"The Dependent Variables (Numerical Target):\n{y} {y.shape}\n\n"
     f"The Dependent Variables (Categorical Target):\n{y2} {y2.shape}")
        
print("\nAll Categories and Encoded Values:")
categories = list(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
for category, encoded_value in categories:
    print(f"{category}: {encoded_value}")
    
# Group the crops based on mean values of the features
crops_grouped = crops.groupby('label').agg({'N': 'mean', 'P': 'mean', 'K': 'mean', 'temperature': 'mean', 'humidity': 'mean', 'ph': 'mean', 'rainfall': 'mean' }).reset_index()

print("\nGrouped Data by crop (Mean value):")
print(crops_grouped)

# DATA VISUALIZATION

# Countplot to show the distribution of labels
sns.countplot(y='label', data=crops, hue='label', palette="plasma_r", legend=False) # Create the count plot
plt.xlabel('Count') # Set labels for x axis
plt.ylabel('Label')  # Set labels for y axis
plt.yticks(rotation=0) # Rotate y-axis labels if they are long
plt.title('Distribution of Labels') # Set title (optional)
plt.show() # Output the plot

# Correlation matrix visualization
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# Create a figure with two subplots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1) # First subplot for 'temperature' distribution
sns.histplot(crops['temperature'], color="red", bins=15, kde=True, stat="density", alpha=0.2)
plt.subplot(1, 2, 2) # Second subplot for 'ph' distribution
sns.histplot(crops['ph'], color="green", bins=15, kde=True, stat="density", alpha=0.2)
plt.show() # Show the plot

# A Boxplot highligting the importance of ph 
sns.boxplot(y='label',x='ph',data=crops, palette = "Set2", hue = 'label')
plt.show()

# A pairplot to visualize the distribution of all classes
cropX = crops.drop(columns = "label_encoded")
sns.pairplot(cropX, hue='label')
plt.show()

# A pairplot to assess the effect of humidity and rainfall on crops
sns.jointplot(x="rainfall",y="humidity",data=cropX[(cropX['temperature']<30) & (cropX['rainfall']>120)],hue="label")

# STEP 4: SPLIT THE DATA 
# Split the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining Set Shape: {X_train.shape}, Testing Set Shape: {X_test.shape}")

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a dictionary to store the model performance for each feature
feature_dictn = {}

# Reveal the best predictive feature
for feature in ["N","P","K","temperature","humidity","ph","rainfall",]:
    log_reg = LogisticRegression(solver='lbfgs')
    log_reg.fit(X_train_scaled[:, X.columns.get_loc(feature)].reshape(-1, 1), y_train) 
    y_pred = log_reg.predict(X_test_scaled[:, X.columns.get_loc(feature)].reshape(-1, 1))
    
    # Calculate F1 score, the harmonic mean of precision and recall
    # Could also use balanced_accuracy_score
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    
    # Add feature-f1 score pairs to the dictionary
    feature_dictn[feature] = f1
    print(f"F1-score for {feature}: {f1}")
    
# Humidity produced the best F1 score
# Store in best_predictive_feature dictionary
best_predictive_feature = {"Humidity produced the best F1 score": feature_dictn["humidity"]}
print(best_predictive_feature)

# Plotting the F1 scores of the features
features = list(feature_dictn.keys())
f1_scores = list(feature_dictn.values())

plt.figure(figsize=(10, 6))
plt.barh(features, f1_scores, color='skyblue')
plt.xlabel('F1 Score')
plt.title('F1 Scores for Different Features')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# ## SUPPORT VECTOR MACHINE (SVM) ALGORITHM WITH PARAMETER TUNING

# In[62]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the data
file_path = r"C:\Users\HP win10\Desktop\Course Moodle\COM7003 Artificial Intelligence\crop_data.csv"
crops = pd.read_csv(file_path)

X = crops.drop('label', axis=1)  
y = crops['label']  

# Split the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVC models
svc_linear = SVC(kernel='linear').fit(X_train_scaled, y_train)
print("\nLinear Kernel Accuracy: ", svc_linear.score(X_test_scaled, y_test))

svc_rbf = SVC(kernel='rbf').fit(X_train_scaled, y_train)
print("\nRbf Kernel Accuracy: ", svc_rbf.score(X_test_scaled, y_test))

svc_poly = SVC(kernel='poly').fit(X_train_scaled, y_train)
print("\nPoly Kernel Accuracy: ", svc_poly.score(X_test_scaled, y_test))

# Using GridSearchCV to increase SVC Linear model accuracy by parameter tuning.
parameters = {'C': np.logspace(-3, 2, 6).tolist(), 'gamma': np.logspace(-3, 2, 6).tolist()}

model = GridSearchCV(estimator=SVC(kernel="linear"), param_grid=parameters, n_jobs=-1, cv=4)
model.fit(X_train_scaled, y_train)  # Use X_train_scaled here
print("\nBest Score from GridSearchCV: ", model.best_score_)
print("\nBest Parameters from GridSearchCV: ", model.best_params_)


# In[84]:


get_ipython().system('pip install yellowbrick')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ClassificationReport

# Load the data
file_path = r"C:\Users\HP win10\Desktop\Course Moodle\COM7003 Artificial Intelligence\crop_data.csv"
crops = pd.read_csv(file_path)

X = crops.drop('label', axis=1)  
y = crops['label']  

# Split the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest Models
clf = RandomForestClassifier(max_depth=4,n_estimators=100,random_state=42).fit(X_train, y_train)

print('RF Accuracy on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('RF Accuracy on test set: {:.2f}'.format(clf.score(X_test, y_test)))

#Classification report 
# Let's use yellowbrick for classification report as they are great for visualizing in a tabular format
classes = list(targets.values())
visualizer = ClassificationReport(clf, classes=classes, support=True, cmap="Blues")

visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()


# In[ ]:




