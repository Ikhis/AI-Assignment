#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import pandas as pd
import pandas as pd

# Load the dataset from the local directory as CSV
# The dataset is a CSV file
file_path = r"C:\Users\HP win10\Desktop\Course Moodle\COM7003 Artificial Intelligence\crop_data.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("The first five rows of the dataset:")
print(df.head())


# In[7]:


# Identify missing values using .isnull() function
# Then count using .sum()
missing_values = df.isnull().sum()
print("\nThe missing values for each column in the dataset:")
print(missing_values)


# In[ ]:




