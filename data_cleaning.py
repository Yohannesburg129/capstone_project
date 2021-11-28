# Import Numpy, Pandas, Matplotlib, and Seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load in the airline dataset
df = pd.read_csv('capstone_airline_reviews 1.csv')

# Check the dataset
df.head()

# Checking the shape of the dataset
df.shape

df.info()

# Check all float columns
df.select_dtypes('float64').head()

# Check all object columns
df.select_dtypes('object').head()

# Import datetime
from datetime import datetime

# Convert the review_date and date_flown columns from object types to datetime formats
df['review_date'] = pd.to_datetime(df['review_date'])

# Check to see whether the review_date data type has been updated to datetime
df.info()

numeric_columns = df.select_dtypes(exclude='object').columns
categorical_columns = df.select_dtypes('object').columns

# Check
list(numeric_columns)

# Check
list(categorical_columns)

df[numeric_columns].describe()

# Distribution of Overall Scores

plt.figure(figsize=(12,8))

# Overall Score Distribution
plt.hist(df['overall'])
plt.xlabel('Overall Scores 1-10')
plt.xticks()
plt.ylabel('Frequency')
plt.title('Distribution of Overall Scores for all Airlines')

plt.show()

# Plot the distribution spread for traveller_type and cabin

columns = ['traveller_type','cabin']

plt.subplots(1,2, figsize=(13,10))

counter = 0

for col in columns:
    counter+=1
    plt.subplot(2,2,counter)
    df[col].value_counts().plot.bar(rot=45)
    plt.title(columns[counter-1])
    
plt.tight_layout()
plt.show()

# Percentage breakdown of traveller_type
df['traveller_type'].value_counts(normalize=True)

# Percentage breakdown of cabin
df['cabin'].value_counts(normalize=True)

# Check
df.head(10)

# Drop the date_flown column
df.drop(columns='date_flown', inplace=True)

# Check to see if date_flown column has been dropped
df.head()

# Check for missing values
df.isna().sum()

df.isna().sum()/len(df)*100

# Drop rows for overall, cabin, seat_comfort, cabin_service, value_for_money and recommended
df.dropna(subset=['overall',
                  'cabin',
                  'seat_comfort',
                  'cabin_service',
                  'value_for_money',
                  'recommended'], inplace=True)

# Check the new shape
df.shape

# Updated check for missing values
df.isna().sum()

# Check percentage of missing values for remaining columns
df.isna().sum()/len(df)*100

#### Food and Beverage Imputation
df['food_bev'].median()

df['food_bev'].mode()

# Finding out the total number of counts for food and beverage
df['food_bev'].value_counts()

# Creating a summary dataframe of Average Score counts
foodbev_score_df = pd.DataFrame(df['food_bev'].value_counts())
foodbev_score_df.columns = ['Count']
foodbev_score_df.head()

# Plotting the Food_Bev Score Distribution
plt.figure(figsize=(10,8))

# Add the mode
plt.axvline(df['food_bev'].mode()[0], color='red', label='mode')
# Add the median
plt.axvline(df['food_bev'].median(), color='blue', label='median')

plt.bar(foodbev_score_df.index, foodbev_score_df['Count'])
plt.xticks(foodbev_score_df.index)
plt.xlabel('Food and Beverage Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Food and Beverage Scores', fontsize=10)
plt.legend()
plt.show()

# Fill in the missing values with the median score of 3.0
df['food_bev'] = df['food_bev'].fillna(3.0)

# Updated visualizatio of the distribution Food and Beverage scores
df['food_bev'].value_counts()

# Creating a summary dataframe of Average Score counts
foodbev_score_df = pd.DataFrame(df['food_bev'].value_counts())
foodbev_score_df.columns = ['Count']
foodbev_score_df.head()

# Plotting the Food_Bev Score Distribution
plt.figure(figsize=(10,8))

# Add the mode
plt.axvline(df['food_bev'].mode()[0], color='red', label='mode')
# Add the median
plt.axvline(df['food_bev'].median(), color='blue', label='median')

plt.bar(foodbev_score_df.index, foodbev_score_df['Count'])
plt.xticks(foodbev_score_df.index)
plt.xlabel('Food and Beverage Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Food and Beverage Scores', fontsize=10)
plt.legend()
plt.show()

#### Entertainment Imputation
df['entertainment'].median()

df['entertainment'].mode()

# Finding out the total number of counts for Entertainment
df['entertainment'].value_counts()

# Creating a summary dataframe of Average Score counts
entertainment_score_df = pd.DataFrame(df['entertainment'].value_counts())
entertainment_score_df.columns = ['Count']
entertainment_score_df.head()

# Plotting the Food_Bev Score Distribution
plt.figure(figsize=(10,8))

# Add the mode
plt.axvline(df['entertainment'].mode()[0], color='red', label='mode')
# Add the median
plt.axvline(df['entertainment'].median(), color='blue', label='median')

plt.bar(entertainment_score_df.index, entertainment_score_df['Count'])
plt.xticks(entertainment_score_df.index)
plt.xlabel('Entertainment Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Entertainment Scores', fontsize=10)
plt.legend()
plt.show()

# Fill in the missing values with the median score of 3.0
df['entertainment'] = df['entertainment'].fillna(3.0)

# Updated visualizatio of the distribution Entertainment scores
df['entertainment'].value_counts()

# Creating a summary dataframe of Average Score counts
entertainment_score_df = pd.DataFrame(df['entertainment'].value_counts())
entertainment_score_df.columns = ['Count']
entertainment_score_df.head()

# Plotting the Food_Bev Score Distribution
plt.figure(figsize=(10,8))

# Add the mode
plt.axvline(df['entertainment'].mode()[0], color='red', label='mode')
# Add the median
plt.axvline(df['entertainment'].median(), color='blue', label='median')

plt.bar(entertainment_score_df.index, entertainment_score_df['Count'])
plt.xticks(entertainment_score_df.index)
plt.xlabel('Entertainment Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Entertainment Scores', fontsize=10)
plt.legend()
plt.show()


#### Ground Service Imputation
