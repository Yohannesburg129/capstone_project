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