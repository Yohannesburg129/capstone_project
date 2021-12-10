# Import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Read in the airline dataset
df = pd.read_csv('capstone_airline_reviews 1.csv')

# Check the dataset
df.head()

# Checking shape
df.shape

# Check data types
df.info()

# Check all float columns
df.select_dtypes('float64').head()

# Check all object columns
df.select_dtypes('object').head()


# Convert the review_date to datetime 
# Drop date_flown column
df.drop(columns='date_flown', inplace=True)
df['review_date'] = pd.to_datetime(df['review_date'])

# Check the dataset using .info()
df.info()

# Check both numeric and categorical data
numeric_columns = df.select_dtypes(exclude='object').columns
categorical_columns = df.select_dtypes('object').columns

# Check
list(numeric_columns)

# Check
list(categorical_columns)

# Use describe() method
df[numeric_columns].describe()

# Check
df.head()

# Check for missing values
df.isna().sum()

# Percentage of missing values
df.isna().sum()/len(df)*100

# Drop missing rows for variables that contain <10% missing data
df.dropna(subset=['overall',
                  'cabin',
                  'seat_comfort',
                  'cabin_service',
                  'value_for_money',
                  'recommended'], inplace=True)

# Check the new shape
df.shape

# Check percentage of missing values for remaining columns
df.isna().sum()/len(df)*100


##### Food and Beverage Imputation
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


##### Entertainment Imputation
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


##### Ground Service Imputation
df['ground_service'].median()
df['ground_service'].mode()

# Initial visualizatio of the distribution Entertainment scores
df['ground_service'].value_counts()

# Creating a summary dataframe of Average Score counts
groundservice_score_df = pd.DataFrame(df['ground_service'].value_counts())
groundservice_score_df.columns = ['Count']
groundservice_score_df.head()

# Plotting the Food_Bev Score Distribution
plt.figure(figsize=(10,8))

# Add the mode
plt.axvline(df['ground_service'].mode()[0], color='red', label='mode')
# Add the median
plt.axvline(df['ground_service'].median(), color='blue', label='median')

plt.bar(groundservice_score_df.index, groundservice_score_df['Count'])
plt.xticks(groundservice_score_df.index)
plt.xlabel('Ground Service Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Ground Service Scores', fontsize=10)
plt.legend()
plt.show()

# Fill in the missing values with the median score of 3.0
df['ground_service'] = df['ground_service'].fillna(3.0)

# Updated visualizatio of the distribution Entertainment scores
df['ground_service'].value_counts()

# Creating a summary dataframe of Average Score counts
groundservice_score_df = pd.DataFrame(df['ground_service'].value_counts())
groundservice_score_df.columns = ['Count']
groundservice_score_df.head()

# Plotting the Food_Bev Score Distribution
plt.figure(figsize=(10,8))

# Add the mode
plt.axvline(df['ground_service'].mode()[0], color='red', label='mode')
# Add the median
plt.axvline(df['ground_service'].median(), color='blue', label='median')

plt.bar(groundservice_score_df.index, groundservice_score_df['Count'])
plt.xticks(groundservice_score_df.index)
plt.xlabel('Ground Service Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Ground Service Scores', fontsize=10)
plt.legend()
plt.show()

# Check updated dataset
df.isna().sum()

# Make a new copy of the dataset
df2 = df.copy()

# Drop aircraft column
df2.drop(columns=['aircraft'], inplace=True)
# Check
df2.isna().sum()

# Check route column
df2['route'].head()

# Use the .str.split() function
df2['route'].str.split(' to ', expand=True)

# Split string element-by-element in the pandas series using .str accessor
cols_df2 = df2['route'].str.split(' to ', expand=True)

# Make a new copy of the dataframe (so as not to break anything)
df3 = df2.copy()

# Blow away the jobs column with the new origin column
df3['origin'] = cols_df2[0]

# Add new, temporary column called destination 1 
df3['destination 1'] = cols_df2[1]

# Check
df3.head()

# Split string element-by-element in the pandas series using .str accessor
cols1_df = df3['destination 1'].str.split(' via ', expand=True)

# Make a new copy of the dataframe (so as not to break anything)
df4 = df3.copy()

# Blow away the jobs column with the new jobs column
df4['destination'] = cols1_df[0]

# Check new dataframe for origin and destination column
df4.head()

# Drop route and destination 1 columns
df4.drop(columns=['route','destination 1'], inplace=True)

# Check dataframe
df4.head()

# Drop the author column
df4.drop(columns=['author'], inplace=True)

# Check
df4.head()

# Check missing values once again
df4.isna().sum()/len(df4)*100

# Drop missing rows from destination
df4.dropna(subset=['destination'], inplace=True)

# Check updated dataframe
df4.isna().sum()

# Drop missing values from traveller type
df4.dropna(subset=['traveller_type'], inplace=True)

# Check df4 missing values
df4.isna().sum()

# Check for duplicates
df4.duplicated().sum()

# Drop duplicates
df4.drop_duplicates(keep=False, inplace=True)

# Check
df4.duplicated().sum()

df4.shape

# Convert numeric colums from float to integer
df4[['overall','seat_comfort','cabin_service','food_bev','entertainment','ground_service','value_for_money']] = df4[['overall',
                                                                                                                    'seat_comfort',
                                                                                                                    'cabin_service',
                                                                                                                    'food_bev',
                                                                                                                    'entertainment',
                                                                                                                    'ground_service',
                                                                                                                    'value_for_money']].astype('int64')

# Check updated data types
df4.info()

# Check
df4.head()

# Create a Week column
df4['week_of_year'] = df4['review_date'].dt.week

# Create a Month column
df4['month_of_year'] = df4['review_date'].dt.month

# Create an Year column
df4['year'] = df4['review_date'].dt.year

# Create a day of the month column
df4['day_of_month'] = df4['review_date'].dt.day

# Check
df4.head()

# Drop review_date column
df4.drop(columns=['review_date'], inplace = True)

# Check
df4.head()

df4.shape

display(df4['origin'].value_counts().head(20))
display(df4['destination'].value_counts().head(20))

df4['destination'] = df4['destination'].apply(lambda x: x.replace("HKG", "Hong Kong"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("DXB", "Dubai"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("LHR", "London"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("JFK", "New York"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("SYD", "Sydney"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("AKL", "Auckland"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("MEL", "Melbourne"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("BKK", "Bangkok"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("SIN", "Singapore"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("YYZ", "Toronto"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("CDG", "Paris"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("AMS", "Amsterdam"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("ORD", "Chicago"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("LAX", "Los Angeles"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("IST", "Istanbul"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("LGW", "Gatwick"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("ZRH", "Zurich"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("AYT", "Antalya"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("MIA", "Miami"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("BRU", "Brussels"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("FCO", "Rome"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("VIE", "Vienna"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("MAD", "Madrid"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("BNE", "Brisbane"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("MAN", "Manchester"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("PER", "Perth"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("CMB", "Colombo"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("BHX", "Birmingham"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("ATH", "Athens"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("DEL", "Delhi"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("LAS", "Las Vegas"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("LIS", "Lisbon"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("KUL", "Kuala Lumpur"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("PEK", "Beijing"))
df4['destination'] = df4['destination'].apply(lambda x: x.replace("MCO", "Orlando"))

df4['destination'].value_counts().head(15)

df4['origin'] = df4['origin'].apply(lambda x: x.replace("HKG", "Hong Kong"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("DXB", "Dubai"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("LHR", "London"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("JFK", "New York"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("SYD", "Sydney"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("AKL", "Auckland"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("MEL", "Melbourne"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("BKK", "Bangkok"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("SIN", "Singapore"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("YYZ", "Toronto"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("CDG", "Paris"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("AMS", "Amsterdam"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("ORD", "Chicago"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("LAX", "Los Angeles"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("IST", "Istanbul"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("LGW", "Gatwick"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("ZRH", "Zurich"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("AYT", "Antalya"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("MIA", "Miami"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("BRU", "Brussels"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("FCO", "Rome"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("VIE", "Vienna"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("MAD", "Madrid"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("BNE", "Brisbane"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("MAN", "Manchester"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("PER", "Perth"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("CMB", "Colombo"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("BHX", "Birmingham"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("ATH", "Athens"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("DEL", "Delhi"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("LAS", "Las Vegas"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("LIS", "Lisbon"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("KUL", "Kuala Lumpur"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("PEK", "Beijing"))
df4['origin'] = df4['origin'].apply(lambda x: x.replace("MCO", "Orlando"))


# Check origin column
df4['origin'].value_counts().head(15)

# Save the cleaned data
df4.to_csv('airlines_data_cleaned_finalized.csv', index=False)