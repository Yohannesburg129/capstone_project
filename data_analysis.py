# Import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

# Load in the cleaned airline dataset
df = pd.read_csv('airlines_data_cleaned_finalized.csv')

# Check the dataset
df.head()

# Checking the shape of the dataset
df.shape

# Create an airline averages dataframe
df_airline_averages = df.groupby('airline')['overall',
                                             'seat_comfort',
                                             'cabin_service',
                                             'food_bev',
                                             'ground_service',
                                             'value_for_money',
                                             'recommended'].mean()
# Check
df_airline_averages = df_airline_averages.sort_values('overall', ascending=False)
display(df_airline_averages)


#### EDA: Middle Eastern Airlines
# Create a dataframe for Middle Eastern airlines
middle_eastern_airlines = df[(df['airline'] == 'Emirates') | (df['airline'] == 'Qatar Airways') 
                             | (df['airline'] == 'Turkish Airlines') | (df['airline'] == 'Etihad Airways')
                             | (df['airline'] == 'Gulf Air') | (df['airline'] == 'Saudi Arabian Airlines')
                                    ]
middle_eastern_airlines.shape

# Save the airline data
middle_eastern_airlines.to_csv('me_airlines.csv', index=False)

# Check the count for each Middle Eastern airline
plt.figure(figsize=(8,8))
middle_eastern_airlines['airline'].value_counts().plot(kind='bar')
plt.xlabel('Middle Eastern Airlines')
plt.ylabel('Count')
plt.title('Total Number of Flights for each Middle Eastern Airline')

plt.show()

# Traveller type breakdown for each airline
middle_eastern_airlines.groupby(['airline','traveller_type'])['traveller_type'].count().unstack().plot(kind="barh", figsize=(10,8))
plt.xlabel('Count')
plt.ylabel('Airline')
plt.title('Traveller Type Count for each Airline')
plt.show()

# Perform an analysis between the relationship of traveller type and value for money score
# Create a dataframe based on traveller type and value for money (vfm)
me_average_vfm_traveller = middle_eastern_airlines.groupby(['airline',
                                                            'traveller_type'])['value_for_money'].agg('mean').reset_index()

# Assign new column name for Average Value for Money (VFM) score
me_average_vfm_traveller.rename(columns={'value_for_money': 'Avg VFM Score'}, inplace=True)

# Display
me_average_vfm_traveller.head()

# Create a boxplot distribution of traveller type and value for money for Middle Eastern airlines
plt.figure(figsize=(10,8))

sns.boxplot(x='traveller_type', y='Avg VFM Score', data=me_average_vfm_traveller)
plt.xlabel('Traveller Type')
plt.ylabel('Value for Money - Middle Eastern Airline Score Count')
plt.title('Boxplot Distribution - Average VFM Scores', fontsize=16)

plt.show()

# Create a boxplot distribution of airlinee and value for money for Middle Eastern airlines
plt.figure(figsize=(12,8))

sns.boxplot(x='airline', y='Avg VFM Score', data=me_average_vfm_traveller)
plt.xlabel('Airline')
plt.ylabel('Value for Money - Middle Eastern Airline Score Count')
plt.title('Boxplot Distribution - Average VFM Scores by Airline')
plt.tight_layout()

plt.show()

