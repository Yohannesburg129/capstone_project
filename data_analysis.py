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


# Create a dataframe for recommended airlines based on count
recommended_per_me_airline = middle_eastern_airlines.groupby(middle_eastern_airlines['airline'])['recommended'].value_counts()

# Convert to dataframe
recommended_per_me_airline = pd.DataFrame(recommended_per_me_airline)

# Create a new Count column
recommended_per_me_airline.columns = ['Count']
recommended_per_me_airline = recommended_per_me_airline.reset_index()

# Display
recommended_per_me_airline.head()

# Pivot to get the no and yes counts on the same row
recommended_per_me_airline = recommended_per_me_airline.pivot(index='airline', 
                                                              columns='recommended', 
                                                              values='Count').reset_index()
# Display
recommended_per_me_airline

# Assign no/yes recommended variables
no_recommended = recommended_per_me_airline['no']
yes_recommended = recommended_per_me_airline['yes']

# calculate percentage yes recommended
recommended_per_me_airline['yes_recommend_pct'] = yes_recommended/(no_recommended+yes_recommended)*100
recommended_per_me_airline


# Plot yes percentage by airline
plt.figure(figsize=(10,8))

plt.barh(recommended_per_me_airline['airline'], recommended_per_me_airline['yes_recommend_pct']);
plt.xlabel('Percentage of Yes Recommendations')
plt.ylabel('Airline')
plt.title('Yes Recommendation % by Middle Eastern Airline')
plt.axvline(50, color='black', linestyle='--')

plt.show()

# Create a dataframe exploring the relationship between recommended and traveller type
me_traveller_recommended = middle_eastern_airlines.groupby(['traveller_type'])['recommended'].value_counts()

# Create a dataframe traveller_recommend breakdown
me_traveller_recommended = pd.DataFrame(me_traveller_recommended)

# Assign a Count column
me_traveller_recommended.columns = ['Count']
me_traveller_recommended = me_traveller_recommended.reset_index()

# Display
me_traveller_recommended


# Pivot to get the positive and negative counts on the same row
me_traveller_recommended = me_traveller_recommended.pivot(index='traveller_type', 
                                                          columns='recommended', 
                                                          values='Count').reset_index()
# Display
me_traveller_recommended


# Assign no/yes recommended variables
no_traveller_recommended = me_traveller_recommended['no']
yes_traveller_recommended = me_traveller_recommended['yes']

# calculate percentage yes recommended
me_traveller_recommended['yes_recommend_pct'] = yes_traveller_recommended/(no_traveller_recommended+yes_traveller_recommended)*100
me_traveller_recommended


# Plot yes percentage by traveller type for Middle Eastern carriers
plt.figure(figsize=(10,8))

plt.barh(me_traveller_recommended['traveller_type'], me_traveller_recommended['yes_recommend_pct']);
plt.xlabel('Percentage of Yes Recommendations')
plt.ylabel('Traveller Type')
plt.title('Yes Recommendation % by Middle Eastern Airline Traveller Type')
plt.axvline(50, color='black', linestyle='--')

plt.show()


# Average overall scores per month
me_average_overall_per_month = middle_eastern_airlines.groupby(['airline',
                                                                'month_of_year'])['overall'].agg('mean').reset_index()

# Assign a new column name
me_average_overall_per_month.rename(columns={'overall': 'Avg Overall Score'}, inplace=True)

# Displau
me_average_overall_per_month


# Create a plot comparing the trend of overall scores for Middle Eastern carriers per month
plt.figure(figsize=(14,8))

sns.lineplot(data=me_average_overall_per_month, x='month_of_year', y='Avg Overall Score', hue='airline')
plt.xlabel('Month', fontsize=15)
plt.ylabel('Average Overall Score', fontsize=15)
plt.title('Midde Eastern Airline Overall Score Trend by Month', fontsize=18)
plt.grid()

plt.show()


