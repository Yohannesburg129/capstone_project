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


##### Cabin Service Trend
# Average cabin service scores per month
me_average_cabinservice_per_month = middle_eastern_airlines.groupby(['airline',
                                                                     'month_of_year'])['cabin_service'].agg('mean').reset_index()

# Assign new column name
me_average_cabinservice_per_month.rename(columns={'cabin_service': 'Avg Cabin Service Score'}, inplace=True)
me_average_cabinservice_per_month

# Create a plot displaying the trend of cabin service scores for Middle Eastern Airlines
plt.figure(figsize=(13,8))

sns.lineplot(data=me_average_cabinservice_per_month, x='month_of_year', y='Avg Cabin Service Score', hue='airline')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Cabin Service Score', fontsize=14)
plt.title('Middle Eastern Airline Cabin Service Score Trend by Month', fontsize=16)
plt.grid()

plt.show()


##### Food and Beverage Trend
# Average food/bev scores per month
me_average_foodbev_per_month = middle_eastern_airlines.groupby(['airline', 
                                                                'month_of_year'])['food_bev'].agg('mean').reset_index()

# Assign new column name
me_average_foodbev_per_month.rename(columns={'food_bev': 'Avg Food_Bev Score'}, inplace=True)
me_average_foodbev_per_month

# Create a plot displaying the trend of food/bev scores for Middle Eastern Airlines
plt.figure(figsize=(13,8))

sns.lineplot(data=me_average_foodbev_per_month, x='month_of_year', y='Avg Food_Bev Score', hue='airline')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Food/Bev Score', fontsize=14)
plt.title('Middle Eastern Airline Food and Beverage Score Trend by Month', fontsize=16)
plt.grid()

plt.show()


##### Ground Service Trend
# Average ground service scores per month
me_average_groundservice_per_month = middle_eastern_airlines.groupby(['airline',
                                                                      'month_of_year'])['ground_service'].agg('mean').reset_index()

# Assign new column name
me_average_groundservice_per_month.rename(columns={'ground_service': 'Avg Ground Service Score'}, inplace=True)
me_average_groundservice_per_month

# Create a plot displaying the trend of ground service scores for Middle Eastern Airlines
plt.figure(figsize=(15,8))

sns.lineplot(data=me_average_groundservice_per_month, x='month_of_year', y='Avg Ground Service Score', hue='airline')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Ground Service Score', fontsize=14)
plt.title('Middle Eastern Ground Service Score Trend by Month', fontsize=18)
plt.grid()

plt.show()


##### Value for Money Trend
# Average value for money scores per month
me_average_vfm_per_month = middle_eastern_airlines.groupby(['airline',
                                                            'month_of_year'])['value_for_money'].agg('mean').reset_index()

# Assign new column name
me_average_vfm_per_month.rename(columns={'value_for_money': 'Avg Value for Money Score'}, inplace=True)
me_average_vfm_per_month

# Create a plot displaying the trend of value for money scores for Middle Eastern Airlines
plt.figure(figsize=(15,8))

sns.lineplot(data=me_average_vfm_per_month, x='month_of_year', y='Avg Value for Money Score', hue='airline')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average VFM Score', fontsize=14)
plt.title('Middle Eastern VFM Score Trend by Month', fontsize=18)
plt.grid()

plt.show()


##### Entertainment Trend
# Average entertainment scores per month
me_average_entertainment_per_month = middle_eastern_airlines.groupby(['airline',
                                                                      'month_of_year'])['entertainment'].agg('mean').reset_index()

# Assign new column name
me_average_entertainment_per_month.rename(columns={'entertainment': 'Avg Entertainment Score'}, inplace=True)
me_average_entertainment_per_month

# Create a plot displaying the trend of entertainment scores for Middle Eastern Airlines
plt.figure(figsize=(13,8))

sns.lineplot(data=me_average_entertainment_per_month, x='month_of_year', y='Avg Entertainment Score', hue='airline')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Entertainment Score', fontsize=14)
plt.title('Middle Eastern Entertainment Trend by Month', fontsize=16)
plt.grid()

plt.show()


##### Seat Comfort Trend
# Average seat comfort scores per month
me_average_seatcomfort_per_month = middle_eastern_airlines.groupby(['airline', 
                                                                    'month_of_year'])['seat_comfort'].agg('mean').reset_index()

# Assign new column name
me_average_seatcomfort_per_month.rename(columns={'seat_comfort': 'Avg Seat Comfort Score'}, inplace=True)
me_average_seatcomfort_per_month

# Create a plot displaying the trend of seat comfort scores for Middle Eastern Airlines
plt.figure(figsize=(13,8))

sns.lineplot(data=me_average_seatcomfort_per_month, x='month_of_year', y='Avg Seat Comfort Score', hue='airline')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Seat Comfort Score', fontsize=14)
plt.title('Middle Eastern Airlines Seat Comfort Score Trend by Month', fontsize=16)
plt.grid()

plt.show()


##### Emirates Analysis
# Create a dataframe for Emirates
emirates = df[(df['airline'] == 'Emirates')]
emirates.head()

# Average all services for Emirates 
emirates_per_month = emirates.groupby(['airline','month_of_year'])['value_for_money',
                                                                                'seat_comfort',
                                                                                'cabin_service',
                                                                                'food_bev',
                                                                                'entertainment',
                                                                                'ground_service'].agg('mean').reset_index()
emirates_per_month.head()


# Create a plot displaying the trend of all scores
plt.figure(figsize=(13,8))

sns.lineplot(data=emirates_per_month, x='month_of_year', y='value_for_money', label='Value for Money')
sns.lineplot(data=emirates_per_month, x='month_of_year', y='seat_comfort', label='Seat Comfort')
sns.lineplot(data=emirates_per_month, x='month_of_year', y='cabin_service', label='Cabin Service')
sns.lineplot(data=emirates_per_month, x='month_of_year', y='food_bev', label='Food/Beverage')
sns.lineplot(data=emirates_per_month, x='month_of_year', y='entertainment', label='Entertainment')
sns.lineplot(data=emirates_per_month, x='month_of_year', y='ground_service', label='Ground Service')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Score', fontsize=14)
plt.title('Emirates Service Trends by Month', fontsize=16)
plt.grid()
plt.show()


# Creating a summary dataframe of Average Score counts
entertainment_score_df = pd.DataFrame(emirates['entertainment'].value_counts())
entertainment_score_df.columns = ['Count']
entertainment_score_df.head()

# Plotting the Entertainment Score Distribution
plt.figure(figsize=(10,6))

plt.bar(entertainment_score_df.index, entertainment_score_df['Count'])
plt.xticks(entertainment_score_df.index)
plt.xlabel('Emirates Entertainment Score', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Emirates Distribution of Entertainment Scores', fontsize=12)

plt.show()


# Creating a summary dataframe of Average Score counts
foodbev_score_df = pd.DataFrame(emirates['food_bev'].value_counts())
foodbev_score_df.columns = ['Count']
foodbev_score_df.head()

# Plotting the Food_Bev Score Distribution
plt.figure(figsize=(10,6))

plt.bar(foodbev_score_df.index, foodbev_score_df['Count'])
plt.xticks(foodbev_score_df.index)
plt.xlabel('Emirates Food/Bev Score', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Emirates Distribution of Food/Bev Scores', fontsize=12)

plt.show()


# Creating a summary dataframe of Average Score counts
cabin_service_score_df = pd.DataFrame(emirates['cabin_service'].value_counts())
cabin_service_score_df.columns = ['Count']
cabin_service_score_df.head()

# Plotting the Cabin Service Score Distribution
plt.figure(figsize=(10,6))

plt.bar(cabin_service_score_df.index, cabin_service_score_df['Count'])
plt.xticks(cabin_service_score_df.index)
plt.xlabel('Emirates Cabin Service Score', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Emirates Distribution of Cabin Service Scores', fontsize=12)

plt.show()


# Creating a summary dataframe of Average Score counts
vfm_score_df = pd.DataFrame(emirates['value_for_money'].value_counts())
vfm_score_df.columns = ['Count']
vfm_score_df.head()

# Plotting the VFM Score Distribution
plt.figure(figsize=(10,6))

plt.bar(vfm_score_df.index, vfm_score_df['Count'])
plt.xticks(vfm_score_df.index)
plt.xlabel('Emirates Value for Money Score', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Emirates Distribution of Value for Money Scores', fontsize=12)

plt.show()


# Emirates Overall Score Distribution
plt.figure(figsize=(10,6))

plt.hist(emirates['overall'])
plt.xlabel('Overall Scores 1-10')
plt.xticks()
plt.ylabel('Frequency')
plt.title('Distribution of Overall Scores for Emirates')

plt.show()


# Create a dataframe exploring the relationship between recommended and traveller type - Emirates
emirates_traveller_recommended = emirates.groupby(['traveller_type'])['recommended'].value_counts()

emirates_traveller_recommended = pd.DataFrame(emirates_traveller_recommended)
emirates_traveller_recommended.columns = ['Count']
emirates_traveller_recommended = emirates_traveller_recommended.reset_index()
emirates_traveller_recommended.head()


# Pivot to get the yes and no counts on the same row
emirates_traveller_recommended = emirates_traveller_recommended.pivot(index='traveller_type',
                                                                      columns='recommended', 
                                                                      values='Count').reset_index()
emirates_traveller_recommended

# Assign yes/no variables
no_traveller_recommended = emirates_traveller_recommended['no']
yes_traveller_recommended = emirates_traveller_recommended['yes']

# Calculate percentage yes recommended
emirates_traveller_recommended['yes_recommend_pct'] = yes_traveller_recommended/(no_traveller_recommended+yes_traveller_recommended)*100
emirates_traveller_recommended


# plot yes percentage by traveller type for Emirates
plt.figure(figsize=(10,6))

plt.barh(emirates_traveller_recommended['traveller_type'], emirates_traveller_recommended['yes_recommend_pct']);
plt.xlabel('Percentage of Yes Recommendations')
plt.ylabel('Traveller Type')
plt.title('Yes Recommendation % by Traveller Type - Emirates')
plt.axvline(50, color='black', linestyle='--')

plt.show()


# Create a new dataframe for number of Yes Recommends per month
emirates_yes_per_month = emirates.groupby(emirates['month_of_year'])['recommended'].value_counts()

emirates_yes_per_month = pd.DataFrame(emirates_yes_per_month)
emirates_yes_per_month.columns = ['Count']
emirates_yes_per_month = emirates_yes_per_month.reset_index()
emirates_yes_per_month.head()

# Pivot to get the yes and no counts on the same row
emirates_yes_per_month = emirates_yes_per_month.pivot(index='month_of_year', columns='recommended', values='Count')
emirates_yes_per_month.head()

# Calculate percentage of yes

emirates_no = emirates_yes_per_month['no']
emirates_yes = emirates_yes_per_month['yes']

emirates_yes_per_month['Fraction Yes Recommend'] = emirates_yes/(emirates_no+emirates_yes)
emirates_yes_per_month.head()

# Plot the results
plt.figure(figsize=(12,6))

plt.bar(emirates_yes_per_month.index, emirates_yes_per_month['Fraction Yes Recommend'])
plt.xticks(emirates_yes_per_month.index)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Fraction of Yes Recommendations', fontsize=14)
plt.title('Fraction of Yes Recommendations per Month - Emirates', fontsize=16)
plt.axhline(0.5, color='black', linestyle='--')

plt.show()


emirates['destination'].value_counts().head(20)

# Plot the top 20 destination count for Emirates
plt.figure(figsize=(10,6))

emirates['destination'].value_counts().head(20).plot(kind='bar')
plt.xlabel('Destinations')
plt.ylabel('Count')
plt.title('Top 20 Destinations - Emirates')

plt.show()


# Create a dataframe where the destination == London
df_london = emirates.loc[emirates['destination'] == 'London']
df_london.head()

df_london.shape

# Create a dataframe exploring the relationship between recommended and traveller type for the London route
emirates_london = df_london.groupby(['traveller_type'])['recommended'].value_counts()

emirates_london = pd.DataFrame(emirates_london)
emirates_london.columns = ['Count']
emirates_london = emirates_london.reset_index()
emirates_london.head()

# Pivot to get the yes and no counts on the same row
emirates_london = emirates_london.pivot(index='traveller_type', 
                                        columns='recommended', 
                                        values='Count').reset_index()
emirates_london

# Assign yes/no variables
no_traveller_recommended = emirates_london['no']
yes_traveller_recommended = emirates_london['yes']

# Calculate percentage yes recommended
emirates_london['yes_recommend_pct'] = yes_traveller_recommended/(no_traveller_recommended+yes_traveller_recommended)*100
emirates_london

# plot yes percentage by traveller type for Southwest
plt.figure(figsize=(10,6))

plt.barh(emirates_london['traveller_type'], emirates_london['yes_recommend_pct']);
plt.xlabel('Percentage of Yes Recommendations')
plt.ylabel('Traveller Type')
plt.title('Yes Recommendation % by Traveller Type - Emirates')
plt.axvline(50, color='black', linestyle='--')

plt.show()


#### Statistical Analysis - Using the Middle Eastern Airlines Dataframe
middle_eastern_airlines.head()

# Traveller Type relation with Recommended
airline_recommend = pd.crosstab(middle_eastern_airlines['airline'], middle_eastern_airlines['recommended'])
airline_recommend

result = stats.chi2_contingency(airline_recommend)
result

# Make copy
me_copy = middle_eastern_airlines.copy()
me_copy.head()

# Convert recommended column into binary
me_copy['recommended'] = np.where(me_copy['recommended'] == 'yes', 1, 0)

# Check
me_copy.head()


# Convert traveller_type and cabin to dummy variables
me_copy = pd.get_dummies(me_copy, columns=['airline', 'traveller_type', 'cabin'], drop_first=True)

# Check
me_copy.head()



