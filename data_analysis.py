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
