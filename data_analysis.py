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