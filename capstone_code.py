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

