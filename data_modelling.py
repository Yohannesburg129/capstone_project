# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Import libraries
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report


# Read in the dataset
airlines = pd.read_csv('airlines_data_cleaned_finalized.csv')
airlines.head()


### 1. Basic EDA
# Convert recommended variable to binary
airlines['recommended'] = np.where(airlines['recommended'] == 'yes', 1, 0)

# Check
airlines.head()

# Check unique number of airlines
airlines['airline'].unique()

# Create a Middle Eastern airline dataframe
me_airlines = airlines[(airlines['airline'] == 'Turkish Airlines') | (airlines['airline'] == 'Qatar Airways')
                      | (airlines['airline'] == 'Egyptair') | (airlines['airline'] == 'Etihad Airways')
                      | (airlines['airline'] == 'Royal Jordanian Airlines') | (airlines['airline'] == 'flydubai')
                      | (airlines['airline'] == 'Saudi Arabian Airlines') | (airlines['airline'] == 'Air Arabia') 
                      | (airlines['airline'] == 'Gulf Air') | (airlines['airline'] == 'Kuwait Airways')
                       | (airlines['airline'] == 'Emirates')
                      ]
# Check shape of new dataframe
me_airlines.shape

me_airlines.to_csv('me_airlines.csv', index=False)

# Check for Class Imbalance
recommended_totals = me_airlines['recommended']

plt.figure(figsize=(10,6))
sns.histplot(x=recommended_totals, discrete=True, shrink=0.8)
plt.xlabel('Yes/No Recommendation')
plt.ylabel('Count')
plt.title("Yes/No Distribution - Non-binarized 'recommended'")

plt.show()


### 2. Feature Engineering
#### `Airline`
# Using OneHotEncoder to encode the airline column
ohe = OneHotEncoder()

# Fit the OneHotEncoder to the bookings column and transform
# Expects a 2D array
me_airlines_df = pd.DataFrame(me_airlines['airline'])
ohe_me_airline = ohe.fit_transform(me_airlines_df)

ohe_me_airline

# Convert the sparse matrix to a dense matrix
ohe_me_airline_dense = ohe_me_airline.toarray()

me_airlines_df = pd.DataFrame(ohe_me_airline_dense, columns=ohe.categories_[0]).astype('int')

# Inspect the airline columns
me_airlines_df.head()

# Add data-frames together
me_airlines = pd.concat([me_airlines, me_airlines_df.set_index(me_airlines.index)], axis=1)

me_airlines.drop(columns='airline', inplace=True)

# Check
me_airlines.head()