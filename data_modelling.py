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


#### `traveller_type`
# Fit the OneHotEncoder to the bookings column and transform
me_traveller_df = pd.DataFrame(me_airlines['traveller_type'])
ohe_me_traveller_df = ohe.fit_transform(me_traveller_df)

ohe_me_traveller_df

# Convert the sparse matrix to a dense matrix
ohe_me_traveller_dense = ohe_me_traveller_df.toarray()

me_traveller_df = pd.DataFrame(ohe_me_traveller_dense, columns=ohe.categories_[0]).astype('int')

# Inspect
me_traveller_df.head()

#Add data_frames together
me_airlines = pd.concat([me_airlines, me_traveller_df.set_index(me_airlines.index)], axis=1)
me_airlines.drop(columns='traveller_type', inplace=True)
# Check
me_airlines.head(2)


#### `cabin`
# Fit the OneHotEncoder to the bookings column and transform
me_cabin_df = pd.DataFrame(me_airlines['cabin'])
ohe_me_cabin_df = ohe.fit_transform(me_cabin_df)

ohe_me_cabin_df

# Convert the sparse matrix to a dense matrix
ohe_me_cabin_dense = ohe_me_cabin_df.toarray()

me_cabin_df = pd.DataFrame(ohe_me_cabin_dense, columns=ohe.categories_[0]).astype('int')

# Inspect
me_cabin_df.head()

me_airlines = pd.concat([me_airlines, me_cabin_df.set_index(me_airlines.index)], axis=1)

# Drop old cabin column
me_airlines.drop(columns='cabin', inplace=True)

# Check
me_airlines.head()


#### `origin`
# Fit the OneHotEncoder to the bookings column and transform
me_origin_df = pd.DataFrame(me_airlines['origin'])
ohe_me_origin_df = ohe.fit_transform(me_origin_df)

ohe_me_origin_df

# Convert the sparse matrix to a dense matrix
ohe_me_origin_dense = ohe_me_origin_df.toarray()

me_origin_df = pd.DataFrame(ohe_me_origin_dense, columns=ohe.categories_[0]).astype('int')

# Inspect
me_origin_df.head()
me_airlines = pd.concat([me_airlines, me_origin_df.set_index(me_airlines.index)], axis=1)

# Drop old cabin column
me_airlines.drop(columns='origin', inplace=True)

# Check
me_airlines.head()


#### `destination`
# Fit the OneHotEncoder to the bookings column and transform
me_destination_df = pd.DataFrame(me_airlines['destination'])
ohe_me_destination_df = ohe.fit_transform(me_destination_df)

ohe_me_destination_df
# Convert the sparse matrix to a dense matrix
ohe_me_destination_dense = ohe_me_destination_df.toarray()

me_destination_df = pd.DataFrame(ohe_me_destination_dense, columns=ohe.categories_[0]).astype('int')

# Inspect
me_destination_df.head()

me_airlines = pd.concat([me_airlines, me_destination_df.set_index(me_airlines.index)], axis=1)

# Drop old cabin column
me_airlines.drop(columns='destination', inplace=True)

# Check
me_airlines.head()
me_airlines.shape


##### `CountVectorizer()`
X1 = me_airlines['customer_review']
y1 = me_airlines['recommended']

# Perform a train-test split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, 
                                                    test_size=0.33, 
                                                    stratify=y1,
                                                    random_state=42)

bagofwords_count = CountVectorizer(stop_words="english")
bagofwords_count.fit(X1_train)

X1_train_transformed = bagofwords_count.transform(X1_train) 
X1_test_transformed = bagofwords_count.transform(X1_test) 

X1_train_transformed.shape

words = bagofwords_count.get_feature_names()
word_counts = X1_train_transformed.toarray().sum(axis=0)

# Fitting a model
logreg = LogisticRegression(C = 0.1)
logreg.fit(X1_train_transformed, y1_train)

# Training and test score
print(f"Train score: {logreg.score(X1_train_transformed, y1_train)}")
print(f"Test score: {logreg.score(X1_test_transformed, y1_test)}")

def plot_coefs(logreg, words):
    coef_df = pd.DataFrame({"coefficient": logreg.coef_[0], "token": words})
    coef_df = coef_df.sort_values("coefficient", ascending=False)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # smallest coefficient -> tokens indicating negative sentiment 
    coef_df.tail(20).set_index("token").plot(kind="bar", rot=45, ax=axs[0], color="red")
    axs[0].set_title("Negative indicators")
 
    
    # largest coefficient -> tokens indicating positive sentiment 
    coef_df.head(20).set_index("token").plot(kind="bar", rot=45, ax=axs[1], color="blue")
    axs[1].set_title("Positive indicators")
    
    
    sns.despine()
    plt.tight_layout()
    plt.show()
    
plot_coefs(logreg, words)

# Create the coefficients dataframe
coef_df = pd.DataFrame({"coefficient": logreg.coef_[0], "token": words})
coef_df = coef_df.sort_values("coefficient", ascending=False)

# Create dataframes for the top 20 positive/negative words
coef_df_pos = coef_df.head(20)
coef_df_neg = coef_df.tail(20)


##### `TfidfVectorizer()`
bagofwords_tfidf = TfidfVectorizer(stop_words="english")
bagofwords_tfidf.fit(X1_train)

X1_train_transformed = bagofwords_tfidf.transform(X1_train) 
X1_test_transformed = bagofwords_tfidf.transform(X1_test) 

X1_train_transformed.shape

words_tfidf = bagofwords_tfidf.get_feature_names()
word_counts_tfidf = X1_train_transformed.toarray().sum(axis=0)

# Fitting a model
logreg_tfidf = LogisticRegression(C = 0.1)
logreg_tfidf.fit(X1_train_transformed, y1_train)

# Training and test score
print(f"Train score: {logreg_tfidf.score(X1_train_transformed, y1_train)}")
print(f"Test score: {logreg_tfidf.score(X1_test_transformed, y1_test)}")

def plot_coefs_tfidf(logreg_tfidf, words_tfidf):
    coef_df_tfidf = pd.DataFrame({"coefficient": logreg_tfidf.coef_[0], "token": words_tfidf})
    coef_df_tfidf = coef_df_tfidf.sort_values("coefficient", ascending=False)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # smallest coefficient -> tokens indicating negative sentiment 
    coef_df_tfidf.tail(20).set_index("token").plot(kind="bar", rot=45, ax=axs[0], color="red")
    axs[0].set_title("Negative indicators")
 
    
    # largest coefficient -> tokens indicating positive sentiment 
    coef_df_tfidf.head(20).set_index("token").plot(kind="bar", rot=45, ax=axs[1], color="blue")
    axs[1].set_title("Positive indicators")
    
    
    sns.despine()
    plt.tight_layout()
    plt.show()
    
plot_coefs_tfidf(logreg_tfidf, words_tfidf)

# Create a dataframe for the positive/negative sentiment words 
# Tfidf
coef_df_tfidf = pd.DataFrame({"coefficient": logreg_tfidf.coef_[0], "token": words_tfidf})
coef_df_tfidf = coef_df_tfidf.sort_values("coefficient", ascending=False)

# Create dataframes for the top 20 positive/negative words
tfidf_df_pos = coef_df_tfidf.head(20)
tfidf_df_neg = coef_df_tfidf.tail(20)



### 3. Data Splitting
# Create X and y variables
X = me_airlines.drop(columns=['overall','seat_comfort','cabin_service','entertainment',
                              'food_bev','ground_service','value_for_money','recommended'])
y = me_airlines['recommended']

# Creating a chunk for the 20% test set which I'll leave to the side
X_rem, X_test, y_rem, y_test = train_test_split(X,
                                               y,
                                               test_size=0.2,
                                               stratify=y,
                                               random_state=42)

# Inspect the dimensions of each data subset
print(X_rem.shape, X_test.shape, y_rem.shape, y_test.shape)

# Instantiate vectorizer
bagofwords = CountVectorizer(stop_words="english",
                            min_df=10,
                            max_features=3000,
                            ngram_range=(1,3))

# Fit vectorizer
bagofwords.fit(X_rem['customer_review'])

# Transform text
rem_transformed = bagofwords.transform(X_rem['customer_review'])
rem_transformed
rem_transformed.toarray()

# Transform the text of the test subset
test_transformed = bagofwords.transform(X_test['customer_review'])
test_transformed

# Convert the sparse matrix to dense for the remainder subset
customer_review_rem_df = pd.DataFrame(columns=bagofwords.get_feature_names(),
                                     data=rem_transformed.toarray())

customer_review_rem_df

# Convert the sparse matrix to dense for the test subset
customer_review_test_df = pd.DataFrame(columns=bagofwords.get_feature_names(),
                                     data=test_transformed.toarray())

customer_review_test_df

X_rem = pd.concat([X_rem, customer_review_rem_df.set_index(X_rem.index)], axis=1)

# Check
X_rem.head()

# Drop the customer review column
X_rem.drop(columns='customer_review', inplace=True)
X_rem.head()
X_test = pd.concat([X_test, customer_review_test_df.set_index(X_test.index)], axis=1)

# Drop customer review column
X_test.drop(columns='customer_review', inplace=True)

# Check
X_test.head()

# Visualize the 30 most common tokens 
words = bagofwords.get_feature_names()
word_counts = rem_transformed.toarray().sum(axis=0)

X_rem_df = pd.DataFrame({"token": words,
                        "occurrences": word_counts})

X_rem_df_sorted = X_rem_df.sort_values("occurrences", ascending=False)

plt.figure(figsize=(15,8))
sns.barplot(data = X_rem_df_sorted.head(30), x="token", y="occurrences", color="blue")
plt.title('Total Count of Tokenized Words')
plt.xticks(rotation=90)
plt.show()

# Split the remainder subset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_rem,
                                                 y_rem,
                                                 test_size=0.3,
                                                  stratify=y_rem,
                                                 random_state=42)
# Check shapes
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)


#### Scaling Data
# Instantiate Scaler
scaler = MinMaxScaler()

# Fit to X_train and transform
X_train_scaled = scaler.fit_transform(X_train)

# Transform other subsets
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)



