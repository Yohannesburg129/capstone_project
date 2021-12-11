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


### 4. Modelling
#### Model 4.1 - Logistic Regression

# Check default solver
log_reg1 = LogisticRegression(random_state=42)

log_reg1.solver

# Instantiate and fit basic Logistic Regression model
log_reg1.fit(X_train_scaled, y_train)

# Model performance with train and validation data
log_reg1_train_acc = log_reg1.score(X_train_scaled, y_train)
log_reg1_val_acc = log_reg1.score(X_val_scaled, y_val)

print(f"Train accuracy: {round(log_reg1_train_acc,4)}")
print(f"Validation accuracy: {round(log_reg1_val_acc,4)}")


#### Hyperparameter Optimization
# Values of C to iterate over
C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Empty lists to store accuracy scores
train_acc = []
val_acc = []

# Loop model iteration across all C-values
for C in C_values:
    logit_C = LogisticRegression(random_state=42, max_iter=3000, C=C).fit(X_train_scaled, y_train)
    train_acc.append(logit_C.score(X_train_scaled, y_train))
    val_acc.append(logit_C.score(X_val_scaled, y_val))
    print(f"C={C} completed")
    
# Visualize accuracy scores across C-values
plt.figure(figsize=(10,6))
plt.plot(C_values, train_acc, marker='o', label='Train data')
plt.plot(C_values, val_acc, marker='o', label='Validation data')
plt.xscale('log')
plt.xlabel('Values for C (Regularization Strength)')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy with Different Regularization Strengths')
plt.grid()

plt.show()

# Instantiate model
log_reg2 = LogisticRegression(random_state=42, max_iter=3000, C=0.1).fit(X_train_scaled, y_train)

# Score model against data subsets
log_reg2_train_acc = log_reg2.score(X_train_scaled, y_train)
log_reg2_val_acc = log_reg2.score(X_val_scaled, y_val)
log_reg2_test_acc = log_reg2.score(X_test_scaled, y_test)

print(f"Train Accuracy: {round(log_reg2_train_acc,4)*100}%")
print(f"Validation Accuracy: {round(log_reg2_val_acc,4)*100}%")
print(f"Test Accuracy: {round(log_reg2_test_acc,4)*100}%")

# Model predictions on test data
y_pred = log_reg2.predict(X_test_scaled)

# Call confusion matrix
plot_confusion_matrix(log_reg2, X_test_scaled, y_test, cmap='viridis')

print(classification_report(y_test, y_pred))


##### Logistic Regression - PCA
# Instantiate PCA model
pca_method = PCA()

# Fit PCA to scaled data
pca_method.fit(X_train_scaled)

explained_variance = pca_method.explained_variance_ratio_

# Plot the figure
plt.figure(figsize=(12,7))

plt.plot(range(1, len(explained_variance)+1), explained_variance.cumsum(), marker='.')
plt.axhline(0.8, color='black', linestyle='--')
plt.axvline(700, color='black', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Plotting Cumulative Explained Variance by Principal Components')

plt.show()

# Fit the new PCA parameter
pca_method = PCA(n_components=0.8)

pca_method.fit(X_train_scaled)

# Transform each data subset
X_train_pca = pca_method.transform(X_train_scaled)
X_val_pca = pca_method.transform(X_val_scaled)
X_test_pca = pca_method.transform(X_test_scaled)
pca_method.n_components_

# Values of C to iterate over
C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Empty lists to store accuracy scores
train_acc_pca = []
val_acc_pca = []

# Loop model iteration across all C-values
for C in C_values:
    logit_C = LogisticRegression(random_state=42, max_iter=3000, C=C).fit(X_train_pca, y_train)
    train_acc_pca.append(logit_C.score(X_train_pca, y_train))
    val_acc_pca.append(logit_C.score(X_val_pca, y_val))
    print(f"C={C} completed")
    
# Visualize accuracy scores across C-values
plt.figure(figsize=(10,6))
plt.plot(C_values, train_acc_pca, marker='o', label='Train data')
plt.plot(C_values, val_acc_pca, marker='o', label='Validation data')
plt.xscale('log')
plt.xlabel('Values for C (Regularization Strength)')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy with Different Regularization Strengths')
plt.grid()

plt.show()

# Instantiate model
log_reg3 = LogisticRegression(random_state=42, max_iter=3000, C=0.1).fit(X_train_pca, y_train)

# Score model against data subsets
log_reg3_train_acc = log_reg3.score(X_train_pca, y_train)
log_reg3_val_acc = log_reg3.score(X_val_pca, y_val)
log_reg3_test_acc = log_reg3.score(X_test_pca, y_test)

print(f"Train Accuracy: {round(log_reg3_train_acc,4)*100}%")
print(f"Validation Accuracy: {round(log_reg3_val_acc,4)*100}%")
print(f"Test Accuracy: {round(log_reg3_test_acc,4)*100}%")

pd.DataFrame({'3911 Dimensions': [log_reg2_train_acc, log_reg2_val_acc, log_reg2_test_acc],
                 '710 Dimensions': [log_reg3_train_acc, log_reg3_val_acc, log_reg3_test_acc]},
            index = ['Train Accuracy', 'Validation Accuracy', 'Test Accuracy'])


#### Model 4.2 - K Nearest Neighbor Classifier
# Simple KNN with no hyperparamter optimization
my_knn = KNeighborsClassifier()

my_knn.fit(X_train_scaled, y_train)
print(f"Training Accuracy: {my_knn.score(X_train_scaled, y_train)}")
print(f"Validation Accuracy: {my_knn.score(X_val_scaled, y_val)}")

k_values = np.arange(1, 101, 2)

# Empty lists for metrics
knn_train_acc = []
knn_val_acc = []

# Iterate over different 'k' values
for k in k_values:
    
    my_knn = KNeighborsClassifier(n_neighbors=k)
    my_knn.fit(X_train_scaled, y_train)
    knn_train_acc.append(my_knn.score(X_train_scaled, y_train))
    knn_val_acc.append(my_knn.score(X_val_scaled, y_val))
    
    print(f"{k} nearest neighbors modelled", end='\r')

# Visualize relationship between 'k' and model accuracy
plt.figure(figsize=(12,7))
plt.plot(k_values, knn_train_acc, marker='o', label='Train Data')
plt.plot(k_values, knn_val_acc, marker='o', label='Validation Data')
plt.legend()
plt.xlabel('n_neighbors (k)')
plt.ylabel('Model Accuracy')
plt.title('Relationship between K and Model Accuracy')

plt.show()

# Instantiate KNeighbors Transformer 
knn_transformer = KNeighborsTransformer(mode='distance', n_neighbors=101)
knn_classifier = KNeighborsClassifier(metric='precomputed')

# Put in pipeline
knn_pipeline = Pipeline([('scaler', MinMaxScaler()),
                           ('transformer', knn_transformer),
                           ('classifier', knn_classifier)],
                         )

# Set up the grid search
grid_params = {'classifier__n_neighbors':k_values}

# Instantiate
my_gridsearch = GridSearchCV(knn_pipeline, grid_params, verbose=1, cv=3)

# Fit
my_gridsearch.fit(X_rem, y_rem)

# Find out best parameters
my_gridsearch.best_estimator_

my_knn_optimal = KNeighborsClassifier(n_neighbors=99)

my_knn_optimal.fit(X_train_scaled, y_train)

knn_optimal_train = my_knn_optimal.score(X_train_scaled, y_train)
knn_optimal_val = my_knn_optimal.score(X_val_scaled, y_val)
knn_optimal_test = my_knn_optimal.score(X_test_scaled, y_test)

print("Optimized KNN Model")
print(f"Train Accuracy: {knn_optimal_train}")
print(f"Validation Accuracy: {knn_optimal_val}")
print(f"Test Accuracy: {knn_optimal_test}")

# Model predictions on test data
y_pred_knn = my_knn_optimal.predict(X_test_scaled)

# Call confusion matrix
plot_confusion_matrix(my_knn_optimal, X_test_scaled, y_test, cmap='viridis')

print(classification_report(y_test, y_pred_knn))


#### Model 4.3 - Decision Tree Classifier

