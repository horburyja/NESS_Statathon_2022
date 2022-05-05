# import utility modules
import pandas as pd
import numpy as np

# load raw training data
train_raw = pd.read_csv('data/train.csv')
cols_to_drop = ['id', 'cancel', 'year', 'zip.code']
train_raw = train_raw.drop(cols_to_drop, axis=1) # id and response not used for training, others are irrelevant

print("\n--DISPLAYING DATAFRAME INFO--\n")
print(train_raw.info(), "\n")

# missing data percentage list
print("Missing training data:")

total_pct_missing, colnum = 0, 0
for col in train_raw.columns:
    mean_missing = np.mean(train_raw[col].isnull())
    pct_missing = round(mean_missing * 100, 5) # round to 5 decimal places
    total_pct_missing += pct_missing
    
    print("{}. {} - {}%".format(colnum, col, pct_missing))
    colnum += 1

print("\n{}% missing in total\n".format(total_pct_missing))

# drop observations with over 0 missing values
print("--DROPPING OBSERVATIONS WITH MISSING VALUES--\n")
train_clean = train_raw.dropna()
print(train_clean.shape, "\n")

train_raw_numeric = train_raw.select_dtypes(include=[np.number])
numeric_cols = train_raw_numeric.columns.values
# numeric_cols = np.delete(numeric_cols, ['ni.marital.status']) # must be removed manually

train_clean_non_numeric = train_clean.select_dtypes(exclude=[np.number])
non_numeric_cols = train_clean_non_numeric.columns.values
# non_numeric_cols.append('ni.marital.status')

# drop observations with same key features
print("--DROPPING DUPLICATE OBSERVATIONS--\n")
key = numeric_cols
train_clean = train_clean.drop_duplicates(subset=key)
print(train_clean.shape, "\n")

# remove outliers from numerical data
print("--REMOVING OUTLIERS--\n")
for col in numeric_cols:
    q_low = train_clean[col].quantile(q=0.01)
    q_high = train_clean[col].quantile(q=0.99)
    train_clean = train_clean[(train_clean[col] < q_high) & (train_clean[col] > q_low)]
print(train_clean.shape, "\n")

# verify no inconsistencies in numerical or categorical values
print("Numeric cols [{}]: {}\n".format(len(numeric_cols), numeric_cols))

print("Non numeric cols [{}]: {}\n".format(len(non_numeric_cols),non_numeric_cols))
for col in non_numeric_cols:
    print("Categories of {}: {}".format(col, train_clean[col].unique()))
    print(train_clean[col].value_counts(dropna=False), "\n")

# make every category a dummy variable
print("--CREATING DUMMY VARIABLES--\n")
train_clean = pd.get_dummies(train_clean, non_numeric_cols)
print(train_clean.shape, "\n")

