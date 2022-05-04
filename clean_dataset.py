# import utility modules
import pandas as pd
import numpy as np

# load raw training data
train_raw = pd.read_csv('data/train.csv')
cols_to_drop = ['id', 'cancel']
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

# drop observations with all same features
print("--DROPPING DUPLICATE OBSERVATIONS--\n")
train_clean = train_clean.drop_duplicates()
print(train_clean.shape, "\n")

"""
train_raw_numeric = train_raw.select_dtypes(include=[np.number])
numeric_cols = train_raw_numeric.columns.values
print("Numeric cols [{}]: {}\n".format(len(numeric_cols), numeric_cols))
"""

train_raw_non_numeric = train_raw.select_dtypes(exclude=[np.number])
non_numeric_cols = train_raw_non_numeric.columns.values
print("Non numeric cols [{}]: {}\n".format(len(non_numeric_cols),non_numeric_cols))

# check that all categories of categorical data are consistent
for col in non_numeric_cols:
    print("Categories of {}: {}".format(col, train_clean[col].unique()))

