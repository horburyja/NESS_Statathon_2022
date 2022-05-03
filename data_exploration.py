# import utility modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load raw data
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')
print("\nRaw training df of shape {}\n".format(train_raw.shape))

# missing data percentage list
print("Missing training data:")

types = train_raw.dtypes

total_pct_missing, colnum = 0, 1
for col in train_raw.columns:
    mean_missing = np.mean(train_raw[col].isnull())
    pct_missing = round(mean_missing * 100, 5) # round to 5 decimal places
    total_pct_missing += pct_missing
    
    print('{}. {} - {}% ({})'.format(colnum, col, pct_missing, types[col]))
    colnum += 1

print('\n{}% missing in total\n'.format(total_pct_missing))

train_raw_numeric = train_raw.select_dtypes(include=[np.number])
numeric_cols = train_raw_numeric.columns.values
print("Numeric cols [{}]: {}\n".format(len(numeric_cols), numeric_cols))

train_raw_non_numeric = train_raw.select_dtypes(exclude=[np.number])
non_numeric_cols = train_raw_non_numeric.columns.values
print("Non numeric cols [{}]: {}\n".format(len(non_numeric_cols),non_numeric_cols))

# generate histograms
for col in numeric_cols:
    hist = train_raw[[col]].hist(bins=100) # keep number of bins at 100
    plt.title("{}_hist".format(col))
    plt.savefig('figs/{}_hist.png'.format(col))

