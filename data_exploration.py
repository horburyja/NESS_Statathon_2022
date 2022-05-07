# import utility modules
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# load raw data
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')

train_raw_non_numeric = train_raw.select_dtypes(exclude=[np.number])
non_numeric_cols = train_raw_non_numeric.columns.values

train_raw_numeric = train_raw.select_dtypes(include=[np.number])
numeric_cols = train_raw_numeric.columns.values

"""
# generate histograms and display descriptive stats
print("--DISPLAY DESCRIPTIVE STATISTICS--")
for col in numeric_cols:
    hist = train_raw[[col]].hist(bins=100) # keep number of bins at 100
    plt.title("{}_hist".format(col))
    plt.savefig('figs/histograms/{}_hist.png'.format(col))

    print("{} descriptive statistics:\n{}\n".format(col, train_raw[col].describe()))
    """
