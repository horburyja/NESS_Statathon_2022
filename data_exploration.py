# import utility modules
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# load raw data
train_clean = pd.read_csv('data/train_clean.csv')
cancel = train_clean['cancel']
train_clean = train_clean.drop(['cancel'], axis=1)

train_clean_non_numeric = train_clean.select_dtypes(exclude=[np.number])
non_numeric_cols = train_clean_non_numeric.columns.values

train_clean_numeric = train_clean.select_dtypes(include=[np.number])
numeric_cols = train_clean_numeric.columns.values

# generate histograms
for col in numeric_cols:
    hist = train_clean[[col]].hist(bins=100) # keep number of bins at 100
    plt.title("{}_hist".format(col))
    plt.savefig('figs/histograms/{}_hist.png'.format(col))
