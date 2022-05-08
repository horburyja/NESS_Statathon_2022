# import utility modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# import ml tools for prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# load cleaned training data
train_clean = pd.read_csv('data/train_clean.csv')
cancel = train_clean['cancel']
train_clean = train_clean.drop(['cancel'], axis=1)

# train model
train_x, test_x, train_y, test_y = train_test_split(train_clean, cancel, test_size=0.2)







