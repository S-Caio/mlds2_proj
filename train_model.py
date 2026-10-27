#%%
import numpy as np
import pandas as pd

#%%
# Some pre-processing is needed in order to get decent feature names
feature_names = list(pd.read_csv("dataset/features.txt", header = None).iloc[:, 0])
feature_names = [i.strip() for i in feature_names]

# Reading the train dataset
X_train = pd.read_csv("dataset/Train/X_train.txt", sep = " ", header=None)
X_train.columns = feature_names
X_train

y_train = pd.read_csv("dataset/Train/y_train.txt", sep = " ", header = None)
y_train
#%%
