#%%
import numpy as np
import pandas as pd

#%%
# Some pre-processing is needed in order to get decent feature names
feature_names = list(pd.read_csv("dataset/features.txt", header = None).iloc[:, 0])
feature_names = [i.strip() for i in feature_names]

# Reading the design matrix of the train dataset
X_train = pd.read_csv("dataset/Train/X_train.txt", sep = " ", header=None)
X_train.columns = feature_names
X_train

# Reading the target vector of train dataset
y_train = pd.read_csv("dataset/Train/y_train.txt", sep = " ", header = None)
y_train
#%%
import matplotlib.pyplot as plt

raw_data = pd.read_csv("dataset/RawData/acc_exp01_user01.txt", sep = " ", header = None)
plt.plot(np.arange(len(raw_data)), raw_data.iloc[:, 0])