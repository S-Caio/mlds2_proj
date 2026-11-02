#%%
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from IPython.display import display

# unchanged helpers (loading, visualisation) come from your original module
from process_data_utils import (import_raw_signals, import_labels_file, visualize_triaxial_signals)

# everything that changed (filtering, FFT block, features, pipeline) comes from the fixed module
from process_data_utils_fixed import (generate_time_signals, Windowing_type_1, Windowing_type_2,
                                      fast_fourier_transform, Dataset_Generation_PipeLine,
                                      scale_to_unit_range)

plt.style.use('bmh')

#%% ---------------- load raw data (unchanged) ----------------
Raw_data_paths = sorted(glob("dataset/RawData/*"))
assert len(Raw_data_paths) == 123, "expected 61 acc + 61 gyro + labels.txt"

raw_dic = {}
for i in range(61):
    key = Raw_data_paths[i][-16:-4]                       # expXX_userYY
    acc = import_raw_signals(Raw_data_paths[i], ['acc_X', 'acc_Y', 'acc_Z'])
    gyro = import_raw_signals(Raw_data_paths[i + 61], ['gyro_X', 'gyro_Y', 'gyro_Z'])
    raw_dic[key] = pd.concat([acc, gyro], axis=1)

Labels_Data_Frame = import_labels_file(
    Raw_data_paths[-1],
    ['experiment_number_ID', 'user_number_ID', 'activity_number_ID', 'Label_start_point', 'Label_end_point'])

#%% ---------------- time-domain signals (Butterworth instead of FFT masking) ----------------
time_sig_dic = generate_time_signals(raw_dic)

#%% ---------------- windowing ----------------
# Basic activities only: reproduces the authors' basic-activity windows (verified for user 1).
t_dic_win_type_I = Windowing_type_1(time_sig_dic, Labels_Data_Frame, max_activity=6)

# All 12 activities incl. postural transitions: 10,908 windows vs. the authors' 10,929 (close, not identical).
t_dic_win_all = Windowing_type_1(time_sig_dic, Labels_Data_Frame, max_activity=12)

# (Windowing_type_2 = the repository's voting scheme; it does NOT reproduce the authors' windows.)

#%% ---------------- frequency-domain windows + features ----------------
# Pick the window set you need:
t_windows = t_dic_win_all
f_windows = {'f' + k[1:]: w.pipe(fast_fourier_transform) for k, w in t_windows.items()}

Dataset = Dataset_Generation_PipeLine(t_windows, f_windows)
print('Dataset shape:', Dataset.shape)          # (n_windows, 563) = 561 features + activity_Id + user_Id
display(Dataset.head(3))

#%% ---------------- optional: scale to [-1, 1] like the authors' files ----------------
feature_cols = Dataset.columns[:-2]
Dataset_scaled = Dataset.copy()
Dataset_scaled[feature_cols] = scale_to_unit_range(Dataset[feature_cols].values)
# For a proper train/test split, fit the scaler on train only:
#   scale_to_unit_range(X_test, fit_on=X_train)

#%%

Dataset_scaled.to_csv("created_dataset.csv", index = False)