#%%
import numpy as np
import pandas as pd
from glob import glob
import os
import matplotlib
matplotlib.use("Agg")                      # headless backend (no display on cluster nodes)
import matplotlib.pyplot as plt
display = print # IPython isn't installed, so I use this

from har_utils import (import_raw_signals, import_labels_file, generate_time_signals,
                       Windowing_type_1, fast_fourier_transform,
                       Dataset_Generation_PipeLine, scale_to_unit_range)

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

from scipy.stats import skew, kurtosis
keys = sorted(f_windows)
col = 'f_body_acc_jerk_mag'
for name, sl in {'bins 1:65': slice(1, 65), 'bins 0:64': slice(0, 64), 'bins 1:128': slice(1, 128)}.items():
    cand = np.array([skew(f_windows[k][col].values[sl]) for k in keys])

#%% ---------------- optional: scale to [-1, 1] like the authors' files ----------------
feature_cols = Dataset.columns[:-2]
Dataset_scaled = Dataset.copy()
Dataset_scaled[feature_cols] = scale_to_unit_range(Dataset[feature_cols].values)
# For a proper train/test split, fit the scaler on train only:
#   scale_to_unit_range(X_test, fit_on=X_train)

#%% ---------------- save results ----------------
os.makedirs("output", exist_ok=True)
Dataset.to_csv("output/har_features.csv", index=False)
Dataset_scaled.to_csv("output/har_features_scaled.csv", index=False)
print("Saved to output/")

#%%
#%% Append to the main script (needs t_windows, Dataset, time_sig_dic, Labels_Data_Frame in memory)
# and har_utils.py (CFG, shift=, ...). Also needs raw_dic for Test 5.
import numpy as np
import pandas as pd
import har_utils as pf

#%% ---------------- their data + window alignment ----------------
theirs = pd.read_csv("dataset/Train/X_train.txt", sep=r"\s+", header=None)
y_tr = pd.read_csv("dataset/Train/y_train.txt", header=None)[0].values
s_tr = pd.read_csv("dataset/Train/subject_id_train.txt", header=None)[0].values
names = [n.strip() for n in pd.read_csv("dataset/features.txt", sep=r"\s+", header=None)[0]]
Xt = theirs.iloc[:, :561].values

keys = sorted(t_windows)                       # same order as Dataset rows
a = Dataset[["user_Id", "activity_Id"]].astype(int).copy()
a["k"] = a.groupby(["user_Id", "activity_Id"]).cumcount()
a["ia"] = np.arange(len(a))
b = pd.DataFrame({"user_Id": s_tr, "activity_Id": y_tr})
b["k"] = b.groupby(["user_Id", "activity_Id"]).cumcount()
b["ib"] = np.arange(len(b))
m = a.merge(b, on=["user_Id", "activity_Id", "k"])
IA, IB = m.ia.values, m.ib.values
print(len(IA), "aligned windows")


def spearman_cols(A, B):
    A = pd.DataFrame(A).rank().values
    B = pd.DataFrame(B).rank().values
    A = (A - A.mean(0)) / A.std(0)
    B = (B - B.mean(0)) / B.std(0)
    return (A * B).mean(0)


def sweep(variants, build, cols, n=1500, seed=0):
    """variants: {label: CFG overrides}; build(t_window) -> 1D list aligned with `cols`
    (slice or list of 0-based feature columns). Returns DataFrame feature x variant of Spearman rho."""
    sel = np.random.default_rng(seed).choice(len(IA), min(n, len(IA)), replace=False)
    B = Xt[IB[sel]][:, cols]
    saved = dict(pf.CFG)
    out = {}
    try:
        for label, cfg in variants.items():
            pf.CFG.clear(); pf.CFG.update(saved); pf.CFG.update(cfg)
            A = np.array([build(t_windows[keys[IA[j]]]) for j in sel], dtype=float)
            out[label] = spearman_cols(A, B)
    finally:
        pf.CFG.clear(); pf.CFG.update(saved)
    return pd.DataFrame(out, index=[names[c] for c in np.arange(561)[cols]])


#%% ---------------- Test 1: is their tGravityAccMag a copy of tBodyAccMag? ----------------
print("their tGravityAccMag == their tBodyAccMag:",
      np.allclose(Xt[:, 213:226], Xt[:, 200:213], atol=1e-6))

#%% ---------------- Test 2: frequency-domain variants (Min, skewness, kurtosis, ...) ----------------
def build_freq(tw):
    fw = pf.fast_fourier_transform(tw)
    return pf.f_axial_features_generation(fw) + pf.f_mag_features_generation(fw)

FREQ = slice(265, 554)
variants = {f"demean={d}|bins={lo}:{hi}": dict(fft_demean=d, bins=(lo, hi))
            for d in (False, True) for lo, hi in ((0, 128), (0, 64), (1, 65), (1, 128))}
res = sweep(variants, build_freq, FREQ)
group = np.array([n.split("-")[1] for n in res.index])
print(res.groupby(group).median().T.round(2).to_string())     # variant x feature type

#%% ---------------- Test 3: entropy definition (time-domain) ----------------
ENT_COLS = [40 * g + 22 + k for g in range(5) for k in range(3)] + [200 + 13 * j + 8 for j in range(5)]

def build_ent(tw):
    ax, mg = tw.iloc[:, :15].values, tw.iloc[:, 15:].values
    return [pf.entropy_mag(ax[:, j]) for j in range(15)] + [pf.entropy_mag(mg[:, j]) for j in range(5)]

res_e = sweep({mode: dict(entropy=mode) for mode in ("log_energy", "shannon_raw", "shannon_norm", "shannon_abs")},
              build_ent, ENT_COLS)
print(res_e.round(2).to_string())

#%% ---------------- Test 4: window offset x jerk lag (means of body / jerk / gyro signals) ----------------
MEAN_COLS = [40 * g + k for g in range(5) for k in range(3)]
sub = np.arange(0, len(IA), 4)
B4 = Xt[IB[sub]][:, MEAN_COLS]
groups = ["BodyAcc", "GravityAcc", "BodyAccJerk", "BodyGyro", "BodyGyroJerk"]
for lag in (0, 1):
    pf.CFG["jerk_lag"] = lag
    tsd = pf.generate_time_signals(raw_dic)
    rows = {}
    for s in range(-6, 1):
        tw = pf.Windowing_type_1(tsd, Labels_Data_Frame, max_activity=12, shift=s)
        ks = sorted(tw)
        A = np.array([tw[ks[IA[j]]].iloc[:, :15].mean().values for j in sub])
        rows[f"shift={s:+d}"] = spearman_cols(A, B4).reshape(5, 3).mean(1)
    print(f"jerk_lag = {lag}")
    print(pd.DataFrame(rows, index=groups).round(3).to_string())
pf.CFG["jerk_lag"] = 0

#%% ---------------- Test 5: gyro bias removal (body-gyro mean, std, entropy) ----------------
GYRO_COLS = list(range(120, 123)) + list(range(123, 126)) + list(range(142, 145))
saved_gyro = (pf.CFG["gyro_hp_hz"], pf.CFG["gyro_center"])
for label, cfg in {"none": dict(gyro_hp_hz=None, gyro_center=False),
                   "hp 0.02": dict(gyro_hp_hz=0.02, gyro_center=False),
                   "hp 0.05": dict(gyro_hp_hz=0.05, gyro_center=False),
                   "hp 0.10": dict(gyro_hp_hz=0.10, gyro_center=False),
                   "hp 0.15": dict(gyro_hp_hz=0.15, gyro_center=False),
                   "center only": dict(gyro_hp_hz=None, gyro_center=True)}.items():
    pf.CFG.update(cfg)
    tsd = pf.generate_time_signals(raw_dic)
    tw = pf.Windowing_type_1(tsd, Labels_Data_Frame, max_activity=12)
    ks = sorted(tw)
    A = []
    for j in sub:
        g = tw[ks[IA[j]]].iloc[:, 9:12]                      # t_body_gyro_X/Y/Z
        A.append(np.r_[g.mean().values, g.std().values, [pf.entropy_mag(g.iloc[:, c]) for c in range(3)]])
    print(f"{label:12s}", spearman_cols(np.array(A), Xt[IB[sub]][:, GYRO_COLS]).round(2))
pf.CFG["gyro_hp_hz"], pf.CFG["gyro_center"] = saved_gyro


#%% ---------------- Test 6: Burg AR on the demeaned window? (AR coefficients, axial + magnitude) ----------------
def build_ar(tw):
    out = []
    for g in range(5):
        out += pf.t_arburg_axial(tw.iloc[:, 3 * g:3 * g + 3])
    for j in range(5):
        out += pf.t_arburg_mag(tw.iloc[:, 15 + j])
    return out

AR_COLS = [40 * g + 25 + k for g in range(5) for k in range(12)] + [200 + 13 * j + 9 + k for j in range(5) for k in range(4)]
res_ar = sweep({"ar_demean=False": dict(ar_demean=False), "ar_demean=True": dict(ar_demean=True)}, build_ar, AR_COLS)
labels = ["BodyAcc", "GravAcc", "AccJerk", "BodyGyro", "GyroJerk"]
tab = {c: pd.Series(list(res_ar[c].values[:60].reshape(5, 12).mean(1)) + list(res_ar[c].values[60:].reshape(5, 4).mean(1)),
                    index=[l + " (axial)" for l in labels] + [l + " (mag)" for l in labels]) for c in res_ar}
print(pd.DataFrame(tab).round(3).to_string())

#%% ---------------- Test 7: noise-filter variants x window shift (basic time stats + frequency features) ----------------
def build_basic(tw):
    out = []
    for g in range(5):
        d = tw.iloc[:, 3 * g:3 * g + 3]
        out += pf.mean_axial(d) + pf.std_axial(d) + pf.mad_axial(d) + pf.max_axial(d) + pf.min_axial(d)
    for j in range(5):
        c = tw.iloc[:, 15 + j]
        out += [pf.mean_mag(c), pf.std_mag(c), pf.mad_mag(c), pf.max_mag(c), pf.min_mag(c)]
    return out

BASIC_COLS = [40 * g + k for g in range(5) for k in range(15)] + [200 + 13 * j + k for j in range(5) for k in range(5)]
sub7 = np.arange(0, len(IA), 8)
B_basic, B_freq = Xt[IB[sub7]][:, BASIC_COLS], Xt[IB[sub7]][:, FREQ]
filters = {"baseline": {}, "no median": dict(median_k=1), "order 2": dict(noise_order=2),
           "order 4": dict(noise_order=4), "20->15 Hz": dict(noise_hz=15.0), "zero phase": dict(zero_phase=True)}
saved7 = dict(pf.CFG)
rows_b, rows_f = {}, {}
try:
    for fname, fcfg in filters.items():
        pf.CFG.clear(); pf.CFG.update(saved7); pf.CFG.update(fcfg)
        tsd = pf.generate_time_signals(raw_dic)
        for s in (-3, -2, -1, 0, 1):
            tw = pf.Windowing_type_1(tsd, Labels_Data_Frame, max_activity=12, shift=s)
            ks = sorted(tw)
            Ab = np.array([build_basic(tw[ks[IA[j]]]) for j in sub7], dtype=float)
            Af = np.array([build_freq(tw[ks[IA[j]]]) for j in sub7], dtype=float)
            rows_b[(fname, s)] = np.nanmedian(spearman_cols(Ab, B_basic))
            rows_f[(fname, s)] = np.nanmedian(spearman_cols(Af, B_freq))
finally:
    pf.CFG.clear(); pf.CFG.update(saved7)
print("median rho, basic time-domain stats");  print(pd.Series(rows_b).unstack().round(3).to_string())
print("median rho, frequency-domain features"); print(pd.Series(rows_f).unstack().round(3).to_string())