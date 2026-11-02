"""
process_data_utils_fixed.py

Drop-in corrections for process_data_utils.py so that the generated features follow the
UCI "Smartphone-Based Recognition of Human Activities and Postural Transitions" (HAPT)
feature set as closely as can be verified (561 features instead of 640).

Unchanged helpers are imported from your existing process_data_utils.py; everything that
needed to change is redefined here. Because Python functions resolve names in the module
where they are defined, every function that *calls* a changed function is redefined here too.

Changes (see the numbered comments below):
  1. Filtering: median(3) -> causal 3rd-order Butterworth low-pass 20 Hz (noise);
                gravity = causal 3rd-order Butterworth low-pass 0.3 Hz; body = total - gravity.
  2. FFT block: no fBodyGyroJerk-XYZ block (the authors have 3 axial + 4 magnitude frequency signals).
  3. meanFreq: uses positive frequencies only (the old code used signed fftfreq values, which cancel).
  4. Time-domain energy: sum of squares / number of values (as in the authors' feature description).
  5. Entropy: log-energy entropy, sum(log(x^2)) (best empirical match to the authors' values).
  6. Windowing_type_1 can optionally window all 12 activities (max_activity=12).
  7. Dataset pipeline builds rows in a list (the old .loc append was O(n^2)).
  8. Optional scale_to_unit_range() since the authors' features are scaled to [-1, 1].
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, medfilt
from scipy import fftpack

# ---- unchanged helpers, reused from your existing module ----
from process_data_utils import (
    import_raw_signals, import_labels_file,
    jerk_one_signal, mag_3_signals,
    normalize2, normalize5, find_activity, Windowing_type_2,
    mean_axial, std_axial, mad_axial, max_axial, min_axial, IQR_axial,
    mean_mag, std_mag, mad_mag, max_mag, min_mag, IQR_mag,
    t_sma_axial, t_arburg_axial, t_corr_axial,
    t_sma_mag, t_arburg_mag,
    f_sma_axial, f_energy_axial, f_max_Inds_axial, f_skewness_and_kurtosis_axial,
    f_all_bands_energy_axial,
    f_sma_mag, f_energy_mag, f_max_Inds_mag, f_skewness_mag, f_kurtosis_mag,
    angle_features, angle_columns, time_features_names,
)

SAMPLING_FREQ = 50.0
NYQ = SAMPLING_FREQ / 2.0

# ---------------------------------------------------------------------------
# 1. Filtering (replaces median + components_selection_one_signal)
# ---------------------------------------------------------------------------
# Causal (lfilter) on purpose: with these filters the gravity features match the
# authors' values (rank corr ~1.0); zero-phase filtfilt matched worse.
_B_NOISE, _A_NOISE = butter(3, 20.0 / NYQ)    # 20 Hz low-pass, noise removal
_B_GRAV, _A_GRAV = butter(3, 0.3 / NYQ)       # 0.3 Hz low-pass, gravity


def butterworth_components(signal, is_acc):
    """Returns (gravity, body) for an acceleration signal, (None, body) for gyro."""
    smoothed = medfilt(np.asarray(signal, dtype=float), kernel_size=3)
    total = lfilter(_B_NOISE, _A_NOISE, smoothed)
    if not is_acc:
        return None, total
    gravity = lfilter(_B_GRAV, _A_GRAV, total)
    return gravity, total - gravity


_TIME_COLUMNS = [
    't_body_acc_X', 't_body_acc_Y', 't_body_acc_Z',
    't_grav_acc_X', 't_grav_acc_Y', 't_grav_acc_Z',
    't_body_acc_jerk_X', 't_body_acc_jerk_Y', 't_body_acc_jerk_Z',
    't_body_gyro_X', 't_body_gyro_Y', 't_body_gyro_Z',
    't_body_gyro_jerk_X', 't_body_gyro_jerk_Y', 't_body_gyro_jerk_Z',
]


def generate_time_signals(raw_dic):
    """raw_dic[expXX_userYY] (acc_X..gyro_Z) -> time_sig_dic with the 15 axial + 5 magnitude signals."""
    time_sig_dic = {}
    dt = 1.0 / SAMPLING_FREQ
    for key in sorted(raw_dic.keys()):
        raw_df = raw_dic[key]
        out = {}
        for column in raw_df.columns:
            axis = column[-1]
            if 'acc' in column:
                grav, body = butterworth_components(raw_df[column], is_acc=True)
                out['t_body_acc_' + axis] = body[:-1]
                out['t_grav_acc_' + axis] = grav[:-1]
                out['t_body_acc_jerk_' + axis] = np.diff(body) / dt
            elif 'gyro' in column:
                _, body = butterworth_components(raw_df[column], is_acc=False)
                out['t_body_gyro_' + axis] = body[:-1]
                out['t_body_gyro_jerk_' + axis] = np.diff(body) / dt
        df = pd.DataFrame({c: out[c] for c in _TIME_COLUMNS})
        for i in range(0, 15, 3):
            name = _TIME_COLUMNS[i][:-1] + 'mag'
            df[name] = np.sqrt(df[_TIME_COLUMNS[i]] ** 2 +
                               df[_TIME_COLUMNS[i + 1]] ** 2 +
                               df[_TIME_COLUMNS[i + 2]] ** 2)
        time_sig_dic[key] = df
    return time_sig_dic


# ---------------------------------------------------------------------------
# 6. Windowing type 1, optionally including postural transitions
# ---------------------------------------------------------------------------
def Windowing_type_1(time_sig_dic, Labels_Data_Frame, max_activity=6):
    """128-sample windows, step 64, fully inside one labelled segment.
    max_activity=6 -> basic activities only (as before); 12 -> also the 6 postural transitions."""
    columns = time_sig_dic['exp01_user01'].columns
    window_ID = 0
    out = {}
    rows = np.array(Labels_Data_Frame[Labels_Data_Frame["activity_number_ID"] <= max_activity])
    for line in rows:
        file_key = 'exp' + normalize2(int(line[0])) + '_user' + normalize2(int(line[1]))
        act_ID = line[2]
        for cursor in range(line[3], line[4] - 127, 64):
            data = np.array(time_sig_dic[file_key].iloc[cursor:cursor + 128])
            key = 't_W' + normalize5(window_ID) + '_' + file_key + '_act' + normalize2(act_ID)
            out[key] = pd.DataFrame(data=data, columns=columns)
            window_ID += 1
    return out


# ---------------------------------------------------------------------------
# 2. FFT: 9 axial + 4 magnitude frequency signals (no gyro-jerk axial block)
# ---------------------------------------------------------------------------
def fast_fourier_transform_one_signal(t_signal):
    return np.abs(fftpack.fft(t_signal))


def fast_fourier_transform(t_window):
    """Frequency-domain window: fBodyAcc-XYZ, fBodyAccJerk-XYZ, fBodyGyro-XYZ (9 columns)
    followed by fBodyAccMag, fBodyAccJerkMag, fBodyGyroMag, fBodyGyroJerkMag (4 columns).
    Gravity signals and the gyro-jerk axial signals are NOT transformed."""
    f_window = pd.DataFrame()
    for column in t_window.columns:
        if 'grav' in column:
            continue
        if column.startswith('t_body_gyro_jerk_') and not column.endswith('mag'):
            continue
        f_window['f_' + column[2:]] = fast_fourier_transform_one_signal(np.array(t_window[column]))
    return f_window


# ---------------------------------------------------------------------------
# 5. Entropy (log-energy)
# ---------------------------------------------------------------------------
def _log_energy_entropy(x):
    x2 = np.asarray(x, dtype=float) ** 2
    out = np.zeros_like(x2)
    m = x2 > 0
    out[m] = np.log(x2[m])
    return float(out.sum())


def entropy_axial(df):
    array = np.array(df)
    return [_log_energy_entropy(array[:, i]) for i in range(array.shape[1])]


def entropy_mag(mag_column):
    return _log_energy_entropy(np.array(mag_column))


# ---------------------------------------------------------------------------
# 4. Time-domain energy = sum of squares / number of values
# ---------------------------------------------------------------------------
def t_energy_axial(df):
    array = np.array(df)
    return list((array ** 2).sum(axis=0) / float(len(array)))


def t_energy_mag(mag_column):
    array = np.array(mag_column)
    return float((array ** 2).sum() / float(len(array)))


# ---------------------------------------------------------------------------
# 3. meanFreq with positive frequencies only (bins 1..64, DC excluded)
# ---------------------------------------------------------------------------
_POS_FREQS = np.fft.rfftfreq(128, d=1.0 / SAMPLING_FREQ)[1:65]


def f_mean_Freq_axial(df):
    array = np.array(df)[1:65, :]
    return [float(np.dot(_POS_FREQS, array[:, i]) / array[:, i].sum()) for i in range(3)]


def f_mean_Freq_mag(mag_column):
    array = np.array(mag_column)[1:65]
    return float(np.dot(_POS_FREQS, array) / array.sum())


# ---------------------------------------------------------------------------
# Feature pipelines that call the changed functions (must be redefined here)
# ---------------------------------------------------------------------------
def t_axial_features_generation(t_window):
    axial_df = t_window[t_window.columns[0:15]]
    cols = axial_df.columns
    out = []
    for c in range(0, 15, 3):
        df = axial_df[cols[c:c + 3]]
        out += (mean_axial(df) + std_axial(df) + mad_axial(df) + max_axial(df) + min_axial(df)
                + [t_sma_axial(df)] + t_energy_axial(df) + IQR_axial(df) + entropy_axial(df)
                + t_arburg_axial(df) + t_corr_axial(df))
    return out  # 5 signals x 40 = 200


def t_mag_features_generation(t_window):
    mags = t_window[t_window.columns[15:]]
    out = []
    for col in mags:
        m = mags[col]
        out += [mean_mag(m), std_mag(m), mad_mag(m), max_mag(m), min_mag(m), t_sma_mag(m),
                t_energy_mag(m), IQR_mag(m), entropy_mag(m)] + t_arburg_mag(m)
    return out  # 5 x 13 = 65


def f_axial_features_generation(f_window):
    axial_df = f_window[f_window.columns[0:9]]          # 3 signals x XYZ
    cols = axial_df.columns
    out = []
    for c in range(0, 9, 3):
        df = axial_df[cols[c:c + 3]]
        out += (mean_axial(df) + std_axial(df) + mad_axial(df) + max_axial(df) + min_axial(df)
                + [f_sma_axial(df)] + f_energy_axial(df) + IQR_axial(df) + entropy_axial(df)
                + f_max_Inds_axial(df) + f_mean_Freq_axial(df)
                + f_skewness_and_kurtosis_axial(df) + f_all_bands_energy_axial(df))
    return out  # 3 x 79 = 237


def f_mag_features_generation(f_window):
    mags = f_window[f_window.columns[-4:]]
    out = []
    for col in mags:
        m = mags[col]
        out += [mean_mag(m), std_mag(m), mad_mag(m), max_mag(m), min_mag(m), f_sma_mag(m),
                f_energy_mag(m), IQR_mag(m), entropy_mag(m), f_max_Inds_mag(m),
                f_mean_Freq_mag(m), f_skewness_mag(m), f_kurtosis_mag(m)]
    return out  # 4 x 13 = 52


def frequency_features_names():
    axial_signals = [
        ['f_body_acc_X', 'f_body_acc_Y', 'f_body_acc_Z'],
        ['f_body_acc_Jerk_X', 'f_body_acc_Jerk_Y', 'f_body_acc_Jerk_Z'],
        ['f_body_gyro_X', 'f_body_gyro_Y', 'f_body_gyro_Z'],
    ]
    mag_signals = ['f_body_acc_Mag', 'f_body_acc_Jerk_Mag', 'f_body_gyro_Mag', 'f_body_gyro_Jerk_Mag']
    n1 = ['_mean()', '_std()', '_mad()', '_max()', '_min()']
    n2 = ['_energy()', '_iqr()', '_entropy()', '_maxInd()', '_meanFreq()']
    n3 = ['_skewness()', '_kurtosis()']
    n4 = ['_BE[1-8]', '_BE[9-16]', '_BE[17-24]', '_BE[25-32]', '_BE[33-40]', '_BE[41-48]',
          '_BE[49-56]', '_BE[57-64]', '_BE[1-16]', '_BE[17-32]', '_BE[33-48]', '_BE[49-64]',
          '_BE[1-24]', '_BE[25-48]']
    names = []
    for columns in axial_signals:
        for f in n1:
            names += [c[:-2] + f + c[-2:] for c in columns]
        names.append(columns[-1][:-2] + '_sma()')
        for f in n2:
            names += [c[:-2] + f + c[-2:] for c in columns]
        for c in columns:
            names += [c[:-2] + f + c[-2:] for f in n3]
        for c in columns:
            names += [c[:-2] + f + c[-2:] for f in n4]
    for c in mag_signals:
        names += [c + f for f in n1] + [c + '_sma()'] + [c + f for f in n2] + [c + f for f in n3]
    return names


ALL_COLUMNS = time_features_names() + frequency_features_names() + angle_columns + ['activity_Id', 'user_Id']


# ---------------------------------------------------------------------------
# 7. Dataset pipeline
# ---------------------------------------------------------------------------
def Dataset_Generation_PipeLine(t_dic, f_dic):
    """Rows: 561 features + activity_Id + user_Id (563 columns)."""
    rows = []
    t_keys, f_keys = sorted(t_dic.keys()), sorted(f_dic.keys())
    for t_key, f_key in zip(t_keys, f_keys):
        t_window, f_window = t_dic[t_key], f_dic[f_key]
        user_id = int(t_key[-8:-6])
        activity_id = int(t_key[-2:])
        rows.append(t_axial_features_generation(t_window) + t_mag_features_generation(t_window)
                    + f_axial_features_generation(f_window) + f_mag_features_generation(f_window)
                    + angle_features(t_window) + [activity_id, user_id])
    return pd.DataFrame(rows, columns=ALL_COLUMNS)


# ---------------------------------------------------------------------------
# 8. Optional scaling to [-1, 1]
# ---------------------------------------------------------------------------
def scale_to_unit_range(X, fit_on=None):
    """Per-feature min-max scaling to [-1, 1]. The authors' exact scaling scheme is not
    documented; fit on the training set (fit_on) and reuse it for test data."""
    ref = np.asarray(X if fit_on is None else fit_on, dtype=float)
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return 2.0 * (np.asarray(X, dtype=float) - lo) / span - 1.0