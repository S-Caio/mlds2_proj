"""
har_utils.py -- single, self-contained module replacing process_data_utils.py and
process_data_utils_fixed.py.

Pipeline: raw acc/gyro -> filtered time signals -> 128-sample windows (step 64) ->
FFT windows -> 561 HAPT-style features per window.

Everything that was being tuned against the authors' X_train lives in CFG:
  fft_demean : subtract the window mean before the FFT (no effect once the DC bin is excluded)
  bins       : (lo, hi) FFT bins used for the simple frequency statistics
               (mean/std/mad/max/min/sma/energy/iqr/entropy/skewness/kurtosis).
               (1, 65) = one-sided spectrum without DC (won the sweep).
               maxInds, meanFreq and band energies always use the fixed bins 1..64.
  entropy    : 'log_energy' | 'shannon_raw' | 'shannon_norm' | 'shannon_abs'
  gyro_hp_hz : None, or cutoff (Hz) of a low-pass that is subtracted from the gyro signals
  gyro_center: subtract the whole-recording mean from each gyro axis
  window_shift : constant offset (samples) added to every window start (negative = earlier)
  jerk_lag   : 0 = forward difference (jerk[i] = x[i+1]-x[i]); 1 = backward difference
  ar_demean  : subtract the window mean before the Burg AR fit
  noise_hz, noise_order, median_k, zero_phase : noise filter (median kernel <=1 disables it)
Set CFG[...] before calling the pipeline (or change the defaults below).
"""
import math
from math import acos, sqrt

import numpy as np
import pandas as pd
from scipy import fftpack
from scipy.signal import butter, filtfilt, lfilter, medfilt
from scipy.stats import iqr as IQR, skew, kurtosis, entropy, pearsonr

try:
    from statsmodels.robust.scale import mad as median_deviation
except ImportError:                                   # very old statsmodels
    from statsmodels.robust import mad as median_deviation

SAMPLING_FREQ = 50.0
NYQ = SAMPLING_FREQ / 2.0
WINDOW = 128
STEP = 64

CFG = dict(fft_demean=False, bins=(1, 65), entropy="log_energy", gyro_hp_hz=0.1, gyro_center=False,
           window_shift=-2, jerk_lag=0, ar_demean=False,
           noise_hz=20.0, noise_order=3, median_k=3, zero_phase=False)


# ===========================================================================
# Loading
# ===========================================================================
def import_raw_signals(file_path, columns):
    """acc_expXX_userYY.txt / gyro_expXX_userYY.txt -> DataFrame with the given column names."""
    return pd.DataFrame(np.loadtxt(file_path), columns=columns)


def import_labels_file(path, columns):
    """labels.txt -> DataFrame (int)."""
    return pd.DataFrame(np.loadtxt(path, dtype=int), columns=columns)


def normalize2(number):
    return str(number).zfill(2)


def normalize5(number):
    return str(number).zfill(5)


# ===========================================================================
# Time-domain signals
# ===========================================================================
# Causal (lfilter) on purpose: matched the authors' gravity features better than filtfilt.
_B_NOISE, _A_NOISE = butter(3, 20.0 / NYQ)    # 20 Hz low-pass (noise)
_B_GRAV, _A_GRAV = butter(3, 0.3 / NYQ)       # 0.3 Hz low-pass (gravity)

_TIME_COLUMNS = [
    't_body_acc_X', 't_body_acc_Y', 't_body_acc_Z',
    't_grav_acc_X', 't_grav_acc_Y', 't_grav_acc_Z',
    't_body_acc_jerk_X', 't_body_acc_jerk_Y', 't_body_acc_jerk_Z',
    't_body_gyro_X', 't_body_gyro_Y', 't_body_gyro_Z',
    't_body_gyro_jerk_X', 't_body_gyro_jerk_Y', 't_body_gyro_jerk_Z',
]


def _noise_filter(x):
    b, a = butter(CFG["noise_order"], CFG["noise_hz"] / NYQ)
    return filtfilt(b, a, x) if CFG["zero_phase"] else lfilter(b, a, x)


def butterworth_components(signal, is_acc):
    """(gravity, body) for acceleration; (None, body) for gyro."""
    x = np.asarray(signal, dtype=float)
    if CFG["median_k"] > 1:
        x = medfilt(x, kernel_size=CFG["median_k"])
    total = _noise_filter(x)
    if not is_acc:
        if CFG["gyro_center"]:
            total = total - total.mean()
        hz = CFG["gyro_hp_hz"]
        if hz:
            b, a = butter(3, hz / NYQ)
            return None, total - lfilter(b, a, total)
        return None, total
    gravity = lfilter(_B_GRAV, _A_GRAV, total)
    return gravity, total - gravity


def _lag(x, n):
    """Delay a signal by n samples, padding the start with its first value."""
    return x if n == 0 else np.r_[np.repeat(x[0], n), x[:-n]]


def generate_time_signals(raw_dic):
    """raw_dic[expXX_userYY] (acc_X..gyro_Z) -> time_sig_dic: 15 axial + 5 magnitude signals."""
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
                out['t_body_acc_jerk_' + axis] = _lag(np.diff(body) / dt, CFG["jerk_lag"])
            elif 'gyro' in column:
                _, body = butterworth_components(raw_df[column], is_acc=False)
                out['t_body_gyro_' + axis] = body[:-1]
                out['t_body_gyro_jerk_' + axis] = _lag(np.diff(body) / dt, CFG["jerk_lag"])
        df = pd.DataFrame({c: out[c] for c in _TIME_COLUMNS})
        for i in range(0, 15, 3):
            name = _TIME_COLUMNS[i][:-1] + 'mag'
            df[name] = np.sqrt(df[_TIME_COLUMNS[i]] ** 2 +
                               df[_TIME_COLUMNS[i + 1]] ** 2 +
                               df[_TIME_COLUMNS[i + 2]] ** 2)
        # The authors' tGravityAccMag columns are an exact copy of tBodyAccMag.
        df['t_grav_acc_mag'] = df['t_body_acc_mag']
        time_sig_dic[key] = df
    return time_sig_dic


# ===========================================================================
# Windowing
# ===========================================================================
def Windowing_type_1(time_sig_dic, Labels_Data_Frame, max_activity=12, shift=None):
    """128-sample windows, step 64, fully inside one labelled segment (end point inclusive).
    max_activity=6 -> basic activities only; 12 -> also the 6 postural transitions.
    shift (samples) moves every window start by a constant offset (default CFG['window_shift'])."""
    if shift is None:
        shift = CFG["window_shift"]
    columns = next(iter(time_sig_dic.values())).columns
    window_ID = 0
    out = {}
    rows = np.array(Labels_Data_Frame[Labels_Data_Frame["activity_number_ID"] <= max_activity])
    for line in rows:
        file_key = 'exp' + normalize2(int(line[0])) + '_user' + normalize2(int(line[1]))
        act_ID = int(line[2])
        for cursor in range(int(line[3]), int(line[4]) - (WINDOW - 2), STEP):
            st = max(cursor + shift, 0)
            data = time_sig_dic[file_key].iloc[st:st + WINDOW].to_numpy()
            key = 't_W' + normalize5(window_ID) + '_' + file_key + '_act' + normalize2(act_ID)
            out[key] = pd.DataFrame(data=data, columns=columns)
            window_ID += 1
    return out


# ===========================================================================
# Frequency-domain windows: 9 axial + 4 magnitude signals
# (no gravity signals, no axial gyro-jerk block)
# ===========================================================================
def fast_fourier_transform_one_signal(t_signal):
    x = np.asarray(t_signal, dtype=float)
    if CFG["fft_demean"]:
        x = x - x.mean()
    return np.abs(fftpack.fft(x))


def fast_fourier_transform(t_window):
    f_window = pd.DataFrame()
    for column in t_window.columns:
        if 'grav' in column:
            continue
        if column.startswith('t_body_gyro_jerk_') and not column.endswith('mag'):
            continue
        f_window['f_' + column[2:]] = fast_fourier_transform_one_signal(np.array(t_window[column]))
    return f_window


# ===========================================================================
# Basic statistics (axial: DataFrame of 3 columns -> list of 3; mag: 1 column -> float)
# ===========================================================================
def mean_axial(df): return list(np.asarray(df).mean(axis=0))
def std_axial(df):  return list(np.asarray(df).std(axis=0))
def mad_axial(df):  return list(median_deviation(np.asarray(df), axis=0))
def max_axial(df):  return list(np.asarray(df).max(axis=0))
def min_axial(df):  return list(np.asarray(df).min(axis=0))
def IQR_axial(df):  return list(IQR(np.asarray(df), axis=0))

def mean_mag(c): return float(np.mean(c))
def std_mag(c):  return float(np.std(c))
def mad_mag(c):  return float(median_deviation(np.asarray(c)))
def max_mag(c):  return float(np.max(c))
def min_mag(c):  return float(np.min(c))
def IQR_mag(c):  return float(IQR(np.asarray(c)))


def _entropy_value(x):
    x = np.asarray(x, dtype=float)
    mode = CFG["entropy"]
    if mode == "shannon_abs":                         # scipy entropy of |x| (original repo)
        return float(entropy(np.abs(x)))
    x2 = x ** 2
    if mode == "log_energy":                          # sum(log(x^2))
        return float(np.log(x2[x2 > 0]).sum())
    if mode == "shannon_raw":                         # MATLAB wentropy(x, 'shannon')
        p = x2[x2 > 0]
        return float(-(p * np.log(p)).sum())
    if mode == "shannon_norm":
        p = x2 / x2.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())
    raise ValueError("unknown entropy mode: %s" % mode)


def entropy_axial(df):
    a = np.asarray(df)
    return [_entropy_value(a[:, i]) for i in range(a.shape[1])]


def entropy_mag(mag_column):
    return _entropy_value(np.asarray(mag_column))


# ===========================================================================
# Time-domain specific features
# ===========================================================================
def t_sma_axial(df):
    return float(np.abs(np.asarray(df)).sum()) / 3.0


def t_energy_axial(df):
    a = np.asarray(df)
    return list((a ** 2).sum(axis=0) / float(len(a)))


def t_corr_axial(df):
    a = np.asarray(df)
    return [float(pearsonr(a[:, 0], a[:, 1])[0]),
            float(pearsonr(a[:, 0], a[:, 2])[0]),
            float(pearsonr(a[:, 1], a[:, 2])[0])]


def _arburg2(X, order):
    """Fast Burg AR estimate (from github.com/faroit/freezefx). Returns (a, E, ref); a = [1, a1..a_order]."""
    x = np.asarray(X)
    N = len(x)
    if order == 0:
        raise ValueError("order must be > 0")
    rho = np.sum(np.abs(x) ** 2.) / N
    ef = x.astype(complex)
    eb = x.astype(complex)
    a = np.array([1.0 + 0j])
    ref = np.zeros(order, dtype=complex)
    E = np.zeros(order + 1)
    E[0] = rho
    for m in range(order):
        efp = ef[1:]
        ebp = eb[:-1]
        num = -2. * np.dot(ebp.conj().transpose(), efp)
        den = np.dot(efp.conj().transpose(), efp) + np.dot(ebp, ebp.conj().transpose())
        ref[m] = num / den
        ef = efp + ref[m] * ebp
        eb = ebp + ref[m].conj().transpose() * efp
        a = np.concatenate([a, [0j]])
        a = a + ref[m] * np.flipud(a).conjugate()
        E[m + 1] = np.real((1 - ref[m].conj().transpose() * ref[m])) * E[m]
    return a, E[-1], ref


def _ar_input(x):
    x = np.asarray(x, dtype=float)
    return x - x.mean() if CFG["ar_demean"] else x


def t_arburg_axial(df):
    out = []
    for i in range(3):
        out += list(_arburg2(_ar_input(df.iloc[:, i]), 4)[0][1:].real)
    return out


def t_sma_mag(c):    return float(np.abs(np.asarray(c)).sum())
def t_energy_mag(c): a = np.asarray(c); return float((a ** 2).sum() / float(len(a)))
def t_arburg_mag(c): return list(_arburg2(_ar_input(c), 4)[0][1:].real)


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


# ===========================================================================
# Frequency-domain specific features
# ===========================================================================
_FREQS = fftpack.fftfreq(WINDOW, d=1.0 / SAMPLING_FREQ)
_POS_FREQS = np.fft.rfftfreq(WINDOW, d=1.0 / SAMPLING_FREQ)[1:65]      # bins 1..64 (DC excluded)

_B1 = [(1, 9), (9, 17), (17, 25), (25, 33), (33, 41), (41, 49), (49, 57), (57, 65)]
_B2 = [(1, 17), (17, 31), (31, 49), (49, 65)]
_B3 = [(1, 25), (25, 49)]


def f_sma_axial(df):
    return float((np.abs(np.asarray(df)) / math.sqrt(WINDOW)).sum()) / 3.0


def f_energy_axial(df):
    a = np.asarray(df)
    return list((a ** 2).sum(axis=0) / float(len(a)))


def f_max_Inds_axial(df):
    a = np.asarray(df)
    return [_FREQS[a[1:65, i].argmax() + 1] for i in range(3)]


def f_mean_Freq_axial(df):
    a = np.asarray(df)[1:65, :]
    return [float(np.dot(_POS_FREQS, a[:, i]) / a[:, i].sum()) for i in range(3)]


def f_skewness_and_kurtosis_axial(df):
    a = np.asarray(df)
    out = []
    for i in range(3):
        out += [float(skew(a[:, i])), float(kurtosis(a[:, i]))]
    return out


def _band_energy(sig, band):
    seg = sig[band[0]:band[1]]
    return float((seg ** 2).sum() / float(len(seg)))


def f_all_bands_energy_axial(df):
    a = np.asarray(df)
    out = []
    for i in range(3):
        out += [_band_energy(a[:, i], b) for b in _B1 + _B2 + _B3]
    return out  # 3 x 14 = 42


def f_sma_mag(c):
    a = np.asarray(c)
    return float((np.abs(a) / math.sqrt(len(a))).sum())


def f_energy_mag(c):
    a = np.asarray(c)
    return float((a ** 2).sum() / float(len(a)))


def f_max_Inds_mag(c):
    return float(_FREQS[np.asarray(c)[1:65].argmax() + 1])


def f_mean_Freq_mag(c):
    a = np.asarray(c)[1:65]
    return float(np.dot(_POS_FREQS, a) / a.sum())


def f_skewness_mag(c):  return float(skew(np.asarray(c)))
def f_kurtosis_mag(c):  return float(kurtosis(np.asarray(c)))


def f_axial_features_generation(f_window):
    lo, hi = CFG["bins"]
    axial_df = f_window[f_window.columns[0:9]]                # 3 signals x XYZ
    cols = axial_df.columns
    out = []
    for c in range(0, 9, 3):
        full = axial_df[cols[c:c + 3]]
        d = full.iloc[lo:hi]
        out += (mean_axial(d) + std_axial(d) + mad_axial(d) + max_axial(d) + min_axial(d)
                + [f_sma_axial(d)] + f_energy_axial(d) + IQR_axial(d) + entropy_axial(d)
                + f_max_Inds_axial(full) + f_mean_Freq_axial(full)
                + f_skewness_and_kurtosis_axial(d) + f_all_bands_energy_axial(full))
    return out  # 3 x 79 = 237


def f_mag_features_generation(f_window):
    lo, hi = CFG["bins"]
    out = []
    for col in f_window.columns[-4:]:
        full = f_window[col]
        m = full.iloc[lo:hi]
        out += [mean_mag(m), std_mag(m), mad_mag(m), max_mag(m), min_mag(m), f_sma_mag(m),
                f_energy_mag(m), IQR_mag(m), entropy_mag(m), f_max_Inds_mag(full),
                f_mean_Freq_mag(full), f_skewness_mag(m), f_kurtosis_mag(m)]
    return out  # 4 x 13 = 52


# ===========================================================================
# Angle features (7)
# ===========================================================================
def _angle(v1, v2):
    cos_angle = np.dot(v1, v2) / float(sqrt((v1 ** 2).sum()) * sqrt((v2 ** 2).sum()))
    return float(acos(min(1.0, max(-1.0, cos_angle))))


def angle_features(t_window):
    g = np.array(t_window[['t_grav_acc_X', 't_grav_acc_Y', 't_grav_acc_Z']].mean())
    out = []
    for prefix in ('t_body_acc_', 't_body_acc_jerk_', 't_body_gyro_', 't_body_gyro_jerk_'):
        v = np.array(t_window[[prefix + 'X', prefix + 'Y', prefix + 'Z']].mean())
        out.append(_angle(v, g))
    for axis_vec in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        out.append(_angle(np.array(axis_vec), g))
    return out


# ===========================================================================
# Column names
# ===========================================================================
def time_features_names():
    axial = [['t_body_acc_X', 't_body_acc_Y', 't_body_acc_Z'],
             ['t_grav_acc_X', 't_grav_acc_Y', 't_grav_acc_Z'],
             ['t_body_acc_jerk_X', 't_body_acc_jerk_Y', 't_body_acc_jerk_Z'],
             ['t_body_gyro_X', 't_body_gyro_Y', 't_body_gyro_Z'],
             ['t_body_gyro_Jerk_X', 't_body_gyro_Jerk_Y', 't_body_gyro_Jerk_Z']]
    mags = ['t_body_acc_Mag', 't_grav_acc_Mag', 't_body_acc_jerk_Mag', 't_body_gyro_Mag', 't_body_gyro_Jerk_Mag']
    n1 = ['_mean()', '_std()', '_mad()', '_max()', '_min()']
    n2 = ['_energy()', '_iqr()', '_entropy()']
    n3 = ['_AR1()', '_AR2()', '_AR3()', '_AR4()']
    corr = ['_Corr(X,Y)', '_Corr(X,Z)', '_Corr(Y,Z)']
    names = []
    for cols in axial:
        for f in n1:
            names += [c[:-2] + f + c[-2:] for c in cols]
        names.append(cols[-1][:-2] + '_sma()')
        for f in n2:
            names += [c[:-2] + f + c[-2:] for c in cols]
        for c in cols:
            names += [c[:-2] + f + c[-2:] for f in n3]
        names += [cols[-1][:-2] + f for f in corr]
    for m in mags:
        names += [m + f for f in n1] + [m + '_sma()'] + [m + f for f in n2] + [m + f for f in n3]
    return names  # 265


def frequency_features_names():
    axial = [['f_body_acc_X', 'f_body_acc_Y', 'f_body_acc_Z'],
             ['f_body_acc_Jerk_X', 'f_body_acc_Jerk_Y', 'f_body_acc_Jerk_Z'],
             ['f_body_gyro_X', 'f_body_gyro_Y', 'f_body_gyro_Z']]
    mags = ['f_body_acc_Mag', 'f_body_acc_Jerk_Mag', 'f_body_gyro_Mag', 'f_body_gyro_Jerk_Mag']
    n1 = ['_mean()', '_std()', '_mad()', '_max()', '_min()']
    n2 = ['_energy()', '_iqr()', '_entropy()', '_maxInd()', '_meanFreq()']
    n3 = ['_skewness()', '_kurtosis()']
    n4 = ['_BE[1-8]', '_BE[9-16]', '_BE[17-24]', '_BE[25-32]', '_BE[33-40]', '_BE[41-48]',
          '_BE[49-56]', '_BE[57-64]', '_BE[1-16]', '_BE[17-32]', '_BE[33-48]', '_BE[49-64]',
          '_BE[1-24]', '_BE[25-48]']
    names = []
    for cols in axial:
        for f in n1:
            names += [c[:-2] + f + c[-2:] for c in cols]
        names.append(cols[-1][:-2] + '_sma()')
        for f in n2:
            names += [c[:-2] + f + c[-2:] for c in cols]
        for c in cols:
            names += [c[:-2] + f + c[-2:] for f in n3]
        for c in cols:
            names += [c[:-2] + f + c[-2:] for f in n4]
    for m in mags:
        names += [m + f for f in n1] + [m + '_sma()'] + [m + f for f in n2] + [m + f for f in n3]
    return names  # 289


angle_columns = ['angle%d()' % i for i in range(7)]
ALL_COLUMNS = time_features_names() + frequency_features_names() + angle_columns + ['activity_Id', 'user_Id']


# ===========================================================================
# Dataset pipeline
# ===========================================================================
def Dataset_Generation_PipeLine(t_dic, f_dic):
    """Rows: 561 features + activity_Id + user_Id (563 columns), in sorted window-key order."""
    rows = []
    for t_key, f_key in zip(sorted(t_dic.keys()), sorted(f_dic.keys())):
        t_window, f_window = t_dic[t_key], f_dic[f_key]
        user_id = int(t_key[-8:-6])
        activity_id = int(t_key[-2:])
        rows.append(t_axial_features_generation(t_window) + t_mag_features_generation(t_window)
                    + f_axial_features_generation(f_window) + f_mag_features_generation(f_window)
                    + angle_features(t_window) + [activity_id, user_id])
    return pd.DataFrame(rows, columns=ALL_COLUMNS)


def scale_to_unit_range(X, fit_on=None):
    """Per-feature min-max scaling to [-1, 1]; pass fit_on=X_train to scale test data."""
    ref = np.asarray(X if fit_on is None else fit_on, dtype=float)
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return 2.0 * (np.asarray(X, dtype=float) - lo) / span - 1.0