"""Signal/filter.py
Signal filtering functions for photometry (and future) data streams.

The primary entry point is `signal_filter`, which interpolates the raw
signal to a uniform time grid and applies the selected digital filter.
The `despike_signal` helper is provided via Signal.utilities and called
internally when despike=True.
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

from photobatch.Processing.Signal.utilities import despike_signal


def signal_filter(signal_pd, filter_type='lowpass', filter_name='butterworth',
                  filter_order=4, filter_cutoff=6, despike=True,
                  despike_window=2001, despike_threshold=5.0, cheby_ripple=1.0):
    """Filter photometry (or any two-channel) signal data.

    Steps performed:
    1. Optional despiking of both channels via median/MAD outlier removal.
    2. Resampling to a uniform time grid (handles dropped samples from hardware).
    3. Application of the selected digital filter.

    Parameters
    ----------
    signal_pd : pd.DataFrame
        DataFrame with columns ['Time', 'Control', 'Active'] (all float).
        Rows with negative time are ignored.
    filter_type : str
        'lowpass' / 'low_pass' / 'low'  – IIR low-pass filter.
        'smoothing' / 'smooth' / 'savgol' – Savitzky-Golay smoothing.
        Any other value leaves data unfiltered.
    filter_name : str
        For low-pass filters: 'butterworth', 'bessel', or 'chebychev'.
    filter_order : int
        Filter order (IIR) or window length (Savitzky-Golay).
    filter_cutoff : float
        Cutoff frequency in Hz (low-pass only).
    despike : bool
        Apply median/MAD despiking before filtering (default True).
    despike_window : int
        Window size for the despiking median filter (default 2001).
    despike_threshold : float
        MAD multiplier for spike detection (default 5.0).
    cheby_ripple : float
        Pass-band ripple in dB for Chebyshev type-I filter (default 1.0).

    Returns
    -------
    time_data : np.ndarray
        Uniformly spaced time axis after resampling.
    filtered_f0 : np.ndarray
        Filtered Control (isobestic) channel.
    filtered_f : np.ndarray
        Filtered Active channel.
    sample_frequency : float
        Sampling rate derived from the median sample interval (Hz).
    """
    signal_pd_cut = signal_pd[signal_pd['Time'] >= 0]

    time_data = signal_pd_cut['Time'].to_numpy().astype(float)
    f0_data   = signal_pd_cut['Control'].to_numpy().astype(float)
    f_data    = signal_pd_cut['Active'].to_numpy().astype(float)

    # --- Optional despiking ---
    if despike:
        f0_data = despike_signal(f0_data, window=int(despike_window),
                                 threshold=float(despike_threshold))
        f_data  = despike_signal(f_data, window=int(despike_window),
                                 threshold=float(despike_threshold))

    # --- Resample to uniform time grid ---
    time_diffs = np.diff(time_data)
    median_dt  = np.median(time_diffs)
    uniform_time = np.arange(time_data[0], time_data[-1], median_dt)

    if len(uniform_time) >= 2:
        f0_interp = interp1d(time_data, f0_data, kind='linear',
                             fill_value='extrapolate')
        f_interp  = interp1d(time_data, f_data,  kind='linear',
                             fill_value='extrapolate')
        f0_data   = f0_interp(uniform_time)
        f_data    = f_interp(uniform_time)
        time_data = uniform_time
        print(f"Interpolated to uniform grid: {len(time_data)} samples, "
              f"dt={median_dt:.6f}s")

    sample_frequency = 1.0 / median_dt

    # --- Default: no filter ---
    filtered_f0 = f0_data.copy()
    filtered_f  = f_data.copy()

    filter_type_lower = str(filter_type).lower()

    if filter_type_lower in ('lowpass', 'low_pass', 'low'):
        filt_name = str(filter_name).lower()
        order  = int(filter_order) if filter_order is not None else 4
        cutoff = float(filter_cutoff)

        if filt_name in ('butterworth', 'butter'):
            sos = signal.butter(N=order, Wn=cutoff, btype='lowpass',
                                analog=False, output='sos', fs=sample_frequency)
        elif filt_name in ('bessel', 'bess'):
            sos = signal.bessel(N=order, Wn=cutoff, btype='lowpass',
                                analog=False, output='sos', fs=sample_frequency)
        elif filt_name in ('chebychev', 'cheby', 'cheby1'):
            sos = signal.cheby1(N=order, rp=float(cheby_ripple), Wn=cutoff,
                                btype='lowpass', analog=False, output='sos',
                                fs=sample_frequency)
        else:
            # Unknown name – fall back to Butterworth
            sos = signal.butter(N=order, Wn=cutoff, btype='lowpass',
                                analog=False, output='sos', fs=sample_frequency)

        filtered_f0 = signal.sosfiltfilt(sos, f0_data)
        filtered_f  = signal.sosfiltfilt(sos, f_data)

    elif filter_type_lower in ('smoothing', 'smooth', 'savgol', 'savgolay'):
        window_length = int(filter_order) if filter_order is not None else 5
        if window_length < 3:
            window_length = 3
        if window_length % 2 == 0:
            window_length += 1
        polyorder = min(3, window_length - 1)

        try:
            filtered_f0 = signal.savgol_filter(f0_data, window_length, polyorder)
            filtered_f  = signal.savgol_filter(f_data,  window_length, polyorder)
        except (ValueError, TypeError):
            filtered_f0 = f0_data.copy()
            filtered_f  = f_data.copy()

    # else: unknown filter type – return data as-is

    return time_data, filtered_f0, filtered_f, sample_frequency
