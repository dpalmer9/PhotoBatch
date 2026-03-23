"""Signal/utilities.py
Vendor-agnostic signal utility functions.

These helpers operate on any time-series DataFrame that has a 'Time' column
and on raw NumPy arrays, so they are reusable across future vendor modules.
"""

import numpy as np
from scipy.ndimage import median_filter


def despike_signal(sig_array, window=2001, threshold=5.0):
    """Remove transient spikes from a 1-D signal using a median/MAD approach.

    Parameters
    ----------
    sig_array : np.ndarray
        Raw 1-D signal array.
    window : int, optional
        Size of the rolling median window (must be odd; default 2001).
    threshold : float, optional
        Number of MADs beyond which a sample is considered a spike (default 5.0).

    Returns
    -------
    np.ndarray
        Cleaned signal array (same shape as *sig_array*).
    """
    med = median_filter(sig_array, size=window)
    mad = median_filter(np.abs(sig_array - med), size=window)

    # Prevent division by zero in flat regions
    mad[mad == 0] = np.min(mad[mad > 0]) if np.any(mad > 0) else 1e-6

    outliers = np.abs(sig_array - med) > (threshold * mad)

    cleaned = sig_array.copy()
    if np.any(outliers):
        valid_idx = np.flatnonzero(~outliers)
        outlier_idx = np.flatnonzero(outliers)
        cleaned[outliers] = np.interp(outlier_idx, valid_idx, sig_array[valid_idx])

    return cleaned


def crop_signal(signal_pd, start_time_remove=0, end_time_remove=0):
    """Crop leading and/or trailing time from a signal DataFrame.

    The function always removes rows with negative time values (pre-sync
    artefacts).  Additional time can be trimmed from either end by passing
    non-zero *start_time_remove* / *end_time_remove* values.

    Parameters
    ----------
    signal_pd : pd.DataFrame
        DataFrame containing at least a 'Time' column (float seconds).
    start_time_remove : float, optional
        Seconds to remove from the start of the recording (default 0).
    end_time_remove : float, optional
        Seconds to remove from the end of the recording (default 0).

    Returns
    -------
    pd.DataFrame
        Cropped DataFrame with a reset integer index.
    """
    cut = signal_pd[signal_pd['Time'] >= 0]

    if start_time_remove > 0:
        cut = cut[cut['Time'] >= start_time_remove]

    if end_time_remove > 0:
        max_time = cut['Time'].max()
        cut = cut[cut['Time'] <= (max_time - end_time_remove)]

    return cut.reset_index(drop=True)
