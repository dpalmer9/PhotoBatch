"""IO/sync.py
TTL-based synchronization between a behaviour system and a photometry recorder.

The entry point `abet_doric_synchronize` uses cross-correlation on binary
event vectors to estimate the optimal global time lag, pairs the nearest
TTL pulses within tolerance, then fits a linear regression whose affine
transform aligns the photometry time axis to the behaviour time axis.

This module is intentionally vendor-agnostic at the sync level: the
vendor-specific pulse extraction is done upstream in the IO/Behaviour and
IO/Photometry loaders before their DataFrames are passed here.
"""

import logging

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.linear_model import LinearRegression

from photobatch.exceptions import SynchronizationError

logger = logging.getLogger(__name__)


def abet_doric_synchronize(doric_pd, ttl_pd, abet_pd):
    """Synchronize Doric photometry time to ABET behaviour time via TTL pulses.

    Uses cross-correlation on binary pulse vectors to find the global lag,
    then pairs nearest-neighbour pulses (within 50% of median inter-pulse
    interval) and fits a linear regression to derive the affine transform
    applied to the Doric time axis.

    Parameters
    ----------
    doric_pd : pd.DataFrame
        Photometry DataFrame with a 'Time' column (float seconds).
        **Mutated in-place**: the 'Time' column is replaced with the
        ABET-referenced time.
    ttl_pd : pd.DataFrame
        DataFrame with columns ['Time', 'TTL'] from the photometry recorder.
    abet_pd : pd.DataFrame
        Full ABET data table (as returned by
        :func:`~photobatch.Processing.IO.Behaviour.abet.load_abet_data`).

    Returns
    -------
    pd.DataFrame
        A copy of *doric_pd* with the 'Time' column remapped to ABET time.

    Raises
    ------
    SynchronizationError
        If there are too few TTL pulses to align the streams or too few
        pulse pairs remain for regression.
    """
    try:
        doric_ttl_active = ttl_pd.loc[ttl_pd['TTL'] >= 3.00]
    except KeyError:
        logger.warning("No TTL signal detected in photometry data — skipping synchronization.")
        return doric_pd

    try:
        abet_ttl_active = abet_pd.loc[abet_pd['Item_Name'] == 'TTL #1']
    except KeyError:
        logger.warning("ABET file is missing TTL pulse output — skipping synchronization.")
        return doric_pd

    doric_ttl_times_all = doric_ttl_active['Time'].values.astype(float)

    # Keep only the first sample of each pulse (100 ms tolerance)
    pulse_tol = 0.1
    filtered_doric_ttl = []
    last_time = None
    for t in doric_ttl_times_all:
        if last_time is None or (t - last_time) > pulse_tol:
            filtered_doric_ttl.append(t)
            last_time = t
    doric_ttl_times = np.array(filtered_doric_ttl)

    abet_ttl_times = abet_ttl_active.iloc[:, 0].values.astype(float)
    logger.info("TTL pulse counts — Doric: %d, ABET: %d", len(doric_ttl_times), len(abet_ttl_times))

    if len(doric_ttl_times) < 2 or len(abet_ttl_times) < 2:
        raise SynchronizationError(
            "Fewer than 2 TTL pulses in one or both streams - cannot synchronize."
        )

    # --- Binary event vectors on a shared time grid ---
    all_times  = np.concatenate([doric_ttl_times, abet_ttl_times])
    t_min, t_max = all_times.min(), all_times.max()
    median_ipi = min(np.median(np.diff(doric_ttl_times)),
                     np.median(np.diff(abet_ttl_times)))
    grid_res = median_ipi * 0.1
    grid = np.arange(t_min - median_ipi, t_max + median_ipi, grid_res)

    def _times_to_binary(times, grid):
        vec     = np.zeros(len(grid), dtype=float)
        indices = np.clip(np.searchsorted(grid, times, side='left'),
                          0, len(grid) - 1)
        vec[indices] = 1.0
        return vec

    doric_vec = _times_to_binary(doric_ttl_times, grid)
    abet_vec  = _times_to_binary(abet_ttl_times, grid)

    # --- Cross-correlation ---
    correlation     = signal.correlate(doric_vec, abet_vec, mode='full')
    lags            = np.arange(-(len(abet_vec) - 1), len(doric_vec))
    best_lag_idx    = np.argmax(correlation)
    best_lag_secs   = lags[best_lag_idx] * grid_res
    logger.debug("Cross-correlation optimal lag: %.4f s", best_lag_secs)

    # --- Pair nearest pulses within tolerance ---
    abet_shifted   = abet_ttl_times + best_lag_secs
    pair_tolerance = median_ipi * 0.5

    paired_doric = []
    paired_abet  = []
    used_doric   = set()

    for i, at in enumerate(abet_shifted):
        diffs       = np.abs(doric_ttl_times - at)
        closest_idx = np.argmin(diffs)
        if diffs[closest_idx] <= pair_tolerance and closest_idx not in used_doric:
            paired_doric.append(doric_ttl_times[closest_idx])
            paired_abet.append(abet_ttl_times[i])
            used_doric.add(closest_idx)

    paired_doric = np.array(paired_doric)
    paired_abet  = np.array(paired_abet)
    logger.info("Paired %d of %d Doric / %d ABET TTL pulses", len(paired_doric), len(doric_ttl_times), len(abet_ttl_times))

    if len(paired_doric) < 2:
        raise SynchronizationError(
            "Too few paired TTL pulses for regression - synchronization failed."
        )

    # --- Linear regression ---
    ttl_model = LinearRegression()
    ttl_model.fit(paired_doric.reshape(-1, 1), paired_abet)

    predicted     = ttl_model.predict(paired_doric.reshape(-1, 1))
    residual_err  = np.abs(paired_abet - predicted).mean()
    logger.info("Synchronization residual error (mean absolute): %.4f s", residual_err)
    if residual_err > 0.1:
        logger.warning("High synchronization residual error (%.4f s) — check TTL alignment.", residual_err)

    synced_pd = doric_pd.copy()
    synced_pd['Time'] = (doric_pd['Time'] * ttl_model.coef_[0] +
                         ttl_model.intercept_)

    return synced_pd
