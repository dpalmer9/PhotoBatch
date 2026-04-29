"""Process/event.py
Peri-event trial extraction, z-scoring, and summary statistics.

Functions
---------
extract_trial_data    – slice one trial window from photometry arrays.
trial_separator       – build per-trial z-score / delta-F DataFrames.
calculate_max_peak    – mean of per-trial max z-scores.
calculate_auc         – mean AUC across trials.
_generate_event_alias – derive a display label for an event row.
_create_filter_list   – parse filter columns from an event-sheet row.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Event sheet helpers (used by the orchestration layer)
# ---------------------------------------------------------------------------

def _generate_event_alias(event_row):
    """Return a unique display label for an event-sheet row.

    Priority:
    1. Explicit ``event_alias`` column value (non-empty, not NaN).
    2. Auto-generated string: ``event_name [filter1 & filter2 …]``.
    """
    alias = str(event_row.get('event_alias', '')).strip()
    if alias and alias.lower() not in ('nan', 'none', ''):
        return alias

    base = str(event_row.get('event_name', 'Event')).strip()
    try:
        num_filter = int(float(event_row.get('num_filter', 0)))
    except (ValueError, TypeError):
        num_filter = 0

    if num_filter == 0:
        return base

    filter_parts = []
    for i in range(1, num_filter + 1):
        suffix = '' if i == 1 else str(i)
        fname  = str(event_row.get(f'filter_name{suffix}', '')).strip()
        feval  = str(event_row.get(f'filter_eval{suffix}', '')).strip()
        farg   = str(event_row.get(f'filter_arg{suffix}',  '')).strip()
        if fname and feval and farg and fname.lower() not in ('nan', 'none'):
            filter_parts.append(f"{fname}{feval}{farg}")

    return f"{base} [{' & '.join(filter_parts)}]" if filter_parts else base


def _create_filter_list(event_row):
    """Build a list of filter dicts from an event-sheet row.

    Each dict has keys: Type, Name, Group, Arg, Prior, Eval.
    """
    num_filters = event_row.get('num_filter', 0)
    filter_list = []
    for fil in range(1, num_filters + 1):
        if event_row.get('filter_type', None) is not None and fil == 1:
            fil_mod = ''
        else:
            fil_mod = str(fil)
        fil_dict = {
            'Type':  event_row['filter_type'  + fil_mod],
            'Name':  event_row['filter_name'  + fil_mod],
            'Group': str(int(event_row['filter_group' + fil_mod])),
            'Arg':   event_row['filter_arg'   + fil_mod],
            'Prior': event_row['filter_prior' + fil_mod],
            'Eval':  event_row['filter_eval'  + fil_mod],
        }
        filter_list.append(fil_dict)
    return filter_list


# ---------------------------------------------------------------------------
# Trial extraction
# ---------------------------------------------------------------------------

def extract_trial_data(doric_time_array, doric_deltaf_array,
                       start_time, end_time, expected_samples=None):
    """Slice one peri-event window from the photometry time series.

    Parameters
    ----------
    doric_time_array : np.ndarray
        Sorted 1-D time axis from the processed photometry DataFrame.
    doric_deltaf_array : np.ndarray
        Corresponding delta-F values.
    start_time : float
    end_time : float
    expected_samples : int or None
        If provided, the extracted window is resampled to exactly this length
        via linear interpolation so that all trials are the same size.

    Returns
    -------
    trial_time : np.ndarray
    trial_deltaf : np.ndarray
        Both arrays are empty if the window falls outside the data range.
    """
    start_index = np.searchsorted(doric_time_array, start_time, side='left')
    end_index   = np.searchsorted(doric_time_array, end_time,   side='right')

    if (start_index >= len(doric_time_array) or
            end_index <= 0 or start_index >= end_index):
        return np.array([]), np.array([])

    trial_time   = doric_time_array[start_index:end_index]
    trial_deltaf = doric_deltaf_array[start_index:end_index]

    if expected_samples is not None and len(trial_time) != expected_samples:
        target_time  = np.linspace(start_time, end_time, expected_samples)
        trial_deltaf = np.interp(target_time, trial_time, trial_deltaf)
        trial_time   = target_time

    return trial_time, trial_deltaf


# ---------------------------------------------------------------------------
# Trial separation & z-scoring
# ---------------------------------------------------------------------------

def trial_separator(abet_time_list, trial_definition_times, doric_pd,
                    sample_frequency, extra_prior: float = 0,
                    extra_follow: float = 0,
                    trial_normalize='whole', normalize_side='Left',
                    trial_iti_pad: float = 0, center_method='mean',
                    scale_median: bool = False):
    """Extract, normalise, and z-score each peri-event trial.

    Parameters
    ----------
    abet_time_list : pd.DataFrame  columns=['Start_Time','End_Time']
    trial_definition_times : pd.DataFrame  columns=['Start_Time','End_Time']
    doric_pd : pd.DataFrame  columns=['Time','DeltaF']
    sample_frequency : float  Hz
    extra_prior : float  seconds padded before the event.
    extra_follow : float  seconds padded after the event.
    trial_normalize : str
        'whole' – z-score across the full trial window.
        'iti'   – z-score using an inter-trial-interval baseline.
        'prior' – z-score using the padding period before the event.
    normalize_side : str
        For 'prior', use 'Left' / 'Before' or 'Right' / 'After'.
        For 'iti', use 'Start', 'End', or 'Center'. 'Left' / 'Before'
        map to 'End' and 'Right' / 'After' map to 'Start'.
    trial_iti_pad : float  additional ITI padding (seconds).
    center_method : str  'mean' or 'median'.
    scale_median: bool if True, scale MAD by 1.4826 to match std dev for normal distribution.

    Returns
    -------
    partial_dataframe : pd.DataFrame   z-score trial columns only.
    final_dataframe   : pd.DataFrame   time + z-score trial columns.
    partial_deltaf    : pd.DataFrame   delta-F trial columns only.
    final_deltaf      : pd.DataFrame   time + delta-F trial columns.
    """
    left_selection_list  = ['Left', 'Before', 'L', 'l', 'left', 'before', 1]
    iti_end_selection_list = ['End', 'E', 'e', 'end', 'Left', 'Before', 'L', 'l', 'left', 'before', 1]
    iti_start_selection_list = ['Start', 'S', 's', 'start', 'Right', 'After', 'R', 'r', 'right', 'after', 0]
    iti_center_selection_list = ['Center', 'C', 'c', 'center', 'Middle', 'middle', 2]

    doric_time_array   = doric_pd['Time'].values
    doric_deltaf_array = doric_pd['DeltaF'].values

    time_trials    = []
    zscore_trials  = []
    deltaf_trials  = []

    for row in abet_time_list.itertuples():
        index = row.Index
        try:
            start_time  = float(row.Start_Time)
            start_index = np.searchsorted(doric_time_array, start_time, side='left')
            if start_index >= len(doric_time_array):
                print('Trial Start Out of Bounds, Skipping Event')
                continue
        except (IndexError, ValueError, TypeError):
            print('Trial Start Out of Bounds or invalid, Skipping Event')
            continue

        try:
            end_time  = float(row.End_Time)
            end_index = np.searchsorted(doric_time_array, end_time, side='right')
            if end_index < 0 or end_index >= len(doric_time_array):
                print('Trial End Out of Bounds, Skipping Event')
                continue
        except (IndexError, ValueError, TypeError):
            print('Trial End Out of Bounds or invalid, Skipping Event')
            continue

        try:
            length_time = end_time - start_time
            measurements_per_interval = max(1, int(round(length_time * sample_frequency)))
        except (ValueError, TypeError, ArithmeticError):
            measurements_per_interval = 1

        trial_time, trial_deltaf = extract_trial_data(
            doric_time_array, doric_deltaf_array,
            start_time, end_time,
            expected_samples=measurements_per_interval)

        # --- Normalisation baseline ---
        if trial_normalize == 'iti':
            trial_start = abet_time_list.loc[index, 'Start_Time'] + extra_prior

            def_starts = trial_definition_times['Start_Time'].values
            closest_idx = np.searchsorted(def_starts, trial_start) - 1

            if closest_idx >= 0:
                trial_start_window = def_starts[closest_idx]
                if closest_idx > 0:
                    iti_start = trial_definition_times['End_Time'].values[closest_idx - 1]
                else:
                    iti_start = float(doric_time_array[0])  # Start of data if no previous trial
            else:
                iti_start = float(doric_time_array[0])  # Start of data if no previous trial
                trial_start_window = trial_start  # No trial window, use trial start as reference

            iti_end = trial_start_window

            if iti_end <= iti_start:
                iti_data = np.array([], dtype=float)

            else:
                iti_window = float(trial_iti_pad)
                if normalize_side in iti_start_selection_list:
                    baseline_start = iti_start
                    baseline_end = min(iti_start + iti_window, iti_end)
                elif normalize_side in iti_center_selection_list:
                    iti_center = (iti_start + iti_end) / 2.0
                    half_window = iti_window / 2.0
                    baseline_start = max(iti_center - half_window, iti_start)
                    baseline_end = min(iti_center + half_window, iti_end)
                else:
                    baseline_start = max(iti_end - iti_window, iti_start)
                    baseline_end = iti_end
            
            idx_start = np.searchsorted(doric_time_array, baseline_start, side='left')
            idx_end   = np.searchsorted(doric_time_array, baseline_end, side='right')
            iti_data = doric_deltaf_array[idx_start:idx_end]


            if center_method == 'mean':
                z_mean = iti_data.mean() if len(iti_data) > 0 else 0
                z_sd   = iti_data.std()  if len(iti_data) > 0 else 1
            else:  # median
                z_mean = np.median(iti_data) if len(iti_data) > 0 else 0
                z_sd   = np.median(np.abs(iti_data - z_mean)) if len(iti_data) > 0 else 1
                if scale_median and z_sd != 0:
                    z_sd *= 1.4826  # Scale MAD to match std dev for normal distribution

        elif trial_normalize == 'prior':
            if normalize_side in left_selection_list:
                baseline_mask = ((trial_time >= abet_time_list.loc[index, 'Start_Time']) &
                                 (trial_time <= (abet_time_list.loc[index, 'Start_Time'] + extra_prior)))
            else:
                baseline_mask = ((trial_time >= (abet_time_list.loc[index, 'End_Time'] - extra_follow)) &
                                 (trial_time <= abet_time_list.loc[index, 'End_Time']))
            baseline_data = trial_deltaf[baseline_mask]
            if center_method == 'mean':
                z_mean = baseline_data.mean() if len(baseline_data) > 0 else 0
                z_sd   = baseline_data.std()  if len(baseline_data) > 0 else 1
            else:
                z_mean = np.median(baseline_data) if len(baseline_data) > 0 else 0
                z_sd   = np.median(np.abs(baseline_data - z_mean)) if len(baseline_data) > 0 else 1
                if scale_median and z_sd != 0:
                    z_sd *= 1.4826  # Scale MAD to match std dev for normal distribution

        else:  # whole
            if center_method == 'mean':
                z_mean = trial_deltaf.mean() if len(trial_deltaf) > 0 else 0
                z_sd   = trial_deltaf.std()  if len(trial_deltaf) > 0 else 1
            else:
                z_mean = np.median(trial_deltaf) if len(trial_deltaf) > 0 else 0
                z_sd   = np.median(np.abs(trial_deltaf - z_mean)) if len(trial_deltaf) > 0 else 1
                if scale_median and z_sd != 0:
                    z_sd *= 1.4826  # Scale MAD to match std dev for normal distribution

        # Handle safely cases where SD is equal to zero
        if z_sd == 0:
            zscore =  np.full_like(trial_deltaf, np.nan)
        else:
            zscore = (trial_deltaf - z_mean) / z_sd

        time_trials.append(trial_time)
        zscore_trials.append(zscore)
        deltaf_trials.append(trial_deltaf)

    # --- Build DataFrames ---
    if not time_trials:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    max_len = max(len(a) for a in time_trials)

    def _pad(arr):
        return np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)

    time_df    = pd.DataFrame({f'Time Trial {i+1}':    _pad(a) for i, a in enumerate(time_trials)})
    zscore_df  = pd.DataFrame({f'Z-Score Trial {i+1}': _pad(a) for i, a in enumerate(zscore_trials)})
    deltaf_df  = pd.DataFrame({f'Delta-F Trial {i+1}': _pad(a) for i, a in enumerate(deltaf_trials)})

    partial_dataframe = zscore_df
    partial_deltaf    = deltaf_df
    final_dataframe   = pd.concat([time_df, zscore_df], axis=1)
    final_deltaf      = pd.concat([time_df, deltaf_df], axis=1)

    return partial_dataframe, final_dataframe, partial_deltaf, final_deltaf


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def calculate_max_peak(partial_dataframe):
    """Return the mean of the per-trial maximum z-scores."""
    if partial_dataframe is not None and not partial_dataframe.empty:
        return partial_dataframe.max().mean()
    return 0

# Add Check for Numpy trapezoid function
try:
    from numpy import trapezoid as trapz
except ImportError:
    from numpy import trapz


def calculate_auc(partial_dataframe, event_start, event_end):
    """Return the mean area-under-the-curve across all trials."""
    if partial_dataframe is not None and not partial_dataframe.empty:
        auc_values = []
        for col in partial_dataframe.columns:
            trial_data = partial_dataframe[col].dropna().values
            if len(trial_data) > 1:
                time_series = np.linspace(event_start, event_end, len(trial_data))
                # Check for NA values in trial_data and remove them from time_series if present
                if np.isnan(trial_data).any():
                    valid_indices = ~np.isnan(trial_data)
                    trial_data = trial_data[valid_indices]
                    time_series = time_series[valid_indices]
                auc = trapz(trial_data, x=time_series)
                auc_values.append(auc)
        return np.mean(auc_values)
    return 0
