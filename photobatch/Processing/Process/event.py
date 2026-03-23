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
                    sample_frequency, extra_prior=0, extra_follow=0,
                    trial_normalize='whole', normalize_side='Left',
                    trial_iti_pad=0, center_method='mean'):
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
        'Left' / 'Before' for left baseline; 'Right' / 'After' for right.
    trial_iti_pad : float  additional ITI padding (seconds).
    center_method : str  'mean' or 'median'.

    Returns
    -------
    partial_dataframe : pd.DataFrame   z-score trial columns only.
    final_dataframe   : pd.DataFrame   time + z-score trial columns.
    partial_deltaf    : pd.DataFrame   delta-F trial columns only.
    final_deltaf      : pd.DataFrame   time + delta-F trial columns.
    """
    left_selection_list  = ['Left', 'Before', 'L', 'l', 'left', 'before', 1]

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
            if normalize_side in left_selection_list:
                trial_start_diff = trial_definition_times.loc[:, 'Start_Time'].sub(
                    abet_time_list.loc[index, 'Start_Time'] + extra_prior)
                trial_start_diff[trial_start_diff > 0] = np.nan
                trial_start_index = trial_start_diff.abs().idxmin(skipna=True)
                trial_start_window = trial_definition_times.iloc[trial_start_index, 0]
                trial_iti_window   = trial_start_window - float(trial_iti_pad)
                iti_mask = ((doric_time_array >= trial_iti_window) &
                            (doric_time_array <= trial_start_window))
                iti_data = doric_deltaf_array[iti_mask]
            else:
                trial_end_index  = trial_definition_times.loc[:, 'End_Time'].sub(
                    abet_time_list.loc[index, 'End_Time']).abs().idxmin()
                trial_end_window = trial_definition_times.iloc[trial_end_index, 0]
                trial_iti_window = trial_end_window + trial_iti_pad
                iti_mask = ((doric_time_array >= trial_end_window) &
                            (doric_time_array <= trial_iti_window))
                iti_data = doric_deltaf_array[iti_mask]

            if center_method == 'mean':
                z_mean = iti_data.mean() if len(iti_data) > 0 else 0
                z_sd   = iti_data.std()  if len(iti_data) > 0 else 1
            else:  # median
                z_mean = np.median(iti_data) if len(iti_data) > 0 else 0
                z_sd   = np.median(np.abs(iti_data - z_mean)) if len(iti_data) > 0 else 1

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

        else:  # whole
            if center_method == 'mean':
                z_mean = trial_deltaf.mean() if len(trial_deltaf) > 0 else 0
                z_sd   = trial_deltaf.std()  if len(trial_deltaf) > 0 else 1
            else:
                z_mean = np.median(trial_deltaf) if len(trial_deltaf) > 0 else 0
                z_sd   = np.median(np.abs(trial_deltaf - z_mean)) if len(trial_deltaf) > 0 else 1

        zscore = (trial_deltaf - z_mean) / z_sd if z_sd != 0 else (trial_deltaf - z_mean)

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


def calculate_auc(partial_dataframe, event_start, event_end):
    """Return the mean area-under-the-curve across all trials."""
    if partial_dataframe is not None and not partial_dataframe.empty:
        time_series = np.linspace(event_start, event_end,
                                  num=len(partial_dataframe))
        auc_values = [
            np.trapezoid(y=partial_dataframe[col], x=time_series)
            for col in partial_dataframe.columns
        ]
        return np.mean(auc_values)
    return 0
