# Imports
from numbers import Integral, Real
from typing import Any, cast
from dtw import dtw
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd



# Functions - Extract whole trial data
def extract_whole_trial_data(signal_df,trial_window_df):
    """
    Separates out individual behavioural trials from the continuous signal data. Trials are calculated from the trial window,
    which contains a column for the start time of the trial, and the end time of the trial. The function iterates across the rows of
    the trial_window_df and creates a separate DataFrame for each trial.
    Parameters:
    signal_df (pd.DataFrame): DataFrame containing the continuous signal data. Must contain a 'time' column.
    trial_window_df (pd.DataFrame): DataFrame containing the trial window information. Must contain 'start_time' and 'end_time' columns.
    Return:
    trial_signal_list (list): List of DataFrames, where each DataFrame contains the signal data for a single trial.

    """
    # Check consistency of input DataFrames
    if 'time' not in signal_df.columns:
        raise ValueError("signal_df must contain a 'time' column.")
    if 'start_time' not in trial_window_df.columns or 'end_time' not in trial_window_df.columns:
        raise ValueError("trial_window_df must contain 'start_time' and 'end_time' columns.")
    
    # Start efficient loop of trial_window_df using vectorized operations
    trial_signal_list = []
    for _, row in trial_window_df.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        
        # Use boolean indexing to extract the relevant signal data for the current trial
        trial_signal = signal_df[(signal_df['time'] >= start_time) & (signal_df['time'] <= end_time)].copy()
        
        # Append the trial signal DataFrame to the list
        trial_signal_list.append(trial_signal)
    
    return trial_signal_list

def _normalize_baseline_method(baseline_method):
    """Return a normalized baseline method key."""
    method = str(baseline_method).strip().lower()
    if method not in ('whole', 'prior', 'iti'):
        raise ValueError("baseline_method must be 'whole', 'prior', or 'iti'.")
    return method


def _normalize_iti_method(iti_method):
    """Return a normalized ITI selection key."""
    iti_key = str(iti_method).strip().lower()
    if iti_key not in ('start', 'end', 'center'):
        raise ValueError("iti_method must be 'start', 'end', or 'center'.")
    return iti_key


def _get_numeric_signal_columns(signal_df):
    """Return numeric signal columns excluding the time axis."""
    return [
        column for column in signal_df.columns
        if column != 'time' and pd.api.types.is_numeric_dtype(signal_df[column])
    ]


def _validate_trial_signal_list(trial_signal_list):
    """Validate a list of trial DataFrames and return it unchanged."""
    if not isinstance(trial_signal_list, list) or not trial_signal_list:
        raise ValueError('trial_signal_list must be a non-empty list of pandas DataFrames.')

    for index, trial_df in enumerate(trial_signal_list):
        if not isinstance(trial_df, pd.DataFrame):
            raise ValueError(
                f'trial_signal_list[{index}] must be a pandas DataFrame.'
            )
        if 'time' not in trial_df.columns:
            raise ValueError(
                f"trial_signal_list[{index}] must contain a 'time' column."
            )
        if trial_df.empty:
            raise ValueError(
                f'trial_signal_list[{index}] must not be empty.'
            )

    return trial_signal_list


def _normalize_column_names(columns, parameter_name):
    """Normalize a column name parameter to a list of column names."""
    if columns is None:
        return []

    if isinstance(columns, str):
        column_list = [columns]
    else:
        try:
            column_list = list(columns)
        except TypeError as exc:
            raise ValueError(f'{parameter_name} must be a string or an iterable of strings.') from exc

    normalized_columns = []
    for column in column_list:
        column_name = str(column).strip()
        if not column_name:
            raise ValueError(f'{parameter_name} must not contain empty column names.')
        normalized_columns.append(column_name)

    return normalized_columns


def _resolve_signal_column(trial_signal_list, signal_column=None, excluded_columns=None):
    """Resolve the signal column used for one-dimensional trial models."""
    excluded = {'time'}
    if excluded_columns:
        excluded.update(excluded_columns)

    candidate_columns = []
    first_trial = trial_signal_list[0]
    for column in first_trial.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(first_trial[column]):
            candidate_columns.append(column)

    if signal_column is not None:
        signal_key = str(signal_column).strip()
        if not signal_key:
            raise ValueError('signal_column must be a non-empty string when provided.')
        if signal_key not in candidate_columns:
            raise ValueError(
                f"signal_column '{signal_key}' must identify a numeric column in the trial DataFrames."
            )
        return signal_key

    if not candidate_columns:
        raise ValueError('No numeric signal columns were found for GLM-HMM fitting.')

    if len(candidate_columns) > 1:
        raise ValueError(
            'Multiple numeric signal columns are available. Specify signal_column explicitly.'
        )

    return candidate_columns[0]


def _extract_alignment_indices(alignment):
    """Return DTW warping path indices as NumPy arrays."""
    try:
        index1 = np.asarray(getattr(alignment, 'index1'), dtype=int)
        index2 = np.asarray(getattr(alignment, 'index2'), dtype=int)
    except AttributeError as exc:
        raise ValueError('DTW alignment object did not expose warping path indices.') from exc

    if index1.size == 0 or index2.size == 0 or index1.size != index2.size:
        raise ValueError('DTW alignment produced an invalid warping path.')

    return index1, index2


def _align_trial_to_reference(reference_time, trial_df, alignment):
    """Project one trial onto the reference time axis using the DTW path."""
    target_indices, reference_indices = _extract_alignment_indices(alignment)
    aligned_columns = {'time': np.asarray(reference_time, dtype=float)}

    for column in trial_df.columns:
        if column == 'time':
            continue

        source_values = trial_df[column].to_numpy()
        aligned_values = []

        for reference_index in range(len(reference_time)):
            matched_indices = target_indices[reference_indices == reference_index]
            if matched_indices.size == 0:
                raise ValueError(
                    'DTW alignment did not cover the full reference time axis.'
                )

            if pd.api.types.is_numeric_dtype(trial_df[column]):
                aligned_values.append(source_values[matched_indices].astype(float).mean())
            else:
                aligned_values.append(source_values[matched_indices[0]])

        if pd.api.types.is_numeric_dtype(trial_df[column]):
            aligned_columns[column] = np.asarray(aligned_values, dtype=float)
        else:
            aligned_columns[column] = np.asarray(aligned_values, dtype=object)

    return pd.DataFrame(aligned_columns, columns=trial_df.columns)


def _extract_signal_window(signal_df, signal_column, start_time, end_time):
    """Return values from one signal column within a time window."""
    window_start, window_end = sorted((float(start_time), float(end_time)))
    signal_mask = (
        (signal_df['time'] >= window_start) &
        (signal_df['time'] <= window_end)
    )
    return signal_df.loc[signal_mask, signal_column].to_numpy(dtype=float)


def _resolve_baseline_values(
    signal_df,
    trial_window_df,
    trial_signal,
    trial_index,
    signal_column,
    baseline_method,
    baseline_window,
    iti_method,
):
    """Return the baseline values used to normalise one trial signal column."""
    method = _normalize_baseline_method(baseline_method)

    if method == 'whole':
        return trial_signal[signal_column].to_numpy(dtype=float)

    if baseline_window is None:
        raise ValueError(
            "baseline_window is required when baseline_method is not 'whole'."
        )

    trial_row = trial_window_df.iloc[trial_index]
    trial_start = float(trial_row['start_time'])
    trial_end = float(trial_row['end_time'])

    if isinstance(baseline_window, (list, tuple, np.ndarray)) and len(baseline_window) == 2:
        window_start, window_end = sorted(float(value) for value in baseline_window)
        return _extract_signal_window(signal_df, signal_column, window_start, window_end)

    if not isinstance(baseline_window, Real) or isinstance(baseline_window, bool):
        raise ValueError(
            'baseline_window must be a positive number or a two-value time window.'
        )

    window_size = float(baseline_window)
    if window_size <= 0:
        raise ValueError('baseline_window must be greater than 0.')

    if method == 'prior':
        return _extract_signal_window(
            signal_df,
            signal_column,
            trial_start - window_size,
            trial_start,
        )

    iti_selection = _normalize_iti_method(iti_method)
    signal_start = float(signal_df['time'].min())
    iti_start = signal_start
    if trial_index > 0:
        iti_start = float(trial_window_df.iloc[trial_index - 1]['end_time'])
    iti_end = trial_start

    if iti_end <= iti_start:
        return np.array([], dtype=float)

    if iti_selection == 'end':
        baseline_start = max(iti_start, iti_end - window_size)
        baseline_end = iti_end
    elif iti_selection == 'start':
        baseline_start = iti_start
        baseline_end = min(iti_end, iti_start + window_size)
    else:
        iti_center = (iti_start + iti_end) / 2.0
        half_window = window_size / 2.0
        baseline_start = max(iti_start, iti_center - half_window)
        baseline_end = min(iti_end, iti_center + half_window)

    return _extract_signal_window(signal_df, signal_column, baseline_start, baseline_end)


def _calculate_zscore_stats(signal_values, center_method):
    """Return the baseline center and spread using mean/std or median/MAD."""
    center_key = str(center_method).strip().lower()
    values = np.asarray(signal_values, dtype=float)
    values = values[~np.isnan(values)]

    if values.size == 0:
        return 0.0, 1.0

    if center_key == 'mean':
        z_mean = values.mean()
        z_sd = values.std()
    elif center_key == 'median':
        z_mean = np.median(values)
        z_sd = np.median(np.abs(values - z_mean))
    else:
        raise ValueError("center_method must be either 'mean' or 'median'.")

    return z_mean, z_sd if z_sd != 0 else 0.0


def normalize_trial_data(signal_df, trial_window_df, trial_signal_list,
                         baseline_method='whole', baseline_window=None,
                         iti_method='end', center_method='mean'):
    """
    Z-score each trial DataFrame using full-signal baseline statistics.

    Parameters:
    signal_df (pd.DataFrame): Continuous signal DataFrame. Must contain a 'time' column and
        the numeric signal columns present in each trial DataFrame.
    trial_window_df (pd.DataFrame): DataFrame containing per-trial 'start_time' and 'end_time'
        rows aligned with trial_signal_list.
    trial_signal_list (list): List of trial DataFrames. Each DataFrame must contain a 'time' column
        and at least one additional numeric signal column.
    baseline_method (str): Baseline selection method. Supported values are 'whole', 'prior', and 'iti'.
        'whole' uses the full trial. 'prior' uses signal data immediately before the trial start.
        'iti' uses inter-trial signal data before or after the trial depending on iti_method.
    baseline_window (float | tuple | list | np.ndarray | None): Baseline window definition when
        baseline_method is not 'whole'. A scalar value selects a duration in seconds. A two-value
        sequence selects an explicit absolute time range from signal_df.
    iti_method (str): For 'iti' normalization, select whether the baseline window comes from the
        start, end, or center of the pre-trial inter-trial interval.
    center_method (str): Z-score center and spread calculation. Supported values are 'mean' and 'median'.

    Return:
    normalized_trial_list (list): List of DataFrames containing the original 'time' column and
        z-scored signal columns.
    """
    if not isinstance(signal_df, pd.DataFrame):
        raise ValueError('signal_df must be a pandas DataFrame.')
    if 'time' not in signal_df.columns:
        raise ValueError("signal_df must contain a 'time' column.")
    if not isinstance(trial_window_df, pd.DataFrame):
        raise ValueError('trial_window_df must be a pandas DataFrame.')
    if 'start_time' not in trial_window_df.columns or 'end_time' not in trial_window_df.columns:
        raise ValueError("trial_window_df must contain 'start_time' and 'end_time' columns.")
    if not isinstance(trial_signal_list, list):
        raise ValueError('trial_signal_list must be a list of pandas DataFrames.')
    if len(trial_window_df) != len(trial_signal_list):
        raise ValueError('trial_window_df and trial_signal_list must have the same length.')

    signal_columns = _get_numeric_signal_columns(signal_df)
    if not signal_columns:
        raise ValueError(
            "signal_df must contain at least one numeric signal column besides 'time'."
        )

    normalized_trial_list = []
    for trial_index, trial_signal in enumerate(trial_signal_list):
        if not isinstance(trial_signal, pd.DataFrame):
            raise ValueError('Each trial in trial_signal_list must be a pandas DataFrame.')
        if 'time' not in trial_signal.columns:
            raise ValueError("Each trial DataFrame must contain a 'time' column.")

        trial_signal_columns = [
            column for column in signal_columns
            if column in trial_signal.columns and pd.api.types.is_numeric_dtype(trial_signal[column])
        ]
        if not trial_signal_columns:
            raise ValueError(
                "Each trial DataFrame must contain at least one numeric signal column besides 'time'."
            )

        normalized_trial = trial_signal.copy()

        for signal_column in trial_signal_columns:
            trial_values = normalized_trial[signal_column].to_numpy(dtype=float)
            baseline_values = _resolve_baseline_values(
                signal_df,
                trial_window_df,
                normalized_trial,
                trial_index,
                signal_column,
                baseline_method,
                baseline_window,
                iti_method,
            )
            z_mean, z_sd = _calculate_zscore_stats(baseline_values, center_method)
            if z_sd == 0:
                normalized_trial[signal_column] = trial_values - z_mean
            else:
                normalized_trial[signal_column] = (trial_values - z_mean) / z_sd

        normalized_trial_list.append(normalized_trial)

    return normalized_trial_list

# Time Warping Functions

def _align_data_dtw(trial_signal_list, reference_trial_index=0):
    """
    Standard implementation of the dynamic time warping algorithm. Uses the dtw-python package for implementation.
    Uses an initial reference trial (user defined as index within trial_signal_list) to align all other trials to.

    Parameters:
    trial_signal_list (list): List of trial Pandas DataFrames. Each DataFrame must contain a 'time' column and
        at least one additional numeric signal column.
    reference_trial_index (int): Index of the trial in trial_signal_list to use as the reference for alignment.

    Returns:
    aligned_trial_list (list): List of DataFrames containing the warped 'time' column and signal columns.
    """
    trial_signal_list = _validate_trial_signal_list(trial_signal_list)
    ref_df = trial_signal_list[reference_trial_index]
    ref_time = ref_df['time'].to_numpy()
    signal_columns = _get_numeric_signal_columns(ref_df)

    if not signal_columns:
        raise ValueError("Reference trial must contain at least one numeric signal column besides 'time'.")

    # Use first signal column for alignment path
    ref_signal = ref_df[signal_columns[0]].to_numpy()
    aligned_df_list = []

    for i, df in enumerate(trial_signal_list):
        if i == reference_trial_index:
            aligned_df_list.append(ref_df.copy())
            continue

        target_signal = df[signal_columns[0]].to_numpy()

        # Calculate DTW path using the dtw package
        # Align target_signal to ref_signal
        alignment = dtw(target_signal, ref_signal, keep_internals=True)

        aligned_df_list.append(_align_trial_to_reference(ref_time, df, alignment))

    return aligned_df_list


def _align_data_rtw(trial_signal_list, reference_trial_index=0, window_radius=5):
    """
    Standard implementation of the real-time warping algorithm. Uses a Sakoe-Chiba windowed DTW approach 
    to align all trials to a reference trial.

    Parameters:
    trial_signal_list (list): List of trial Pandas DataFrames. Each DataFrame must contain a 'time' column and
        at least one additional numeric signal column.
    reference_trial_index (int): Index of the trial in trial_signal_list to use as the reference for alignment.
    window_radius (int): Window radius for Sakoe-Chiba constraint.

    Returns:
    aligned_trial_list (list): List of DataFrames containing the warped 'time' column and signal columns.
    """
    trial_signal_list = _validate_trial_signal_list(trial_signal_list)
    ref_df = trial_signal_list[reference_trial_index]
    ref_time = ref_df['time'].to_numpy()
    signal_columns = _get_numeric_signal_columns(ref_df)

    if not signal_columns:
        raise ValueError("Reference trial must contain at least one numeric signal column besides 'time'.")

    ref_signal = ref_df[signal_columns[0]].to_numpy()
    aligned_df_list = []

    for i, df in enumerate(trial_signal_list):
        if i == reference_trial_index:
            aligned_df_list.append(ref_df.copy())
            continue

        target_signal = df[signal_columns[0]].to_numpy()

        # Use dtw-python to conduct windowed DTW (RTW)
        alignment = dtw(target_signal, ref_signal, keep_internals=True,
                        window_type='sakoechiba', window_args={'window_radius': window_radius})
        
        aligned_df_list.append(_align_trial_to_reference(ref_time, df, alignment))

    return aligned_df_list

def time_warp_trial_data(trial_signal_list, time_warp_method='dtw'):
    """
    Time warp each trial DataFrame to a common time axis.

    Parameters:
    trial_signal_list (list): List of trial DataFrames. Each DataFrame must contain a 'time' column and
        at least one additional numeric signal column.
    time_warp_method (str): Time warping method. Supported values are 'dtw' and 'rtw'.
        'dtw' uses dynamic time warping to align trials.
        'rtw' uses real-time warping with a Sakoe-Chiba windowed DTW approach.
    Return:
    warped_trial_list (list): List of DataFrames containing the warped 'time' column and
        signal columns.
    """
    trial_signal_list = _validate_trial_signal_list(trial_signal_list)

    # Check Time Warp Method and call the appropriate alignment function
    method_key = str(time_warp_method).strip().lower()
    if method_key == 'dtw':
        return _align_data_dtw(trial_signal_list)
    elif method_key == 'rtw':
        return _align_data_rtw(trial_signal_list)
    else:
        raise ValueError("time_warp_method must be either 'dtw' or 'rtw'.")
    
# Analysis Functions

# Functional Mixed Linear Model

def calculate_fmlm(trial_signal_list, formula, group_column):
    """
    Calculate a functional mixed linear model (FMLM) on the trial data. The trial_signal_list contains a series of trial DataFrames.
    The dataframes will already be normalised and time warped, so the FMLM will be calculated across the aligned time axis. 
    The formula parameter specifies the fixed and random effects for the FMLM using a Patsy formula string. 
    The group_column parameter specifies the column in the trial DataFrames that identifies the grouping variable for 
    random effects (e.g., subject ID).

    Parameters:
    trial_signal_list (list): List of trial DataFrames. Each DataFrame must contain a 'time' column and
        at least one additional numeric signal column.
    formula (str): A Patsy formula string specifying the fixed and random effects for the FMLM.
    group_column (str): The name of the column in the trial DataFrames that identifies the grouping variable for random effects.

    Returns:
    fmlm_results: The fitted FMLM results object containing parameter estimates and statistics.
    """
    
    trial_signal_list = _validate_trial_signal_list(trial_signal_list)

    formula_key = str(formula).strip()
    if not formula_key:
        raise ValueError('formula must be a non-empty string.')
    
    group_key = str(group_column).strip()
    if not group_key:
        raise ValueError('group_column must be a non-empty string.')

    model_data = pd.concat(trial_signal_list, ignore_index=True)
    if group_key not in model_data.columns:
        raise ValueError(
            f"group_column '{group_key}' was not found in the trial DataFrames."
        )
    if model_data[group_key].isna().any():
        raise ValueError(f"group_column '{group_key}' contains missing values.")
    if model_data[group_key].nunique(dropna=False) < 2:
        raise ValueError('calculate_fmlm requires at least two groups.')
    
    model = smf.mixedlm(
        formula=formula_key,
        data=model_data,
        groups=model_data[group_key],
        re_formula='~time',
        missing='drop',
    )
    fmlm_results = model.fit()
    return fmlm_results

def calculate_glm_hmm(
    trial_signal_list,
    n_states,
    covariate_cols=None,
    signal_column=None,
    seed=42,
    num_iters=50,
    verbose=False,
):
    """
    Calculate a generalized linear model hidden Markov model (GLM-HMM) on the trial data. The trial_signal_list contains a series of trial DataFrames.
    The dataframes will already be normalised and time warped, so the GLM-HMM will be calculated across the aligned time axis.
    This implementation fits a one-dimensional LinearRegressionHMM from Dynamax to one selected signal column,
    optionally using trial covariates as regression inputs.

    Parameters:
    trial_signal_list (list): List of trial DataFrames. Each DataFrame must contain a 'time' column and
        at least one additional numeric signal column.
    n_states (int): The number of hidden states to fit in the GLM-HMM.
    covariate_cols (str | list | tuple | None): Optional covariate column name(s) used as model inputs.
    signal_column (str | None): Numeric signal column to model. Required when multiple candidate
        signal columns are present.
    seed (int): Random seed used to initialize model parameters.
    num_iters (int): Number of EM iterations.
    verbose (bool): Whether Dynamax should display the EM progress bar.

    Returns:
    tuple: The fitted model, fitted parameters, EM log likelihoods, and one posterior per trial.
    """
    trial_signal_list = _validate_trial_signal_list(trial_signal_list)

    if not isinstance(n_states, Integral) or isinstance(n_states, bool) or n_states <= 0:
        raise ValueError('n_states must be a positive integer.')
    if not isinstance(num_iters, Integral) or isinstance(num_iters, bool) or num_iters <= 0:
        raise ValueError('num_iters must be a positive integer.')

    try:
        import jax.numpy as jnp
        import jax.random as jr
        from dynamax.hidden_markov_model import LinearRegressionHMM
    except ModuleNotFoundError as exc:
        raise ImportError(
            'calculate_glm_hmm requires jax, jaxlib, dynamax, and IPython to be installed.'
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            'calculate_glm_hmm could not import the Dynamax GLM-HMM stack. '
            'This usually means the installed jax, jaxlib, optax, and '
            'tensorflow-probability versions are incompatible.'
        ) from exc

    n_states = int(n_states)
    num_iters = int(num_iters)

    covariate_names = _normalize_column_names(covariate_cols, 'covariate_cols')
    signal_key = _resolve_signal_column(
        trial_signal_list,
        signal_column=signal_column,
        excluded_columns=covariate_names,
    )
    key = jr.PRNGKey(int(seed))

    obs_list = []
    inputs_list = []
    trial_lengths = set()

    for index, trial_df in enumerate(trial_signal_list):
        missing_columns = [
            column for column in [signal_key, *covariate_names]
            if column not in trial_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f'trial_signal_list[{index}] is missing required columns: {missing_columns}.'
            )

        trial_lengths.add(len(trial_df))

        observation_values = trial_df[signal_key].to_numpy(dtype=float)
        if np.isnan(observation_values).any():
            raise ValueError(
                f"signal_column '{signal_key}' contains missing values in trial_signal_list[{index}]."
            )
        obs_list.append(observation_values.reshape(-1, 1))

        if covariate_names:
            covariate_values = trial_df[covariate_names].to_numpy(dtype=float)
            if np.isnan(covariate_values).any():
                raise ValueError(
                    f'covariate_cols contain missing values in trial_signal_list[{index}].'
                )
            x = np.column_stack([np.ones(len(trial_df), dtype=float), covariate_values])
        else:
            x = np.ones((len(trial_df), 1), dtype=float)
        inputs_list.append(x)

    if len(trial_lengths) != 1:
        raise ValueError(
            'All trial DataFrames must have the same number of rows for GLM-HMM fitting. '
            'Use time_warp_trial_data to align trials to a common time axis first.'
        )

    observations = jnp.asarray(np.stack(obs_list, axis=0))
    inputs_all = jnp.asarray(np.stack(inputs_list, axis=0))

    input_dim = inputs_all.shape[-1]
    output_dim = 1
    try:
        hmm = LinearRegressionHMM(
            num_states=n_states,
            input_dim=input_dim,
            emission_dim=output_dim,
        )
        params, props = hmm.initialize(key)
        fitted_params, lls = hmm.fit_em(
            params,
            props,
            observations,
            inputs_all,
            num_iters=num_iters,
            verbose=bool(verbose),
        )
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            'calculate_glm_hmm failed while initializing or fitting the Dynamax model. '
            'This usually indicates an incompatible jax, jaxlib, optax, or '
            'tensorflow-probability installation.'
        ) from exc
    posterior_params = cast(Any, fitted_params)

    posteriors = [
        hmm.smoother(posterior_params, observation, model_input)
        for observation, model_input in zip(observations, inputs_all)
    ]

    return hmm, fitted_params, lls, posteriors