# Imports
import os
import logging
import dateutil.parser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

# Sub-package imports
from photobatch.exceptions import (
    PhotobatchError,
    SynchronizationError,
    UnsupportedFileFormatError,
    MissingColumnError,
)
from photobatch.Processing.IO import BEHAVIOUR_REGISTRY, SIGNAL_REGISTRY, SYNC_REGISTRY
from photobatch.Processing.IO.Behaviour.abet import (
    abet_trial_definition as _abet_trial_definition,
    abet_search_event as _abet_search_event,
)
from photobatch.Processing.IO.output import write_data as _write_data, write_summary as _write_summary
from photobatch.Processing.Signal.utilities import crop_signal
from photobatch.Processing.Signal.filter import signal_filter as _signal_filter
from photobatch.Processing.Signal.fitting import signal_fit as _signal_fit
from photobatch.Processing.Process.event import (
    trial_separator as _trial_separator,
    calculate_max_peak as _calculate_max_peak,
    calculate_auc as _calculate_auc,
    _generate_event_alias,
    _create_filter_list,
)
from photobatch.Processing.Process.advanced_analysis import (
    run_flmm_analysis,
    run_glm_hmm_analysis,
    run_moa_hmm_analysis,
)
from photobatch.Processing import hdf_store


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _as_bool(value) -> bool:
    """Return a stable boolean for JSON/config values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _build_session_id(animal_id, date, fallback_path: str) -> str:
    """Return a stable HDF session key for one recording."""
    cleaned_parts = [str(part).strip().replace('/', '-').replace('\\', '-') for part in (animal_id, date) if part]
    if cleaned_parts:
        return '_'.join(cleaned_parts)
    return Path(fallback_path).stem


# ---------------------------------------------------------------------------
# Main orchestration class
# ---------------------------------------------------------------------------

class SignalEventData:
    """Top-level orchestration object for a single photometry analysis session.

    Acts as a thin wrapper that:
    * holds an in-memory copy of loaded/processed data frames.
    * delegates all real work to the dedicated sub-package functions.
    * exposes a stable method API consumed by the GUI and _process_single_file.

    Long-term design note
    ---------------------
    The class is intentionally vendor-agnostic.  Future releases will allow
    the caller to select a behaviour-data vendor (ABET, AnyMaze, …) and a
    signal-data vendor (Doric, Neurophotometrics, …) and this class will route
    to the appropriate IO sub-module dynamically.
    """

    def __init__(self):
        # Folder paths
        self.curr_dir        = Path.cwd()
        self.main_folder_path = Path.cwd()
        self.folder_symbol   = os.sep
        self.data_folder_path = self.main_folder_path / 'Data'

        self.abet_file_path  = ''
        self.abet_file       = ''
        self.doric_file_path = ''
        self.doric_file      = ''
        self.anymaze_file_path = ''
        self.anymaze_file    = ''

        # State flags
        self.behaviour_loaded = False
        self.abet_searched   = False
        self.anymaze_loaded  = False
        self.signal_loaded    = False

        # Numeric
        self.abet_doric_sync_value   = 0
        self.anymaze_doric_sync_value = 0
        self.extra_prior             = 0
        self.extra_follow            = 0
        self.sample_frequency        = 0
        self.doric_time              = 0

        # Descriptors
        self.date      = None
        self.animal_id = None

        # String
        self.event_name_col = ''
        self.time_var_name  = ''
        self.event_name     = ''
        self.event_alias    = ''

        # Lists
        self.abet_time_list = []

        # DataFrames
        self.partial_dataframe  = pd.DataFrame()
        self.final_dataframe    = pd.DataFrame()
        self.partial_deltaf     = pd.DataFrame()
        self.final_deltaf       = pd.DataFrame()
        self.partial_percent    = pd.DataFrame()
        self.final_percent      = pd.DataFrame()
        self.abet_pd            = pd.DataFrame()
        self.doric_pd           = pd.DataFrame()
        self.signal_df          = pd.DataFrame()
        self.ttl_pandas         = pd.DataFrame()
        self.abet_raw_data      = pd.DataFrame()
        self.anymaze_pandas     = pd.DataFrame()
        self.behaviour_df       = pd.DataFrame()
        self.abet_event_times   = pd.DataFrame()
        self.trial_definition_times = pd.DataFrame()

    # ------------------------------------------------------------------
    # Backward-compatibility attribute aliases (deprecated names)
    # ------------------------------------------------------------------
    @property
    def abet_loaded(self):
        return self.behaviour_loaded

    @abet_loaded.setter
    def abet_loaded(self, value):
        self.behaviour_loaded = value

    @property
    def doric_loaded(self):
        return self.signal_loaded

    @doric_loaded.setter
    def doric_loaded(self, value):
        self.signal_loaded = value

    @property
    def abet_pandas(self):
        return self.behaviour_df

    @abet_pandas.setter
    def abet_pandas(self, value):
        self.behaviour_df = value

    @property
    def doric_pandas(self):
        return self.signal_df

    @doric_pandas.setter
    def doric_pandas(self, value):
        self.signal_df = value

    # -----------------------------------------------------------------------
    # IO – generic vendor-dispatched loaders
    # -----------------------------------------------------------------------

    def load_behaviour_data(self, filepath, vendor='abet'):
        """Load behaviour data using the specified vendor plugin."""
        loader = BEHAVIOUR_REGISTRY[vendor]['load']
        self.abet_file_path = filepath
        self.behaviour_loaded = True
        (self.behaviour_df,
         self.animal_id,
         self.date,
         self.time_var_name,
         self.event_name_col) = loader(filepath)
        # Backward compat: set old names
        self.abet_loaded = True
        self.abet_pandas = self.behaviour_df

    def load_signal_data(self, filepath, ch1_col, ch2_col, ttl_col,
                         mode='', vendor='doric'):
        """Load signal data using the specified vendor plugin."""
        loader = SIGNAL_REGISTRY[vendor]['load']
        self.signal_loaded = True
        self.doric_file_path = filepath
        result = loader(filepath, ch1_col, ch2_col, ttl_col, mode)
        if result is None or result[0] is None:
            self.signal_loaded = False
            return
        self.signal_df, self.ttl_pandas = result
        # Backward compat: set old names
        self.doric_loaded = True
        self.doric_pandas = self.signal_df

    def synchronize_time(self, behaviour_vendor='abet', signal_vendor='doric'):
        """Align signal time axis to behaviour time via TTL cross-correlation."""
        if not self.behaviour_loaded or not self.signal_loaded:
            return None
        sync_fn = SYNC_REGISTRY[(behaviour_vendor, signal_vendor)]
        self.signal_df = sync_fn(
            self.signal_df, self.ttl_pandas, self.behaviour_df)
        # Backward compat: update old name
        self.doric_pandas = self.signal_df

    # -----------------------------------------------------------------------
    # IO – Behaviour (ABET)
    # -----------------------------------------------------------------------

    def load_abet_data(self, filepath):
        """Load ABET II / ABET Cognition data from *filepath*. Delegates to load_behaviour_data."""
        self.load_behaviour_data(filepath, vendor='abet')

    def abet_trial_definition(self, start_event_group, end_event_group):
        """Define trial start/end windows from ABET Condition Events."""
        if not self.behaviour_loaded:
            return None
        self.trial_definition_times = _abet_trial_definition(
            self.behaviour_df, self.time_var_name,
            start_event_group, end_event_group)

    def abet_search_event(self, start_event_id='1', start_event_group='',
                          start_event_item_name='', start_event_position=None,
                          filter_event=False, filter_list=None,
                          extra_prior_time=0, extra_follow_time=0,
                          exclusion_list=None, event_alias=''):
        """Search ABET data for a specific event with optional filters."""
        if not self.behaviour_loaded:
            return
        result = cast(
            tuple[pd.DataFrame, str, str, float, float],
            _abet_search_event(
            self.behaviour_df, self.time_var_name, self.event_name_col,
            start_event_id=start_event_id,
            start_event_group=start_event_group,
            start_event_item_name=start_event_item_name,
            start_event_position=start_event_position,
            filter_event=filter_event,
            filter_list=filter_list,
            extra_prior_time=extra_prior_time,
            extra_follow_time=extra_follow_time,
            exclusion_list=exclusion_list,
            event_alias=event_alias,
        ))
        (self.abet_event_times,
         self.event_name,
         self.event_alias,
         self.extra_prior,
         self.extra_follow) = result

    # -----------------------------------------------------------------------
    # IO – Photometry (Doric)
    # -----------------------------------------------------------------------

    def load_doric_data(self, filepath, ch1_col, ch2_col, ttl_col, mode=''):
        """Load Doric photometry data from *filepath* (.csv or .doric). Delegates to load_signal_data."""
        self.load_signal_data(filepath, ch1_col, ch2_col, ttl_col, mode, vendor='doric')

    # -----------------------------------------------------------------------
    # IO – Synchronization
    # -----------------------------------------------------------------------

    def abet_doric_synchronize(self):
        """Align Doric time axis to ABET time via TTL cross-correlation. Delegates to synchronize_time."""
        self.synchronize_time(behaviour_vendor='abet', signal_vendor='doric')

    # -----------------------------------------------------------------------
    # Signal – crop / utilities
    # -----------------------------------------------------------------------

    def doric_crop(self, start_time_remove: float = 0, end_time_remove: float = 0):
        """Crop leading/trailing time from the photometry recording.

        Delegates to :func:`Signal.utilities.crop_signal`.
        The method name is kept for backward compatibility.
        """
        if not self.signal_loaded:
            return None
        self.signal_df = crop_signal(
            self.signal_df,
            start_time_remove=start_time_remove,
            end_time_remove=end_time_remove)

    # -----------------------------------------------------------------------
    # Signal – filter
    # -----------------------------------------------------------------------

    def signal_filter(self, filter_type='lowpass', filter_name='butterworth',
                      filter_order=4, filter_cutoff=6, fs_method='median',
                      despike=True, despike_window=2001, despike_threshold=5.0,
                      cheby_ripple=1.0):
        """Filter photometry signals and return (time, f0, f, sample_freq).

        Delegates to :func:`Signal.filter.signal_filter`.
        """
        time_data, filtered_f0, filtered_f, sample_frequency = _signal_filter(
            self.signal_df,
            filter_type=filter_type,
            filter_name=filter_name,
            filter_order=filter_order,
            filter_cutoff=filter_cutoff,
            despike=despike,
            despike_window=despike_window,
            despike_threshold=despike_threshold,
            cheby_ripple=cheby_ripple,
        )
        self.sample_frequency = sample_frequency
        return time_data, filtered_f0, filtered_f

    # Backward-compat alias
    def doric_filter(self, *args, **kwargs):
        """Deprecated alias for :meth:`signal_filter`."""
        return self.signal_filter(*args, **kwargs)

    # -----------------------------------------------------------------------
    # Signal – fit
    # -----------------------------------------------------------------------

    def signal_fit(self, fit_type, filtered_f0, filtered_f, time_data=None,
                   robust_fit=True, baseline_detrend=None, arpls_lambda=1e5, arpls_max_iter=50,
                   arpls_tol=1e-6, arpls_eps=1e-8, arpls_weight_scale=2.0,
                   huber_epsilon='auto'):
        """Fit the baseline and compute delta-F/F.

        Delegates to :func:`Signal.fitting.signal_fit`.
        Stores the result in ``self.doric_pd`` and clears raw data to free
        memory.
        """
        if time_data is None:
            # Cut based on Iso_Time and Active_Time to ensure we keep all valid samples even if they don't perfectly overlap
            signal_df_cut = self.signal_df[(self.signal_df['Time'] >= 0) ]
            # Calculate a time data based on both Iso_Time and Active_Time to ensure it covers the full range of both
            time_data = signal_df_cut['Time'].to_numpy().astype(float)

        self.doric_pd = _signal_fit(
            fit_type, filtered_f0, filtered_f, time_data,
            robust_fit=robust_fit,
            baseline_detrend=baseline_detrend,
            arpls_lambda=arpls_lambda,
            arpls_max_iter=arpls_max_iter,
            arpls_tol=arpls_tol,
            arpls_eps=arpls_eps,
            arpls_weight_scale=arpls_weight_scale,
            huber_epsilon=huber_epsilon,
        )
        # Free raw photometry memory now that delta-F is computed
        self.signal_df = pd.DataFrame()
        self.ttl_pandas   = pd.DataFrame()

    # Backward-compat alias
    def doric_fit(self, *args, **kwargs):
        """Deprecated alias for :meth:`signal_fit`."""
        return self.signal_fit(*args, **kwargs)

    def doric_process(self, filter_frequency=6):
        """Convenience wrapper: lowpass filter + linear fit."""
        time_data, filtered_f0, filtered_f = self.signal_filter(
            filter_cutoff=filter_frequency)
        self.signal_fit('linear', filtered_f0, filtered_f, time_data)

    # -----------------------------------------------------------------------
    # Process – event separation
    # -----------------------------------------------------------------------

    def trial_separator(self, trial_normalize='whole',
                        normalize_side='Left',
                        trial_iti_pad=0,
                        center_method='mean',
                        scale_median=False):
        """Build per-trial z-score and delta-F DataFrames."""
        if not self.behaviour_loaded:
            return
        self.abet_time_list = self.abet_event_times
        (self.partial_dataframe,
         self.final_dataframe,
         self.partial_deltaf,
         self.final_deltaf) = _trial_separator(
            self.abet_time_list,
            self.trial_definition_times,
            self.doric_pd,
            self.sample_frequency,
            extra_prior=self.extra_prior,
            extra_follow=self.extra_follow,
            trial_normalize=trial_normalize,
            normalize_side=normalize_side,
            trial_iti_pad=trial_iti_pad,
            center_method=center_method,
            scale_median=scale_median,
        )

    # -----------------------------------------------------------------------
    # Process – summary statistics
    # -----------------------------------------------------------------------

    def calculate_max_peak(self):
        return _calculate_max_peak(self.partial_dataframe)

    def calculate_auc(self, event_start, event_end):
        return _calculate_auc(self.partial_dataframe, event_start, event_end)

    def get_peri_event_data(self):
        return self.partial_dataframe

    # -----------------------------------------------------------------------
    # IO – Output
    # -----------------------------------------------------------------------

    def write_data(self, output_data, filename_override='', format='wide'):
        _write_data(
            output_data,
            self.main_folder_path,
            self.animal_id,
            self.date,
            self.event_name,
            self.event_alias,
            self.doric_pd,
            self.partial_dataframe,
            self.final_dataframe,
            self.partial_deltaf,
            self.final_deltaf,
            filename_override=filename_override,
            format=format,
        )

    def write_summary(self, output_data, summary_string='', output_path='',
                      session_string=''):
        _write_summary(
            output_data,
            summary_string,
            output_path,
            session_string,
            self.partial_dataframe,
            self.partial_deltaf,
            self.partial_percent,
        )


# Backward-compatibility alias so existing code using PhotometryData still works
PhotometryData = SignalEventData


# ---------------------------------------------------------------------------
# Module-level re-exports for backward compatibility
# ---------------------------------------------------------------------------
# These were previously defined at module level in data_processor.py.
# They are now implemented in their respective sub-packages and re-exported
# here so any caller that did `from data_processor import X` still works.

# (abet_extract_information, _generate_event_alias, _create_filter_list
#  are already imported at the top of this file.)


# ---------------------------------------------------------------------------
# Per-file worker  (used by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_single_file(args):
    """Module-level worker for ProcessPoolExecutor.

    Processes one ABET/Doric file pair across all events and returns a list
    of per-event result dicts.  Accepts a single tuple so it is safely
    picklable across process boundaries.
    """
    if len(args) == 36:
        (row_dict, event_sheet_path, output_options,
         event_window_prior, event_window_follow,
         trial_start_stage, trial_end_stage,
         iti_prior_trial, center_z, center_method, scale_median, normalize_side,
         filter_type, filter_name, filter_order, filter_cutoff,
         despike, despike_window, despike_threshold, cheby_ripple,
         fit_type, baseline_detrend, robust_fit, huber_epsilon,
         arpls_lambda, arpls_max_iter, arpls_tol, arpls_eps, arpls_weight_scale,
         run_flmm, run_glm_hmm, run_moa_hmm,
         exclusion_list, crop_start, crop_end, output_path) = args
    elif len(args) == 35:
        (row_dict, event_sheet_path, output_options,
         event_window_prior, event_window_follow,
         trial_start_stage, trial_end_stage,
         iti_prior_trial, center_z, center_method, scale_median, normalize_side,
         filter_type, filter_name, filter_order, filter_cutoff,
         despike, despike_window, despike_threshold, cheby_ripple,
         fit_type, baseline_detrend, robust_fit, huber_epsilon,
         arpls_lambda, arpls_max_iter, arpls_tol, arpls_eps, arpls_weight_scale,
         run_flmm, run_glm_hmm,
         exclusion_list, crop_start, crop_end, output_path) = args
        run_moa_hmm = False
    elif len(args) == 33:
        (row_dict, event_sheet_path, output_options,
         event_window_prior, event_window_follow,
         trial_start_stage, trial_end_stage,
         iti_prior_trial, center_z, center_method, scale_median, normalize_side,
         filter_type, filter_name, filter_order, filter_cutoff,
         despike, despike_window, despike_threshold, cheby_ripple,
         fit_type, baseline_detrend, robust_fit, huber_epsilon,
         arpls_lambda, arpls_max_iter, arpls_tol, arpls_eps, arpls_weight_scale,
         exclusion_list, crop_start, crop_end) = args
        run_flmm = False
        run_glm_hmm = False
        run_moa_hmm = False
        output_path = ''
    elif len(args) == 32:
        (row_dict, event_sheet_path, output_options,
         event_window_prior, event_window_follow,
         trial_start_stage, trial_end_stage,
         iti_prior_trial, center_z, center_method, scale_median, normalize_side,
         filter_type, filter_name, filter_order, filter_cutoff,
         despike, despike_window, despike_threshold, cheby_ripple,
         fit_type, baseline_detrend, robust_fit, huber_epsilon,
         arpls_lambda, arpls_max_iter, arpls_tol, arpls_eps, arpls_weight_scale,
         exclusion_list, crop_start, crop_end) = args
        run_flmm = False
        run_glm_hmm = False
        run_moa_hmm = False
        output_path = ''
    else:
        (row_dict, event_sheet_path, output_options,
         event_window_prior, event_window_follow,
         trial_start_stage, trial_end_stage,
         iti_prior_trial, center_z, center_method, scale_median, normalize_side,
         filter_type, filter_name, filter_order, filter_cutoff,
         despike, despike_window, despike_threshold, cheby_ripple,
         fit_type, baseline_detrend, robust_fit, huber_epsilon,
         arpls_lambda, arpls_max_iter, arpls_tol, arpls_eps, arpls_weight_scale,
         exclusion_list, crop_start, crop_end, output_path) = args
        run_flmm = False
        run_glm_hmm = False
        run_moa_hmm = False

    row = pd.Series(row_dict)
    file_results = []

    photometry_data = SignalEventData()
    if output_path:
        photometry_data.main_folder_path = output_path
    try:
        photometry_data.load_behaviour_data(row['abet_path'], vendor='abet')
        photometry_data.load_signal_data(
            row['doric_path'], row['ctrl_col_num'],
            row['act_col_num'], row['ttl_col_num'], row['mode'], vendor='doric')
    except UnsupportedFileFormatError as exc:
        logger.warning(
            "Skipping file pair (%s, %s): %s",
            row.get('abet_path', '?'), row.get('doric_path', '?'), exc,
        )
        return []
    except MissingColumnError as exc:
        logger.warning(
            "Skipping file pair (%s, %s) — validation failed: %s",
            row.get('abet_path', '?'), row.get('doric_path', '?'), exc,
        )
        return []
    except (FileNotFoundError, OSError) as exc:
        logger.error(
            "Cannot open file pair (%s, %s): %s",
            row.get('abet_path', '?'), row.get('doric_path', '?'), exc,
        )
        return []

    if not photometry_data.behaviour_loaded or not photometry_data.signal_loaded:
        logger.warning(
            "Skipping file pair (%s, %s): data load incomplete "
            "(behaviour_loaded=%s, signal_loaded=%s)",
            row.get('abet_path', '?'), row.get('doric_path', '?'),
            photometry_data.behaviour_loaded, photometry_data.signal_loaded,
        )
        return []

    try:
        if photometry_data.behaviour_loaded:
            photometry_data.synchronize_time(behaviour_vendor='abet', signal_vendor='doric')
    except SynchronizationError as exc:
        logger.error(
            "Time synchronization failed for file pair (%s, %s): %s - skipping.",
            row.get('abet_path', '?'), row.get('doric_path', '?'), exc,
        )
        return []

    photometry_data.doric_crop(
        start_time_remove=crop_start, end_time_remove=crop_end)

    raw_signal_df = photometry_data.signal_df.copy(deep=True)

    time_data, filtered_f0, filtered_f = photometry_data.signal_filter(
        filter_type=filter_type,
        filter_name=filter_name,
        filter_order=filter_order,
        filter_cutoff=filter_cutoff,
        despike=despike,
        despike_window=despike_window,
        despike_threshold=despike_threshold,
        cheby_ripple=cheby_ripple,
    )
    photometry_data.signal_fit(
        fit_type, filtered_f0, filtered_f, time_data,
        baseline_detrend=None,
        robust_fit=robust_fit,
        huber_epsilon=huber_epsilon,
        arpls_lambda=arpls_lambda,
        arpls_max_iter=arpls_max_iter,
        arpls_tol=arpls_tol,
        arpls_eps=arpls_eps,
        arpls_weight_scale=arpls_weight_scale,
    )
    photometry_data.abet_trial_definition(trial_start_stage, trial_end_stage)

    try:
        extract_info = BEHAVIOUR_REGISTRY['abet']['extract_info']
        animal_id, date, time, datetime_str, _ = extract_info(row['abet_path'])
    except (FileNotFoundError, OSError, IndexError, ValueError):
        animal_id = date = time = datetime_str = None

    try:
        event_sheet_df = pd.read_csv(event_sheet_path)
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError) as exc:
        logger.error(
            "Cannot read event sheet '%s': %s — skipping file pair (%s, %s)",
            event_sheet_path, exc,
            row.get('abet_path', '?'), row.get('doric_path', '?'),
        )
        return []

    session_events = []
    glm_hmm_result = None
    moa_hmm_result = None
    raw_session_id = '_'.join(str(part) for part in (animal_id, date, time) if part) or Path(row['abet_path']).stem
    session_id = _build_session_id(animal_id, date, row['abet_path'])

    for _, event_row in event_sheet_df.iterrows():
        event_alias = _generate_event_alias(event_row)

        try:
            num_filters = int(float(event_row.get('num_filter', 0)))
            if num_filters < 0:
                num_filters = 0
        except (ValueError, TypeError):
            num_filters = 0

        if num_filters == 0:
            photometry_data.abet_search_event(
                start_event_id=event_row['event_type'],
                start_event_item_name=event_row['event_name'],
                start_event_group=event_row['event_group'],
                extra_prior_time=event_window_prior,
                extra_follow_time=event_window_follow,
                exclusion_list=exclusion_list,
                event_alias=event_alias,
            )
        else:
            filter_list = _create_filter_list(event_row)
            photometry_data.abet_search_event(
                start_event_id=event_row['event_type'],
                start_event_item_name=event_row['event_name'],
                start_event_group=event_row['event_group'],
                extra_prior_time=event_window_prior,
                extra_follow_time=event_window_follow,
                exclusion_list=exclusion_list,
                event_alias=event_alias,
                filter_list=filter_list,
                filter_event=True,
            )

        if not photometry_data.abet_event_times.empty:
            event_times = photometry_data.abet_event_times.copy()
            event_times['event_alias'] = event_alias
            session_events.append(event_times)

        photometry_data.trial_separator(
            trial_normalize=center_z,
            normalize_side=normalize_side,
            trial_iti_pad=iti_prior_trial,
            center_method=center_method,
            scale_median=scale_median,
        )

        for output in output_options:
            if output <= 5:
                photometry_data.write_data(output)
            elif output in (6, 7):
                summary_type = output - 5
                summary_string = f"{event_alias}_" if event_alias else f"{event_name}_"
                session_string = f"{animal_id}_{date}" if (animal_id and date) else Path(row['abet_path']).stem
                photometry_data.write_summary(
                    summary_type,
                    summary_string=summary_string,
                    output_path=output_path,
                    session_string=session_string
                )

        max_peak = photometry_data.calculate_max_peak()
        auc      = photometry_data.calculate_auc(-event_window_prior,
                                                  event_window_follow)
        plot_df  = photometry_data.get_peri_event_data()

        try:
            plot_df_copy = plot_df.copy(deep=True)
        except (ValueError, TypeError):
            plot_df_copy = pd.DataFrame(plot_df)

        advanced_flmm_result = None
        if run_flmm:
            try:
                fixed_covariates = pd.DataFrame({
                    'trial_index': np.arange(plot_df_copy.shape[1], dtype=float),
                    'event_name': [event_alias] * plot_df_copy.shape[1],
                })
                random_effects = pd.DataFrame({
                    'animal_id': [animal_id or 'unknown_animal'] * plot_df_copy.shape[1],
                    'session_id': [raw_session_id] * plot_df_copy.shape[1],
                })
                advanced_flmm_result = run_flmm_analysis(
                    plot_df_copy,
                    covariates=fixed_covariates,
                    random_effects=random_effects,
                )
            except Exception as exc:
                logger.warning(
                    "FLMM analysis failed for file=%s behavior=%s: %s",
                    row.get('abet_path', '?'), event_alias, exc,
                )
                advanced_flmm_result = {
                    'model_type': 'flmm',
                    'status': 'failed',
                    'error': str(exc),
                }

        logger.info(
            "Processed file=%s  behavior=%s  plot_shape=%s",
            row.get('abet_path', '?'), event_alias, plot_df_copy.shape,
        )

        file_results.append({
            'file':      os.path.basename(row['abet_path']),
            'behavior':  event_alias,
            'max_peak':  max_peak,
            'auc':       auc,
            'plot_data': plot_df_copy,
            'animal_id': animal_id,
            'date':      date,
            'time':      time,
            'datetime':  datetime_str,
            'advanced_flmm': advanced_flmm_result,
        })

    if run_glm_hmm and not photometry_data.doric_pd.empty:
        try:
            session_events_df = pd.concat(session_events, ignore_index=True) if session_events else pd.DataFrame()
            glm_hmm_result = run_glm_hmm_analysis(
                photometry_data.doric_pd['Time'].to_numpy(dtype=float, copy=False),
                photometry_data.doric_pd['DeltaF'].to_numpy(dtype=float, copy=False),
                session_events_df,
            )
        except Exception as exc:
            logger.warning(
                "GLM-HMM analysis failed for file=%s: %s",
                row.get('abet_path', '?'), exc,
            )
            glm_hmm_result = {
                'model_type': 'glm_hmm',
                'status': 'failed',
                'error': str(exc),
            }

    if run_moa_hmm and not photometry_data.doric_pd.empty:
        try:
            session_events_df = pd.concat(session_events, ignore_index=True) if session_events else pd.DataFrame()
            moa_hmm_result = run_moa_hmm_analysis(
                photometry_data.doric_pd['Time'].to_numpy(dtype=float, copy=False),
                photometry_data.doric_pd['DeltaF'].to_numpy(dtype=float, copy=False),
                agent_predictions=None,
                events=session_events_df,
            )
        except Exception as exc:
            logger.warning(
                "MoA-HMM analysis failed for file=%s: %s",
                row.get('abet_path', '?'), exc,
            )
            moa_hmm_result = {
                'model_type': 'moa_hmm',
                'status': 'failed',
                'error': str(exc),
            }

    if glm_hmm_result is not None:
        for result in file_results:
            result['advanced_glm_hmm'] = glm_hmm_result
    if moa_hmm_result is not None:
        for result in file_results:
            result['advanced_moa_hmm'] = moa_hmm_result

    trace_table = pd.DataFrame({
        'Time': np.asarray(time_data, dtype=float),
        'Raw_Control': pd.to_numeric(raw_signal_df.get('Control', pd.Series(dtype=float)), errors='coerce').reindex(range(len(time_data))).to_numpy(dtype=float, copy=False)
        if len(raw_signal_df) == len(time_data)
        else np.interp(np.asarray(time_data, dtype=float), raw_signal_df['Time'].to_numpy(dtype=float, copy=False), pd.to_numeric(raw_signal_df['Control'], errors='coerce').to_numpy(dtype=float, copy=False)),
        'Raw_Active': pd.to_numeric(raw_signal_df.get('Active', pd.Series(dtype=float)), errors='coerce').reindex(range(len(time_data))).to_numpy(dtype=float, copy=False)
        if len(raw_signal_df) == len(time_data)
        else np.interp(np.asarray(time_data, dtype=float), raw_signal_df['Time'].to_numpy(dtype=float, copy=False), pd.to_numeric(raw_signal_df['Active'], errors='coerce').to_numpy(dtype=float, copy=False)),
        'Filtered_Control': np.asarray(filtered_f0, dtype=float),
        'Filtered_Active': np.asarray(filtered_f, dtype=float),
        'DeltaF': photometry_data.doric_pd['DeltaF'].to_numpy(dtype=float, copy=False),
    })
    event_table = pd.concat(session_events, ignore_index=True) if session_events else pd.DataFrame(columns=['time', 'event_type'])
    if not event_table.empty:
        event_table = event_table.copy()
        if 'Start_Time' in event_table.columns and 'time' not in event_table.columns:
            event_table['time'] = pd.to_numeric(event_table['Start_Time'], errors='coerce')
        if 'event_alias' in event_table.columns and 'event_type' not in event_table.columns:
            event_table['event_type'] = event_table['event_alias'].astype(str)

    session_trace_record = {
        'session_id': session_id,
        'animal_id': animal_id,
        'date': date,
        'time': time,
        'datetime': datetime_str,
        'source_file': os.path.basename(row['abet_path']),
        'trace_table': trace_table,
        'event_table': event_table,
    }

    for result in file_results:
        result['session_trace'] = session_trace_record

    return file_results


# ---------------------------------------------------------------------------
# Main batch-processing entry point
# ---------------------------------------------------------------------------

def process_files(file_sheet_path, event_sheet_path, output_options,
                  config, num_workers=1, progress_callback=None):
    """Process all file pairs defined in *file_sheet_path* and persist to HDF5.

    Parameters
    ----------
    file_sheet_path : str
    event_sheet_path : str
    output_options : list[int]
    config : ConfigManager
    num_workers : int
        >=2 enables ProcessPoolExecutor parallelism.

    Returns
    -------
    str  Absolute path to the HDF5 results store.
    """
    file_pair_df = pd.read_csv(file_sheet_path)

    event_window_prior  = float(config['Event_Window']['event_prior'])
    event_window_follow = float(config['Event_Window']['event_follow'])

    trial_start_stage = [i.strip() for i in
                         (config['ABET'].get('trial_start_stage') or '').split(',')
                         if i.strip()]
    trial_end_stage   = [i.strip() for i in
                         (config['ABET'].get('trial_end_stage') or '').split(',')
                         if i.strip()]

    iti_prior_trial = float(config['Normalization']['iti_prior_trial'])
    center_z        = config['Normalization']['center_z']
    center_method   = config['Normalization']['center_method']
    scale_median    = bool(config['Normalization'].get('scale_median', False))
    normalize_side  = config['Normalization'].get(
        'normalize_side',
        'End' if str(center_z).strip().lower() == 'iti' else 'Left',
    )

    filter_type   = config['Signal_Filter']['filter_type']
    filter_name   = config['Signal_Filter']['filter_name']
    filter_order  = int(config['Signal_Filter']['filter_order'])
    filter_cutoff = int(config['Signal_Filter']['filter_cutoff'])
    cheby_ripple      = float(config['Signal_Filter'].get('cheby_ripple', 1.0))
    despike           = bool(config['Signal_Utilities'].get('despike', True))
    despike_window    = int(config['Signal_Utilities'].get('despike_window', 2001))
    despike_threshold = float(config['Signal_Utilities'].get('despike_threshold', 5.0))
    crop_start        = float(config['Signal_Utilities'].get('crop_start', 0.0))
    crop_end          = float(config['Signal_Utilities'].get('crop_end', 0.0))
    fit_type          = config['Signal_Fitting']['fit_type']
    baseline_detrend = config['Signal_Fitting'].get('baseline_detrend', None)
    robust_fit        = bool(config['Signal_Fitting'].get('robust_fit', True))
    huber_epsilon     = config['Signal_Fitting'].get('huber_epsilon', 'auto')
    arpls_lambda      = float(config['Signal_Fitting'].get('arpls_lambda', 1e5))
    arpls_max_iter    = int(config['Signal_Fitting'].get('arpls_max_iter', 50))
    arpls_tol         = float(config['Signal_Fitting'].get('arpls_tol', 1e-6))
    arpls_eps         = float(config['Signal_Fitting'].get('arpls_eps', 1e-8))
    arpls_weight_scale = float(config['Signal_Fitting'].get('arpls_weight_scale', 2.0))
    advanced_analysis = config['Advanced_Analysis'] if 'Advanced_Analysis' in config else {}
    run_flmm = _as_bool(advanced_analysis.get('run_flmm', False))
    run_glm_hmm = _as_bool(advanced_analysis.get('run_glm_hmm', False))
    run_moa_hmm = _as_bool(advanced_analysis.get('run_moa_hmm', False))

    exclusion_list = [i.strip() for i in
                      (config['ABET'].get('exclusion_list') or '').split(',')
                      if i.strip()]

    output_path = ''
    if 'Filepath' in config:
        output_path = config['Filepath'].get('output_path', '')

    args_list = [
        (
            row.to_dict(), event_sheet_path, output_options,
            event_window_prior, event_window_follow,
            trial_start_stage, trial_end_stage,
            iti_prior_trial, center_z, center_method, scale_median, normalize_side,
            filter_type, filter_name, filter_order, filter_cutoff,
            despike, despike_window, despike_threshold, cheby_ripple,
            fit_type, baseline_detrend, robust_fit, huber_epsilon,
            arpls_lambda, arpls_max_iter, arpls_tol, arpls_eps, arpls_weight_scale,
            run_flmm, run_glm_hmm, run_moa_hmm,
            exclusion_list, crop_start, crop_end, output_path,
        )
        for _, row in file_pair_df.iterrows()
    ]

    try:
        config_text = config.to_json_string()
    except (AttributeError, TypeError) as exc:
        logger.warning("Could not serialise config to JSON: %s", exc)
        config_text = ''

    hdf5_path = hdf_store.initialize_results_file(
        hdf_store.get_default_results_path(),
        metadata={
            'file_sheet_path':  file_sheet_path,
            'event_sheet_path': event_sheet_path,
            'event_prior':      event_window_prior,
            'event_follow':     event_window_follow,
            'config_text':      config_text,
        },
    )

    index_records  = []
    result_counter = 0
    saved_sessions = set()

    def persist_result_batch(result_batch):
        nonlocal result_counter
        for result in result_batch:
            result_counter += 1
            result_id = f"result_{result_counter:06d}"
            result['result_id'] = result_id
            hdf_store.append_result(hdf5_path, result_id, result)
            session_trace = result.get('session_trace')
            if isinstance(session_trace, dict):
                session_id = str(session_trace.get('session_id') or '')
                if session_id and session_id not in saved_sessions:
                    hdf_store.save_session_traces(hdf5_path, session_id, session_trace)
                    saved_sessions.add(session_id)
            index_records.append({
                'result_id': result_id,
                'file':      result.get('file'),
                'behavior':  result.get('behavior'),
                'animal_id': result.get('animal_id'),
                'date':      result.get('date'),
                'time':      result.get('time'),
                'datetime':  result.get('datetime'),
                'session':   result.get('session', ''),
                'max_peak':  result.get('max_peak'),
                'auc':       result.get('auc'),
            })

    completed_jobs = 0

    if num_workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_process_single_file, args): i
                           for i, args in enumerate(args_list)}
                for future in as_completed(futures):
                    try:
                        persist_result_batch(future.result())
                    except Exception as exc:
                        logger.error(
                            "Worker error for file pair %d: %s",
                            futures[future], exc, exc_info=True,
                        )
                    finally:
                        completed_jobs += 1
                        if progress_callback:
                            try:
                                progress_callback(completed_jobs, len(args_list))
                            except Exception as cb_exc:
                                logger.warning("Progress callback failed: %s", cb_exc)
        except Exception as exc:
            logger.error(
                "ProcessPoolExecutor failed (%s) — falling back to sequential processing.",
                exc, exc_info=True,
            )
            for args in args_list:
                try:
                    persist_result_batch(_process_single_file(args))
                except Exception as fallback_exc:
                    logger.error("Sequential fallback error: %s", fallback_exc, exc_info=True)
                finally:
                    completed_jobs += 1
                    if progress_callback:
                        try:
                            progress_callback(completed_jobs, len(args_list))
                        except Exception as cb_exc:
                            logger.warning("Progress callback failed: %s", cb_exc)
    else:
        for args in args_list:
            try:
                persist_result_batch(_process_single_file(args))
            except Exception as exc:
                logger.error("Processing error: %s", exc, exc_info=True)
            finally:
                completed_jobs += 1
                if progress_callback:
                    try:
                        progress_callback(completed_jobs, len(args_list))
                    except Exception as cb_exc:
                        logger.warning("Progress callback failed: %s", cb_exc)

    # ------------------------------------------------------------------
    # Assign session numbers per animal based on chronological order
    # ------------------------------------------------------------------
    animal_sessions = {}
    for record in index_records:
        aid = record.get('animal_id')
        dt  = record.get('datetime')
        if aid and dt:
            animal_sessions.setdefault(aid, set()).add(dt)

    animal_session_map = {}
    for aid, dt_set in animal_sessions.items():
        try:
            sorted_dts = sorted(
                list(dt_set),
                key=lambda x: dateutil.parser.parse(x) if x else pd.Timestamp(0))
        except (ValueError, TypeError) as exc:
            logger.debug("Could not parse datetime strings for session ordering: %s", exc)
            sorted_dts = sorted(list(dt_set))
        animal_session_map[aid] = {
            dt: f"Session {i+1}" for i, dt in enumerate(sorted_dts)}

    session_updates = {}
    for record in index_records:
        aid = record.get('animal_id')
        dt  = record.get('datetime')
        session_name = (animal_session_map.get(aid, {}).get(dt)
                        or 'Session 1')
        record['session'] = session_name
        session_updates[str(record['result_id'])] = session_name

    hdf_store.update_result_sessions(hdf5_path, session_updates)
    hdf_store.write_index(hdf5_path, index_records)

    return hdf5_path