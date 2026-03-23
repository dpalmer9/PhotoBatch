# Imports
import os
import io
import csv
import configparser
import dateutil.parser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Sub-package imports
from photobatch.Processing.IO.Photometry.doric import (
    load_doric_data as _load_doric_data,
)
from photobatch.Processing.IO.Behaviour.abet import (
    load_abet_data as _load_abet_data,
    abet_extract_information,
    abet_trial_definition as _abet_trial_definition,
    abet_search_event as _abet_search_event,
)
from photobatch.Processing.IO.sync import abet_doric_synchronize as _synchronize
from photobatch.Processing.IO.output import write_data as _write_data, write_summary as _write_summary
from photobatch.Processing.Signal.utilities import despike_signal, crop_signal
from photobatch.Processing.Signal.filter import signal_filter as _signal_filter
from photobatch.Processing.Signal.fitting import signal_fit as _signal_fit
from photobatch.Processing.Process.event import (
    extract_trial_data,
    trial_separator as _trial_separator,
    calculate_max_peak as _calculate_max_peak,
    calculate_auc as _calculate_auc,
    _generate_event_alias,
    _create_filter_list,
)
from photobatch.Processing import hdf_store


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
        self.abet_loaded     = False
        self.abet_searched   = False
        self.anymaze_loaded  = False
        self.doric_loaded    = False

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
        self.doric_pandas       = pd.DataFrame()
        self.ttl_pandas         = pd.DataFrame()
        self.abet_raw_data      = pd.DataFrame()
        self.anymaze_pandas     = pd.DataFrame()
        self.abet_pandas        = pd.DataFrame()
        self.abet_event_times   = pd.DataFrame()
        self.trial_definition_times = pd.DataFrame()

    # -----------------------------------------------------------------------
    # IO – Behaviour (ABET)
    # -----------------------------------------------------------------------

    def load_abet_data(self, filepath):
        """Load ABET II / ABET Cognition data from *filepath*."""
        self.abet_file_path = filepath
        self.abet_loaded    = True
        (self.abet_pandas,
         self.animal_id,
         self.date,
         self.time_var_name,
         self.event_name_col) = _load_abet_data(filepath)

    def abet_trial_definition(self, start_event_group, end_event_group):
        """Define trial start/end windows from ABET Condition Events."""
        if not self.abet_loaded:
            return None
        self.trial_definition_times = _abet_trial_definition(
            self.abet_pandas, self.time_var_name,
            start_event_group, end_event_group)

    def abet_search_event(self, start_event_id='1', start_event_group='',
                          start_event_item_name='', start_event_position=None,
                          filter_event=False, filter_list=None,
                          extra_prior_time=0, extra_follow_time=0,
                          exclusion_list=None, event_alias=''):
        """Search ABET data for a specific event with optional filters."""
        if not self.abet_loaded:
            return
        (self.abet_event_times,
         self.event_name,
         self.event_alias,
         self.extra_prior,
         self.extra_follow) = _abet_search_event(
            self.abet_pandas, self.time_var_name, self.event_name_col,
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
        )

    # -----------------------------------------------------------------------
    # IO – Photometry (Doric)
    # -----------------------------------------------------------------------

    def load_doric_data(self, filepath, ch1_col, ch2_col, ttl_col, mode=''):
        """Load Doric photometry data from *filepath* (.csv or .doric)."""
        self.doric_loaded    = True
        self.doric_file_path = filepath
        result = _load_doric_data(filepath, ch1_col, ch2_col, ttl_col, mode)
        if result is None or result[0] is None:
            self.doric_loaded = False
            return
        self.doric_pandas, self.ttl_pandas = result

    # -----------------------------------------------------------------------
    # IO – Synchronization
    # -----------------------------------------------------------------------

    def abet_doric_synchronize(self):
        """Align Doric time axis to ABET time via TTL cross-correlation."""
        if not self.abet_loaded or not self.doric_loaded:
            return None
        self.doric_pandas = _synchronize(
            self.doric_pandas, self.ttl_pandas, self.abet_pandas)

    # -----------------------------------------------------------------------
    # Signal – crop / utilities
    # -----------------------------------------------------------------------

    def doric_crop(self, start_time_remove=0, end_time_remove=0):
        """Crop leading/trailing time from the photometry recording.

        Delegates to :func:`Signal.utilities.crop_signal`.
        The method name is kept for backward compatibility.
        """
        if not self.doric_loaded:
            return None
        self.doric_pandas = crop_signal(
            self.doric_pandas,
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
            self.doric_pandas,
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
                   robust_fit=True, arpls_lambda=1e5, arpls_max_iter=50,
                   arpls_tol=1e-6, arpls_eps=1e-8, arpls_weight_scale=2.0,
                   huber_epsilon='auto'):
        """Fit the baseline and compute delta-F/F.

        Delegates to :func:`Signal.fitting.signal_fit`.
        Stores the result in ``self.doric_pd`` and clears raw data to free
        memory.
        """
        if time_data is None:
            doric_pandas_cut = self.doric_pandas[self.doric_pandas['Time'] >= 0]
            time_data = doric_pandas_cut['Time'].to_numpy().astype(float)

        self.doric_pd = _signal_fit(
            fit_type, filtered_f0, filtered_f, time_data,
            robust_fit=robust_fit,
            arpls_lambda=arpls_lambda,
            arpls_max_iter=arpls_max_iter,
            arpls_tol=arpls_tol,
            arpls_eps=arpls_eps,
            arpls_weight_scale=arpls_weight_scale,
            huber_epsilon=huber_epsilon,
        )
        # Free raw photometry memory now that delta-F is computed
        self.doric_pandas = pd.DataFrame()
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
                        center_method='mean'):
        """Build per-trial z-score and delta-F DataFrames."""
        if not self.abet_loaded:
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
    (row_dict, event_sheet_path, output_options,
     event_window_prior, event_window_follow,
     trial_start_stage, trial_end_stage,
     iti_prior_trial, center_z, center_method,
     filter_type, filter_name, filter_order, filter_cutoff,
     despike, despike_window, despike_threshold, cheby_ripple,
     fit_type, robust_fit, huber_epsilon,
     arpls_lambda, arpls_max_iter, arpls_tol, arpls_eps, arpls_weight_scale,
     exclusion_list, crop_start, crop_end) = args

    row = pd.Series(row_dict)
    file_results = []

    photometry_data = SignalEventData()
    photometry_data.load_abet_data(row['abet_path'])
    photometry_data.load_doric_data(
        row['doric_path'], row['ctrl_col_num'],
        row['act_col_num'], row['ttl_col_num'], row['mode'])

    if photometry_data.abet_loaded:
        photometry_data.abet_doric_synchronize()

    photometry_data.doric_crop(
        start_time_remove=crop_start, end_time_remove=crop_end)

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
        animal_id, date, time, datetime_str, _ = abet_extract_information(
            row['abet_path'])
    except (FileNotFoundError, OSError, IndexError, ValueError):
        animal_id = date = time = datetime_str = None

    event_sheet_df = pd.read_csv(event_sheet_path)

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

        photometry_data.trial_separator(
            trial_normalize=center_z,
            trial_iti_pad=iti_prior_trial,
            center_method=center_method,
        )

        for output in output_options:
            if output <= 7:
                photometry_data.write_data(output)
            else:
                photometry_data.write_summary(output)

        max_peak = photometry_data.calculate_max_peak()
        auc      = photometry_data.calculate_auc(-event_window_prior,
                                                  event_window_follow)
        plot_df  = photometry_data.get_peri_event_data()

        try:
            plot_df_copy = plot_df.copy(deep=True)
        except (ValueError, TypeError):
            plot_df_copy = pd.DataFrame(plot_df)

        try:
            print(f"Processed file={row['abet_path']} "
                  f"behavior={event_alias} "
                  f"plot_shape={plot_df_copy.shape}")
        except (KeyError, AttributeError):
            print("Processed one result (unable to format debug info)")

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
        })

    return file_results


# ---------------------------------------------------------------------------
# Main batch-processing entry point
# ---------------------------------------------------------------------------

def process_files(file_sheet_path, event_sheet_path, output_options,
                  config, num_workers=1):
    """Process all file pairs defined in *file_sheet_path* and persist to HDF5.

    Parameters
    ----------
    file_sheet_path : str
    event_sheet_path : str
    output_options : list[int]
    config : configparser.ConfigParser
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
                         config['ITI_Window']['trial_start_stage'].split(',')]
    trial_end_stage   = [i.strip() for i in
                         config['ITI_Window']['trial_end_stage'].split(',')]

    iti_prior_trial = float(config['ITI_Window']['iti_prior_trial'])
    center_z        = config['ITI_Window']['center_z']
    center_method   = config['ITI_Window']['center_method']

    filter_type   = config['Photometry_Processing']['filter_type']
    filter_name   = config['Photometry_Processing']['filter_name']
    filter_order  = int(config['Photometry_Processing']['filter_order'])
    filter_cutoff = int(config['Photometry_Processing']['filter_cutoff'])

    def _get_bool(key, default='true'):
        return config['Photometry_Processing'].get(key, default).lower() \
               in ('true', '1', 'yes')

    def _get_float(key, default):
        try:
            return float(config['Photometry_Processing'].get(key, str(default)))
        except ValueError:
            return default

    def _get_int(key, default):
        try:
            return int(config['Photometry_Processing'].get(key, str(default)))
        except ValueError:
            return default

    despike           = _get_bool('despike')
    despike_window    = _get_int('despike_window', 2001)
    despike_threshold = _get_float('despike_threshold', 5.0)
    cheby_ripple      = _get_float('cheby_ripple', 1.0)
    fit_type          = config['Photometry_Processing']['fit_type']
    robust_fit        = _get_bool('robust_fit')
    huber_epsilon     = config['Photometry_Processing'].get('huber_epsilon', 'auto')
    arpls_lambda      = _get_float('arpls_lambda', 1e5)
    arpls_max_iter    = _get_int('arpls_max_iter', 50)
    arpls_tol         = _get_float('arpls_tol', 1e-6)
    arpls_eps         = _get_float('arpls_eps', 1e-8)
    arpls_weight_scale = _get_float('arpls_weight_scale', 2.0)
    crop_start        = _get_float('crop_start', 0.0)
    crop_end          = _get_float('crop_end', 0.0)

    exclusion_list = [i.strip() for i in
                      config['Filter']['exclusion_list'].split(',')]

    args_list = [
        (
            row.to_dict(), event_sheet_path, output_options,
            event_window_prior, event_window_follow,
            trial_start_stage, trial_end_stage,
            iti_prior_trial, center_z, center_method,
            filter_type, filter_name, filter_order, filter_cutoff,
            despike, despike_window, despike_threshold, cheby_ripple,
            fit_type, robust_fit, huber_epsilon,
            arpls_lambda, arpls_max_iter, arpls_tol, arpls_eps, arpls_weight_scale,
            exclusion_list, crop_start, crop_end,
        )
        for _, row in file_pair_df.iterrows()
    ]

    try:
        config_buffer = io.StringIO()
        config.write(config_buffer)
        config_text = config_buffer.getvalue()
    except Exception:
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

    def persist_result_batch(result_batch):
        nonlocal result_counter
        for result in result_batch:
            result_counter += 1
            result_id = f"result_{result_counter:06d}"
            result['result_id'] = result_id
            hdf_store.append_result(hdf5_path, result_id, result)
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

    if num_workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_process_single_file, args): i
                           for i, args in enumerate(args_list)}
                for future in as_completed(futures):
                    try:
                        persist_result_batch(future.result())
                    except Exception as e:
                        print(f"Worker error for file pair "
                              f"{futures[future]}: {e}")
        except Exception as e:
            print(f"ProcessPoolExecutor failed ({e}); "
                  f"falling back to sequential processing.")
            for args in args_list:
                try:
                    persist_result_batch(_process_single_file(args))
                except Exception as ex:
                    print(f"Sequential fallback error: {ex}")
    else:
        for args in args_list:
            try:
                persist_result_batch(_process_single_file(args))
            except Exception as e:
                print(f"Processing error: {e}")

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
        except Exception:
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