from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


SCHEMA_VERSION = "1.0"
DEFAULT_RESULTS_FILENAME = "temp.hdf5"
_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def get_project_root() -> Path:
    """Return the repository root directory as a resolved Path."""
    return Path(__file__).resolve().parents[2]


def get_default_results_path() -> Path:
    """Return the default HDF5 results file path inside the project root."""
    return get_project_root() / DEFAULT_RESULTS_FILENAME


def initialize_results_file(hdf5_path: str | Path, metadata: dict | None = None) -> str:
    """Create (or overwrite) the PhotoBatch HDF5 results store.

    Writes the schema version attribute and empty ``analysis/entries``,
    ``analysis/index``, and ``combined_results`` groups.  Any key/value
    pairs in *metadata* are stored as attributes on the ``meta`` group.

    Parameters
    ----------
    hdf5_path : str or Path
        Destination path for the HDF5 file.  Parent directories are
        created automatically.
    metadata : dict, optional
        Scalar metadata to persist (e.g. file sheet path, event prior/
        follow windows, serialised config JSON).

    Returns
    -------
    str
        Absolute string path to the created file.
    """
    path = Path(hdf5_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, 'w') as hdf5_file:
        hdf5_file.attrs['schema_version'] = SCHEMA_VERSION
        meta_group = hdf5_file.require_group('meta')
        meta_group.attrs['created_at'] = datetime.now(timezone.utc).isoformat()

        analysis_group = hdf5_file.require_group('analysis')
        analysis_group.require_group('entries')
        analysis_group.require_group('index')
        hdf5_file.require_group('combined_results')

        metadata = metadata or {}
        for key, value in metadata.items():
            if key == 'config_text':
                meta_group.create_dataset('config_text', data=str(value), dtype=_STRING_DTYPE)
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                meta_group.attrs[key] = value
            elif value is not None:
                meta_group.attrs[key] = str(value)

    return str(path)


def append_result(hdf5_path: str | Path, result_id: str, result_record: dict) -> None:
    """Append one analysis result to the HDF5 store.

    Creates (or replaces) the group ``analysis/entries/<result_id>``
    and writes per-event metadata attributes and the peri-event
    ``plot_data`` dataset.

    Parameters
    ----------
    hdf5_path : str or Path
    result_id : str
        Unique identifier for this result (e.g. ``'result_000001'``).
    result_record : dict
        Must contain: ``file``, ``behavior``, ``animal_id``, ``date``,
        ``time``, ``datetime``, ``session``, ``max_peak``, ``auc``,
        ``plot_data`` (pd.DataFrame of z-score trials).
    """
    with h5py.File(hdf5_path, 'a') as hdf5_file:
        entries_group = hdf5_file['analysis']['entries']
        if result_id in entries_group:
            del entries_group[result_id]

        entry_group = entries_group.create_group(result_id)
        entry_group.attrs['result_id'] = result_id

        string_fields = ['file', 'behavior', 'animal_id', 'date', 'time', 'datetime', 'session']
        for field in string_fields:
            entry_group.attrs[field] = _normalize_string(result_record.get(field))

        entry_group.attrs['max_peak'] = _normalize_float(result_record.get('max_peak'))
        entry_group.attrs['auc'] = _normalize_float(result_record.get('auc'))

        plot_data = result_record.get('plot_data')
        if not isinstance(plot_data, pd.DataFrame):
            plot_data = pd.DataFrame(plot_data)
        plot_data = plot_data.copy()

        plot_group = entry_group.create_group('plot_data')
        numeric_plot = plot_data.apply(pd.to_numeric, errors='coerce') if not plot_data.empty else plot_data
        values = numeric_plot.to_numpy(dtype=float, copy=True) if not numeric_plot.empty else np.empty((0, 0), dtype=float)
        plot_group.create_dataset('values', data=values, compression='gzip')

        columns = [str(column) for column in plot_data.columns]
        plot_group.create_dataset('columns', data=np.asarray(columns, dtype=object), dtype=_STRING_DTYPE)

        if plot_data.empty:
            plot_group.attrs['index_kind'] = 'numeric'
            plot_group.create_dataset('index', data=np.asarray([], dtype=float))
        else:
            numeric_index = pd.to_numeric(pd.Index(plot_data.index), errors='coerce')
            if np.isnan(numeric_index).any():
                plot_group.attrs['index_kind'] = 'string'
                plot_group.create_dataset(
                    'index',
                    data=np.asarray([str(value) for value in plot_data.index], dtype=object),
                    dtype=_STRING_DTYPE,
                )
            else:
                plot_group.attrs['index_kind'] = 'numeric'
                plot_group.create_dataset('index', data=np.asarray(numeric_index, dtype=float))


def write_index(hdf5_path: str | Path, records: Iterable[dict]) -> None:
    """Write the flat result index to ``analysis/index``.

    Replaces any existing index datasets.  Called once after all results
    have been appended so that the GUI can load the index without
    iterating every entry group.

    Parameters
    ----------
    hdf5_path : str or Path
    records : Iterable[dict]
        Sequence of result record dicts (same structure as those passed
        to :func:`append_result`).
    """
    records_list = list(records)
    columns = ['result_id', 'file', 'behavior', 'animal_id', 'date', 'time', 'datetime', 'session']

    with h5py.File(hdf5_path, 'a') as hdf5_file:
        index_group = hdf5_file['analysis']['index']
        for dataset_name in list(index_group.keys()):
            del index_group[dataset_name]

        for column in columns:
            values = [_normalize_string(record.get(column)) for record in records_list]
            index_group.create_dataset(column, data=np.asarray(values, dtype=object), dtype=_STRING_DTYPE)

        max_peak_values = np.asarray([_normalize_float(record.get('max_peak')) for record in records_list], dtype=float)
        auc_values = np.asarray([_normalize_float(record.get('auc')) for record in records_list], dtype=float)
        index_group.create_dataset('max_peak', data=max_peak_values)
        index_group.create_dataset('auc', data=auc_values)
        index_group.attrs['result_count'] = len(records_list)


def update_result_sessions(hdf5_path: str | Path, session_by_result_id: dict[str, str]) -> None:
    """Patch the ``session`` attribute on existing entry groups.

    Called after chronological session numbers are assigned so that each
    entry reflects its computed session label without requiring a full
    rewrite.

    Parameters
    ----------
    hdf5_path : str or Path
    session_by_result_id : dict[str, str]
        Mapping of ``result_id`` → session label string.
    """
    with h5py.File(hdf5_path, 'a') as hdf5_file:
        entries_group = hdf5_file['analysis']['entries']
        for result_id, session in session_by_result_id.items():
            if result_id in entries_group:
                entries_group[result_id].attrs['session'] = _normalize_string(session)


def load_store_metadata(hdf5_path: str | Path) -> dict:
    """Load top-level metadata from a PhotoBatch HDF5 file.

    Parameters
    ----------
    hdf5_path : str or Path

    Returns
    -------
    dict
        All scalar attributes from the ``meta`` group plus
        ``schema_version`` from the root attributes and, if present,
        the ``config_text`` dataset as a string.

    Raises
    ------
    ValueError
        If *hdf5_path* is not a valid PhotoBatch analysis file.
    """
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        if 'meta' not in hdf5_file:
            raise ValueError('Not a valid PhotoBatch analysis HDF5 file.')

        meta_group = hdf5_file['meta']
        metadata = {key: _decode_scalar(value) for key, value in meta_group.attrs.items()}
        metadata['schema_version'] = _decode_scalar(hdf5_file.attrs.get('schema_version', ''))

        if 'config_text' in meta_group:
            metadata['config_text'] = meta_group['config_text'].asstr()[()]

        return metadata


def load_results_index(hdf5_path: str | Path) -> pd.DataFrame:
    """Load the flat result index as a DataFrame.

    Parameters
    ----------
    hdf5_path : str or Path

    Returns
    -------
    pd.DataFrame
        Columns: ``result_id``, ``file``, ``behavior``, ``animal_id``,
        ``date``, ``time``, ``datetime``, ``session``, ``max_peak``,
        ``auc``.  Returns an empty DataFrame if no index exists.
    """
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        if 'analysis' not in hdf5_file or 'index' not in hdf5_file['analysis']:
            return pd.DataFrame(columns=['result_id', 'file', 'behavior', 'animal_id', 'date', 'time', 'datetime', 'session', 'max_peak', 'auc'])

        index_group = hdf5_file['analysis']['index']
        if 'result_id' not in index_group:
            return pd.DataFrame(columns=['result_id', 'file', 'behavior', 'animal_id', 'date', 'time', 'datetime', 'session', 'max_peak', 'auc'])

        data = {
            'result_id': _read_string_dataset(index_group['result_id']),
            'file': _read_string_dataset(index_group['file']),
            'behavior': _read_string_dataset(index_group['behavior']),
            'animal_id': _read_string_dataset(index_group['animal_id']),
            'date': _read_string_dataset(index_group['date']),
            'time': _read_string_dataset(index_group['time']),
            'datetime': _read_string_dataset(index_group['datetime']),
            'session': _read_string_dataset(index_group['session']),
            'max_peak': np.asarray(index_group['max_peak'][...], dtype=float),
            'auc': np.asarray(index_group['auc'][...], dtype=float),
        }
        return pd.DataFrame(data)


def load_plot_data(hdf5_path: str | Path, result_id: str) -> pd.DataFrame:
    """Load the peri-event plot DataFrame for a single result.

    Parameters
    ----------
    hdf5_path : str or Path
    result_id : str

    Returns
    -------
    pd.DataFrame
        Trial z-score columns reconstructed from the compressed HDF5
        dataset.
    """
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        entry_group = hdf5_file['analysis']['entries'][result_id]
        return _load_plot_data_from_entry(entry_group)


def load_plot_data_map(hdf5_path: str | Path, result_ids: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Load peri-event plot DataFrames for multiple results in one file open.

    Parameters
    ----------
    hdf5_path : str or Path
    result_ids : Iterable[str]
        Sequence of result IDs to load.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of result_id → plot DataFrame.  IDs not found in the
        store are silently omitted.
    """
    plot_data_map: dict[str, pd.DataFrame] = {}
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        entries_group = hdf5_file['analysis']['entries']
        for result_id in result_ids:
            if result_id not in entries_group:
                continue
            plot_data_map[result_id] = _load_plot_data_from_entry(entries_group[result_id])
    return plot_data_map


def _load_plot_data_from_entry(entry_group: h5py.Group) -> pd.DataFrame:
    """Reconstruct a plot DataFrame from an open HDF5 entry group.

    Parameters
    ----------
    entry_group : h5py.Group
        An open group under ``analysis/entries/<result_id>``.

    Returns
    -------
    pd.DataFrame
    """
    plot_group = entry_group['plot_data']
    values = np.asarray(plot_group['values'][...], dtype=float)
    columns = _read_string_dataset(plot_group['columns'])
    index_kind = _decode_scalar(plot_group.attrs.get('index_kind', 'numeric'))
    if index_kind == 'string':
        index = _read_string_dataset(plot_group['index'])
    else:
        index = np.asarray(plot_group['index'][...], dtype=float)
    return pd.DataFrame(values, index=index, columns=columns)


def _normalize_string(value: object) -> str:
    """Coerce *value* to a non-None string, returning '' for None/NaN."""
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except TypeError:
        pass
    return str(value)


def _normalize_float(value: object) -> float:
    """Coerce *value* to float, returning NaN for None/NaN/unconvertible."""
    try:
        if value is None or pd.isna(value):
            return float('nan')
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _read_string_dataset(dataset: h5py.Dataset) -> list[str]:
    """Read an HDF5 string dataset and return a plain Python list of str."""
    try:
        return dataset.asstr()[...].tolist()
    except AttributeError:
        values = dataset[...]
        return [value.decode('utf-8') if isinstance(value, bytes) else str(value) for value in values]


def _decode_scalar(value: object) -> object:
    """Decode a bytes scalar to str; pass through everything else."""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value