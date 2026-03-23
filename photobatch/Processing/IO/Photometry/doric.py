"""IO/Photometry/doric.py
Loaders for Doric Neuroscience fibre-photometry data.

Supported file formats
----------------------
*.csv   – exported from Doric Studio (col_index or input_output_no modes)
*.doric – native HDF5 container (software versions 6.1.5.0, 6.3.1.0, 6.3.2.0)

All public functions are pure: they accept file paths / column identifiers and
return DataFrames; they do not mutate any shared state.
"""

import pandas as pd
import numpy as np
import h5py


def load_doric_data(filepath, ch1_col, ch2_col, ttl_col, mode=''):
    """Dispatcher: load Doric data from a CSV or native HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the Doric file (.csv or .doric).
    ch1_col : str or int
        Column identifier for the isobestic (Control) channel.
    ch2_col : str or int
        Column identifier for the active channel.
    ttl_col : str or int
        Column identifier for the TTL channel.
    mode : str
        'col_index'       – Python column indices (CSV only).
        'input_output_no' – Doric AIN/LockIn notation.

    Returns
    -------
    doric_pd : pd.DataFrame  columns=['Time','Control','Active']
    ttl_pd   : pd.DataFrame  columns=['Time','TTL']
    """
    if '.csv' in filepath:
        return load_doric_data_csv(filepath, ch1_col, ch2_col, ttl_col, mode)
    elif '.doric' in filepath:
        return load_doric_data_h5(filepath, ch1_col, ch2_col, ttl_col, mode)
    else:
        raise ValueError(f"Unsupported Doric file format: {filepath}")


def load_doric_data_csv(filepath, ch1_col, ch2_col, ttl_col, mode=''):
    """Load Doric data from a CSV export.

    Parameters
    ----------
    filepath : str
    ch1_col, ch2_col, ttl_col : str or int
        Column selectors (meaning depends on *mode*).
    mode : str
        'col_index'       – integer column indices (0-based).
        'input_output_no' – 'AIN,AOUT' notation strings.

    Returns
    -------
    doric_pd : pd.DataFrame  columns=['Time','Control','Active']
    ttl_pd   : pd.DataFrame  columns=['Time','TTL']
    """
    doric_colnames = ['Time', 'Control', 'Active']
    ttl_colnames   = ['Time', 'TTL']
    print(mode)

    if mode == 'col_index':
        doric_data = pd.read_csv(filepath, header=1)
        doric_pd   = doric_data.iloc[:, [0, int(ch1_col), int(ch2_col)]].copy()
        ttl_pd     = doric_data.iloc[:, [0, int(ttl_col)]].copy()

    elif mode == 'input_output_no':
        doric_data = pd.read_csv(filepath, header=0, skiprows=[1])
        doric_cols = doric_data.columns.tolist()

        ch1_col = ch1_col.split(',')
        iso_str = ('Analog In. | Ch.' + ch1_col[0]
                   if ch1_col[1] == '1'
                   else 'Analog In. | Ch.' + ch1_col[0] + '.' +
                        str(int(ch1_col[1]) - 1))

        ch2_col = ch2_col.split(',')
        act_str = ('Analog In. | Ch.' + ch2_col[0]
                   if ch2_col[1] == '1'
                   else 'Analog In. | Ch.' + ch2_col[0] + '.' +
                        str(int(ch2_col[1]) - 1))

        ttl_str      = 'Analog In. | Ch.' + str(ttl_col)
        iso_col_idx  = doric_cols.index(iso_str)
        act_col_idx  = doric_cols.index(act_str)
        ttl_col_idx  = doric_cols.index(ttl_str)

        doric_pd = doric_data.iloc[:, [0, iso_col_idx, act_col_idx]].copy()
        ttl_pd   = doric_data.iloc[:, [0, ttl_col_idx]].copy()

    else:
        raise ValueError(f"Unknown CSV mode: '{mode}'")

    doric_pd.columns = doric_colnames
    ttl_pd.columns   = ttl_colnames
    doric_pd = doric_pd.astype('float')
    ttl_pd   = ttl_pd.astype('float')

    return doric_pd, ttl_pd


def load_doric_data_h5(filepath, ch1_col, ch2_col, ttl_col, mode=''):
    """Load Doric data from a native HDF5 (.doric) file.

    Supports Doric Studio software versions 6.1.5.0, 6.3.1.0, and 6.3.2.0.
    Version 6.2.5.0 contains unresolvable time-series data and is skipped.

    Parameters
    ----------
    filepath : str
    ch1_col, ch2_col : str
        'AIN,AOUT' format strings (e.g. '1,1').
    ttl_col : str or int
        Analogue input channel number for TTL.
    mode : str
        Not currently used for HDF5 files; reserved for future use.

    Returns
    -------
    doric_pd : pd.DataFrame  columns=['Time','Control','Active']
    ttl_pd   : pd.DataFrame  columns=['Time','TTL']
    or (None, None) if the file version cannot be parsed.
    """
    doric_h5 = h5py.File(filepath, 'r')

    try:
        software_version = doric_h5.attrs['SoftwareVersion']
    except KeyError:
        software_version = '5'

    if software_version in ('6.3.1.0', '6.3.2.0'):
        doric_dataset = doric_h5['DataAcquisition']['FPConsole']['Signals']['Series0001']
        dataset_keys  = doric_dataset.keys()

        ch1_in  = 'AIN'        + str(ch1_col).split(',')[0].rjust(2, '0')
        ch1_out = 'LockInAOUT' + str(ch1_col).split(',')[1].rjust(2, '0')
        ch2_in  = 'AIN'        + str(ch2_col).split(',')[0].rjust(2, '0')
        ch2_out = 'LockInAOUT' + str(ch2_col).split(',')[1].rjust(2, '0')
        ttl_in  = 'AIN'        + str(ttl_col).rjust(2, '0')

        lock_time = iso_data = act_data = None
        for key in dataset_keys:
            if ch1_out in key:
                key_data = doric_dataset[ch1_out]
                if ch1_in in key_data.keys():
                    lock_time = np.array(key_data['Time'])
                    iso_data  = np.array(key_data[ch1_in])
            if ch2_out in key:
                key_data = doric_dataset[ch2_out]
                if ch2_in in key_data.keys():
                    act_data = np.array(key_data[ch2_in])

        ttl_time = ttl_data = None
        for key in doric_dataset['AnalogIn'].keys():
            if ttl_in in key:
                ttl_time = np.array(doric_dataset['AnalogIn']['Time'])
                ttl_data = np.array(doric_dataset['AnalogIn'][ttl_in])

    elif software_version == '6.1.5.0':
        doric_dataset = doric_h5['DataAcquisition']['FPConsole']['Signals']['Series0001']
        dataset_keys  = doric_dataset.keys()

        ch1_in  = 'AIN'        + str(ch1_col).split(',')[0].rjust(2, '0')
        ch1_out = 'LockInAOUT' + str(ch1_col).split(',')[1].rjust(2, '0')
        ch2_in  = 'AIN'        + str(ch2_col).split(',')[0].rjust(2, '0')
        ch2_out = 'LockInAOUT' + str(ch2_col).split(',')[1].rjust(2, '0')
        ttl_in  = 'AIN'        + str(ttl_col).rjust(2, '0')

        lock_time = iso_data = act_data = None
        for key in dataset_keys:
            if ch1_in in key and ch1_out in key:
                iso_dataset = doric_dataset[key]
                lock_time   = np.array(iso_dataset['Time'])
                iso_data    = np.array(iso_dataset['Values'])
            if ch2_in in key and ch2_out in key:
                act_dataset = doric_dataset[key]
                act_data    = np.array(act_dataset['Values'])

        ttl_time = ttl_data = None
        for key in doric_dataset['AnalogIn'].keys():
            if ttl_in in key:
                ttl_time = np.array(doric_dataset['AnalogIn']['Time'])
                ttl_data = np.array(doric_dataset['AnalogIn'][key])

    elif software_version in ('6.2.5.0', ''):
        print('The HDF5 File contains unresolvable Time Series Data, '
              'moving to next session')
        return None, None

    else:
        print(f'Unrecognised Doric software version: {software_version}')
        return None, None

    doric_pd = pd.DataFrame({'Time': lock_time, 'Control': iso_data,
                             'Active': act_data}).astype('float')
    ttl_pd   = pd.DataFrame({'Time': ttl_time, 'TTL': ttl_data}).astype('float')
    print(act_data)

    return doric_pd, ttl_pd
