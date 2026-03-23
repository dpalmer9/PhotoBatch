"""IO/Behaviour/abet.py
Loaders and event-extraction helpers for ABET II / ABET Cognition data.

Functions
---------
load_abet_data          – parse an ABET CSV export into a DataFrame.
abet_extract_information – lightweight metadata-only parse (animal, date …).
abet_trial_definition   – build trial start/end time windows.
abet_search_event       – search for a specific event with optional filters.
_filter_event_data      – internal helper called by abet_search_event.

All public functions are pure: they accept file paths or DataFrames and return
new objects; they never mutate shared state.
"""

import csv
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_abet_data(filepath):
    """Parse an ABET II / ABET Cognition CSV export.

    Parameters
    ----------
    filepath : str
        Path to the ABET CSV file.

    Returns
    -------
    abet_pd : pd.DataFrame
        Subset of the ABET data table with columns
        [time_var_name, Evnt_ID, Item_Name, Group_ID, Arg1_Value, extra].
    animal_id : str
    date : str  (slashes/colons replaced with dashes)
    time_var_name : str  ('Evnt_Time' or 'Event_Time')
    event_name_col : str  (column name for the event-type string)
    """
    event_time_colname = ['Evnt_Time', 'Event_Time']
    abet_name_list = []
    header_idx = 0
    animal_id = ''
    date = ''
    time_var_name = ''
    event_name_col = ''

    with open(filepath, 'r', encoding='utf-8') as abet_file:
        reader = csv.reader(abet_file)
        for i, row in enumerate(reader):
            if not row:
                continue
            if row[0] == 'Animal ID':
                animal_id = str(row[1])
            elif row[0] == 'Date/Time':
                date = str(row[1]).replace(':', '-').replace('/', '-')
            elif row[0] in event_time_colname:
                header_idx     = i
                time_var_name  = row[0]
                event_name_col = row[2]
                abet_name_list = [row[0], row[1], row[2], row[3], row[5], row[8]]
                break

    abet_full_df = pd.read_csv(filepath, skiprows=header_idx, dtype=str)
    if len(abet_full_df.columns) > 8:
        abet_pd = abet_full_df.iloc[:, [0, 1, 2, 3, 5, 8]].copy()
        abet_pd.columns = abet_name_list
    else:
        abet_pd = pd.DataFrame(columns=abet_name_list)

    return abet_pd, animal_id, date, time_var_name, event_name_col


def abet_extract_information(abet_file_path):
    """Extract metadata fields from an ABET CSV without loading the full table.

    Parameters
    ----------
    abet_file_path : str

    Returns
    -------
    animal_id, date, time, datetime_str, schedule : str
    """
    event_time_colname = ['Evnt_Time', 'Event_Time']
    animal_id = date = time = datetime_str = schedule = ''

    with open(abet_file_path, 'r', encoding='utf-8') as abet_file:
        reader = csv.reader(abet_file)
        for row in reader:
            if not row:
                continue
            if row[0] == 'Animal ID':
                animal_id = str(row[1])
            elif row[0] == 'Date/Time':
                datetime_str = str(row[1])
                dt_clean = datetime_str.replace(':', '-').replace('/', '-')
                parts = dt_clean.split(' ')
                date = parts[0] if len(parts) > 0 else dt_clean
                time = parts[1] if len(parts) > 1 else ''
            elif row[0] in ('Schedule', 'Schedule Name'):
                schedule = str(row[1])
            elif row[0] in event_time_colname:
                break

    return animal_id, date, time, datetime_str, schedule


# ---------------------------------------------------------------------------
# Trial structure
# ---------------------------------------------------------------------------

def abet_trial_definition(abet_pd, time_var_name, start_event_group,
                          end_event_group):
    """Define trial start/end windows from ABET Condition Events.

    Parameters
    ----------
    abet_pd : pd.DataFrame
        Full ABET data table returned by :func:`load_abet_data`.
    time_var_name : str
        Name of the time column ('Evnt_Time' or 'Event_Time').
    start_event_group : str or list of str
        Item name(s) of the Condition Event(s) marking a trial start.
    end_event_group : str or list of str
        Item name(s) of the Condition Event(s) marking a trial end.

    Returns
    -------
    pd.DataFrame  columns=['Start_Time', 'End_Time']
    """
    if isinstance(start_event_group, list) and isinstance(end_event_group, list):
        event_group_list = start_event_group + end_event_group
        filtered_abet = abet_pd[abet_pd.Item_Name.isin(event_group_list)]
    elif isinstance(start_event_group, list):
        filtered_abet = abet_pd.loc[
            ((abet_pd['Item_Name'].isin(start_event_group)) |
             (abet_pd['Item_Name'] == str(end_event_group))) &
            (abet_pd['Evnt_ID'] == '1')]
    elif isinstance(end_event_group, list):
        filtered_abet = abet_pd.loc[
            ((abet_pd['Item_Name'] == str(start_event_group)) |
             (abet_pd['Item_Name'].isin(end_event_group))) &
            (abet_pd['Evnt_ID'] == '1')]
    else:
        filtered_abet = abet_pd.loc[
            ((abet_pd['Item_Name'] == str(start_event_group)) |
             (abet_pd['Item_Name'] == str(end_event_group))) &
            (abet_pd['Evnt_ID'] == '1')]

    filtered_abet = filtered_abet.reset_index(drop=True)

    if isinstance(start_event_group, list):
        if filtered_abet.iloc[0, 3] not in start_event_group:
            filtered_abet = filtered_abet.drop([0])
            print('First Trial Event Not Start Stage. Moving to Next Event.')
    elif isinstance(start_event_group, str):
        if filtered_abet.iloc[0, 3] != str(start_event_group):
            filtered_abet = filtered_abet.drop([0])
            print('First Trial Event Not Start Stage. Moving to Next Event.')

    trial_times  = filtered_abet.loc[:, time_var_name].reset_index(drop=True)
    start_times  = pd.to_numeric(trial_times.iloc[::2],  errors='coerce').reset_index(drop=True)
    end_times    = pd.to_numeric(trial_times.iloc[1::2], errors='coerce').reset_index(drop=True)

    trial_definition_times = pd.concat([start_times, end_times], axis=1)
    trial_definition_times.columns = ['Start_Time', 'End_Time']
    trial_definition_times = trial_definition_times.reset_index(drop=True)
    return trial_definition_times


# ---------------------------------------------------------------------------
# Event search (with filtering)
# ---------------------------------------------------------------------------

def _filter_event_data(event_data, abet_data, time_var_name, event_name_col,
                       filter_type='', filter_name='', filter_group='',
                       filter_arg='', filter_before=1, filter_eval='',
                       exclusion_list=None):
    """Apply a single filter to the set of candidate event times.

    This is an internal helper called repeatedly by :func:`abet_search_event`.

    Parameters
    ----------
    event_data : pd.Series
        Candidate event times (float).
    abet_data : pd.DataFrame
        Full ABET data table.
    time_var_name : str
    event_name_col : str
    filter_type : str
        'Condition Event' or 'Variable Event'.
    filter_name : str
    filter_group : str
    filter_arg : str
    filter_before : int or str  (1 = before, 0 = after)
    filter_eval : str
        Comparison operator or list operator ('=', '!=', '<', '<=', '>', 'inlist', 'notinlist').
    exclusion_list : list, optional

    Returns
    -------
    pd.Series   filtered event times.
    """
    if exclusion_list is None:
        exclusion_list = []

    condition_event_names = ['Condition Event']
    variable_event_names  = ['Variable Event']
    display_event_names   = ['Whisker - Display Image']

    if filter_before == 'True':
        filter_before = 1
    elif filter_before == 'False':
        filter_before = 0

    if filter_type in condition_event_names:
        filter_event_abet = abet_data.loc[
            (abet_data[event_name_col] == str(filter_type)) &
            (abet_data['Group_ID'] == str(int(filter_group))), :]
        filter_event_abet = filter_event_abet[
            ~filter_event_abet.isin(exclusion_list)]
        filter_event_abet = filter_event_abet.dropna(subset=['Item_Name'])

        for index, value in event_data.items():
            sub_values = filter_event_abet.loc[:, time_var_name].astype('float64')
            sub_values = sub_values.sub(float(value))
            filter_before = int(float(filter_before))
            if filter_before == 1:
                sub_values[sub_values > 0] = np.nan
            elif filter_before == 0:
                sub_values[sub_values < 0] = np.nan
            sub_index = sub_values.abs().idxmin(skipna=True)
            sub_null  = sub_values.isnull().sum()
            if sub_null >= sub_values.size:
                event_data[index] = np.nan
                continue
            filter_value = filter_event_abet.loc[sub_index, 'Item_Name']
            if filter_value != filter_name:
                event_data[index] = np.nan

        event_data = event_data.dropna().reset_index(drop=True)

    elif filter_type in variable_event_names:
        filter_event_abet = abet_data.loc[
            (abet_data[event_name_col] == str(filter_type)) &
            (abet_data['Item_Name'] == str(filter_name)), :]
        filter_event_abet = filter_event_abet[
            ~filter_event_abet.isin(exclusion_list)]
        filter_event_abet = filter_event_abet.dropna(subset=['Item_Name'])

        for index, value in event_data.items():
            sub_values  = filter_event_abet.loc[:, time_var_name].astype('float64')
            sub_values  = sub_values.sub(float(value))
            sub_null    = sub_values.isnull().sum()
            filter_before = int(float(filter_before))
            if sub_null >= sub_values.size:
                continue
            if filter_before == 1:
                sub_values[sub_values > 0] = np.nan
            elif filter_before == 0:
                sub_values[sub_values < 0] = np.nan
            sub_index    = sub_values.abs().idxmin(skipna=True)
            filter_value = filter_event_abet.loc[sub_index, 'Arg1_Value']

            if isinstance(filter_arg, str):
                filter_arg_test = filter_arg.replace('.', '', 1)
                if not filter_arg_test.isdigit():
                    filter_val_abet = abet_data.loc[
                        (abet_data[event_name_col] == 'Variable Event') &
                        (abet_data['Item_Name'] == str(filter_arg)), :]
                    filter_index = filter_val_abet.index
                    arg_index    = filter_index.get_indexer([sub_index], method='pad')
                    print(arg_index[0])
                    filter_arg = filter_val_abet.loc[
                        filter_val_abet.index[arg_index[0]], 'Arg1_Value']

            if ',' in str(filter_arg):
                filter_arg = str(filter_arg).split(',')

            if filter_eval == 'inlist':
                if filter_value not in filter_arg:
                    event_data[index] = np.nan
            if filter_eval == 'notinlist':
                if filter_value in filter_arg:
                    event_data[index] = np.nan
            if filter_eval == '=':
                if float(filter_value) != float(filter_arg):
                    event_data[index] = np.nan
            if filter_eval == '!=':
                if float(filter_value) == float(filter_arg):
                    event_data[index] = np.nan
            if filter_eval == '<':
                if float(filter_value) >= float(filter_arg):
                    event_data[index] = np.nan
            if filter_eval == '<=':
                if float(filter_value) > float(filter_arg):
                    event_data[index] = np.nan
            if filter_eval == '>':
                if float(filter_value) <= float(filter_arg):
                    event_data[index] = np.nan
            if filter_eval == '>=':
                if float(filter_value) < float(filter_arg):
                    event_data[index] = np.nan

        event_data = event_data.dropna().reset_index(drop=True)

    elif filter_type in display_event_names:
        return event_data

    return event_data


def abet_search_event(abet_pd, time_var_name, event_name_col,
                      start_event_id='1', start_event_group='',
                      start_event_item_name='', start_event_position=None,
                      filter_event=False, filter_list=None,
                      extra_prior_time=0, extra_follow_time=0,
                      exclusion_list=None, event_alias=''):
    """Search ABET data for a specific event and apply optional filters.

    Parameters
    ----------
    abet_pd : pd.DataFrame
        Full ABET data table returned by :func:`load_abet_data`.
    time_var_name : str
    event_name_col : str
    start_event_id : str
        Event type string (e.g. 'Condition Event').
    start_event_group : str
        Group_ID value.
    start_event_item_name : str
        Item_Name value to match.
    start_event_position : str or None
        Arg1_Value for Touch Up/Down events.
    filter_event : bool
        Apply the items in *filter_list*.
    filter_list : list of dict, optional
        Each dict has keys: Type, Name, Group, Arg, Prior, Eval.
    extra_prior_time : float
        Seconds to subtract from each event start.
    extra_follow_time : float
        Seconds to add to each event start.
    exclusion_list : list, optional
    event_alias : str
        Display label; defaults to *start_event_item_name*.

    Returns
    -------
    abet_event_times : pd.DataFrame  columns=['Start_Time', 'End_Time']
    event_name : str
    event_alias : str
    extra_prior : float
    extra_follow : float
    """
    if filter_list is None:
        filter_list = []
    if exclusion_list is None:
        exclusion_list = []

    touch_event_names = ['Touch Up Event', 'Touch Down Event',
                         'Whisker - Clear Image by Position']

    if start_event_id in touch_event_names:
        filtered_abet = abet_pd.loc[
            (abet_pd[event_name_col] == str(start_event_id)) &
            (abet_pd['Group_ID']     == str(start_event_group)) &
            (abet_pd['Item_Name']    == str(start_event_item_name)) &
            (abet_pd['Arg1_Value']   == str(start_event_position)), :]
    else:
        filtered_abet = abet_pd.loc[
            (abet_pd[event_name_col] == str(start_event_id)) &
            (abet_pd['Group_ID']     == str(start_event_group)) &
            (abet_pd['Item_Name']    == str(start_event_item_name)), :]

    event_times = filtered_abet.loc[:, time_var_name].reset_index(drop=True)
    event_times = pd.to_numeric(event_times, errors='coerce')

    if filter_event:
        for fil in filter_list:
            event_times = _filter_event_data(
                event_times, abet_pd, time_var_name, event_name_col,
                str(fil['Type']), str(fil['Name']),
                str(fil['Group']), str(fil['Arg']),
                str(fil['Prior']), str(fil['Eval']),
                exclusion_list=exclusion_list,
            )

    abet_start_times = event_times - extra_prior_time
    abet_end_times   = event_times + extra_follow_time
    abet_event_times = pd.concat([abet_start_times, abet_end_times], axis=1)
    abet_event_times.columns = ['Start_Time', 'End_Time']
    print(abet_event_times)

    resolved_alias = event_alias if event_alias else start_event_item_name

    return (abet_event_times, start_event_item_name,
            resolved_alias, extra_prior_time, extra_follow_time)
