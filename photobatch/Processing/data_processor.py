# Imports
import os
import sys
import csv
import configparser
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from scipy import signal
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scipy.sparse as sparse
from scipy.sparse.linalg import cg, spilu, LinearOperator
import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LinearRegression, HuberRegressor
from scipy.ndimage import median_filter
from pathlib import Path


# Classes
class PhotometryData:
    def __init__(self):

        # Initialize Folder Path Variables
        self.curr_dir = Path.cwd()
        self.main_folder_path = Path.cwd()
        self.folder_symbol = os.sep
        self.data_folder_path = self.main_folder_path / 'Data'

        self.abet_file_path = ''
        self.abet_file = ''

        self.doric_file_path = ''
        self.doric_file = ''

        self.anymaze_file_path = ''
        self.anymaze_file = ''

        # Initialize Boolean Variables

        self.abet_loaded = False
        self.abet_searched = False
        self.anymaze_loaded = False
        self.doric_loaded = False

        # Initialize Numeric Variables

        self.abet_doric_sync_value = 0
        self.anymaze_doric_sync_value = 0

        self.extra_prior = 0
        self.extra_follow = 0

        self.sample_frequency = 0
        self.doric_time = 0

        # Initialize Descriptor Variables

        self.date = None
        self.animal_id = None

        # Initialize String Variables

        self.event_name_col = ''
        self.time_var_name = ''
        self.event_name = ''
        self.event_alias = ''  # display label; may differ from event_name when filters differentiate identical events

        # Initialize List Variables

        self.abet_time_list = []

        # Initialize Data Objects (Tables, Series, etc)

        self.partial_dataframe = pd.DataFrame()
        self.final_dataframe = pd.DataFrame()
        self.partial_deltaf = pd.DataFrame()
        self.final_deltaf = pd.DataFrame()
        self.partial_percent = pd.DataFrame()
        self.final_percent = pd.DataFrame()
        self.abet_pd = pd.DataFrame()
        self.doric_pd = pd.DataFrame()
        self.doric_pandas = pd.DataFrame()
        self.ttl_pandas = pd.DataFrame()
        self.abet_raw_data = pd.DataFrame()
        self.anymaze_pandas = pd.DataFrame()
        self.abet_pandas = pd.DataFrame()
        self.abet_event_times = pd.DataFrame()
        self.trial_definition_times = pd.DataFrame()

    def load_abet_data(self, filepath):
        """ load_abet_data - Loads in the ABET unprocessed data to the PhotometryData object. Also
        extracts the animal ID and date. csv reader import necessary due to unusual structure of
        ABET II/ABET Cognition data structures. Once the standard data table is detected, a curated
        subset of columns is collected . Output is moved to pandas dataframe.
        Arguments:
        filepath = The filepath for the ABET unprocessed csv. Generated from GUI path """

        self.abet_file_path = filepath
        self.abet_loaded = True
        
        with open(self.abet_file_path, 'r', encoding='utf-8') as abet_file:
            abet_csv_reader = csv.reader(abet_file)
            header_idx = 0
            event_time_colname = ['Evnt_Time', 'Event_Time']
            abet_name_list = []
            for i, row in enumerate(abet_csv_reader):
                if not row:
                    continue
                if row[0] == 'Animal ID':
                    self.animal_id = str(row[1])
                elif row[0] == 'Date/Time':
                    self.date = str(row[1]).replace(':', '-').replace('/', '-')
                elif row[0] in event_time_colname:
                    header_idx = i
                    self.time_var_name = row[0]
                    self.event_name_col = row[2]
                    abet_name_list = [row[0], row[1], row[2], row[3], row[5], row[8]]
                    break
        
        # Vectorized metadata extraction via pd.read_csv for performance
        abet_full_df = pd.read_csv(self.abet_file_path, skiprows=header_idx, dtype=str)
        if len(abet_full_df.columns) > 8:
            self.abet_pandas = abet_full_df.iloc[:, [0, 1, 2, 3, 5, 8]].copy()
            self.abet_pandas.columns = abet_name_list
        else:
            self.abet_pandas = pd.DataFrame(columns=abet_name_list)

    def load_doric_data(self, filepath, ch1_col, ch2_col, ttl_col, mode=''):
        self.doric_loaded = True
        if '.csv' in filepath:
            self.load_doric_data_csv(filepath, ch1_col, ch2_col, ttl_col, mode)
        elif '.doric' in filepath:
            self.load_doric_data_h5(filepath, ch1_col, ch2_col, ttl_col, mode)


    def load_doric_data_csv(self, filepath, ch1_col, ch2_col, ttl_col, mode=''):
        """ load_doric_data - Loads in the doric data to the PhotometryData object. This method uses a
            simple pandas read csv function to import the data. User specified column indexes are used to grab only
            the relevant columns.
             Arguments:
             filepath = The filepath for the doric photometry csv. Generated from GUI path
             ch1_col = The column index for the isobestic channel data
             ch2_col = The column index for the active channel data
             ttl_col = The column index for the TTL data
             mode = Specifies the mode of data definition. col_index indicates that the python column indexes are used. col_name indicates
             the name of the data column is used. input_output_no indicates that two numbers separated by comma are indicating the analog input
             and output from the lockin method (single number to denote non-lockin for ttl)"""
        self.doric_file_path = filepath

        doric_colnames = ['Time', 'Control', 'Active']
        ttl_colnames = ['Time', 'TTL']
        print(mode)
        if mode == 'col_index':
            doric_data = pd.read_csv(self.doric_file_path, header=1)
            self.doric_pandas = doric_data.iloc[:, [0, int(ch1_col), int(ch2_col)]]
            self.ttl_pandas = doric_data.iloc[:, [0, int(ttl_col)]]
        elif mode == 'input_output_no':
            doric_data = pd.read_csv(self.doric_file_path, header=0, skiprows=[1])
            doric_cols = doric_data.columns
            doric_cols = doric_cols.tolist()
            ch1_col = ch1_col.split(',')
            if ch1_col[1] == '1':
                iso_str = 'Analog In. | Ch.' + ch1_col[0]
            else:
                ch1_col[1] = str(int(ch1_col[1]) - 1)
                ch1_col = '.'.join(ch1_col)
                iso_str = 'Analog In. | Ch.' + ch1_col
            if ch2_col[1] == '1':
                act_str = 'Analog In. | Ch.' + ch2_col[0]
            else:
                ch2_col[1] = str(int(ch2_col[1]) - 1)
                ch2_col = '.'.join(ch2_col)
                act_str = 'Analog In. | Ch.' + ch2_col
            ttl_str = 'Analog In. | Ch.' + str(ttl_col)
            iso_col_index = doric_cols.index(iso_str)
            act_col_index = doric_cols.index(act_str)
            ttl_col_index = doric_cols.index(ttl_str)
            self.doric_pandas = doric_data.iloc[:, [0, iso_col_index, act_col_index]]
            self.ttl_pandas = doric_data.iloc[:, [0, ttl_col_index]]



        self.doric_pandas.columns = doric_colnames
        self.ttl_pandas.columns = ttl_colnames
        self.doric_pandas = self.doric_pandas.astype('float')
        self.ttl_pandas = self.ttl_pandas.astype('float')

    def load_doric_data_h5(self, filepath, ch1_col, ch2_col, ttl_col, mode=''):
        self.doric_file_path = filepath
        doric_h5 = h5py.File(self.doric_file_path,'r')

        try:
            software_version = doric_h5.attrs['SoftwareVersion']
        except KeyError:
            software_version = '5'
        
        if software_version in ('6.3.1.0', '6.3.2.0'):
            doric_dataset = doric_h5['DataAcquisition']['FPConsole']['Signals']['Series0001']
            dataset_keys = doric_dataset.keys()
            ch1_in = 'AIN' + str(ch1_col).split(',')[0].rjust(2,'0')
            ch1_out = 'LockInAOUT' + str(ch1_col).split(',')[1].rjust(2,'0')
            ch2_in = 'AIN' + str(ch2_col).split(',')[0].rjust(2,'0')
            ch2_out = 'LockInAOUT' + str(ch2_col).split(',')[1].rjust(2,'0')
            ttl_in = 'AIN' + str(ttl_col).rjust(2,'0')
            
            for key in dataset_keys:
                if ch1_out in key:
                    key_data = doric_dataset[ch1_out]
                    if ch1_in in key_data.keys():
                        lock_time = np.array(key_data['Time'])
                        iso_data = np.array(key_data[ch1_in])
                
                if ch2_out in key:
                    key_data = doric_dataset[ch2_out]
                    if ch2_in in key_data.keys():
                        act_data = np.array(key_data[ch2_in])
            
            for key in doric_dataset['AnalogIn'].keys():
                if ttl_in in key:
                    ttl_time = np.array(doric_dataset['AnalogIn']['Time'])
                    ttl_data = np.array(doric_dataset['AnalogIn'][ttl_in])
        elif software_version == '6.1.5.0':
            doric_dataset = doric_h5['DataAcquisition']['FPConsole']['Signals']['Series0001']
            dataset_keys = doric_dataset.keys()
            ch1_in = 'AIN' + str(ch1_col).split(',')[0].rjust(2,'0')
            ch1_out = 'LockInAOUT' + str(ch1_col).split(',')[1].rjust(2,'0')
            ch2_in = 'AIN' + str(ch2_col).split(',')[0].rjust(2,'0')
            ch2_out = 'LockInAOUT' + str(ch2_col).split(',')[1].rjust(2,'0')
            ttl_in = 'AIN' + str(ttl_col).rjust(2,'0')
            
            for key in dataset_keys:
                if ch1_in in key:
                    if ch1_out in key:
                        key_name = ch1_in + 'x' + ch1_out + '-LockIn'
                        iso_dataset = doric_dataset[key]
                        lock_time = iso_dataset['Time']
                        lock_time = np.array(lock_time)
                        iso_data = iso_dataset['Values']
                        iso_data = np.array(iso_data)

                if ch2_in in key:
                    if ch2_out in key:
                        key_name = ch2_in + 'x' + ch2_out + '-LockIn'
                        act_dataset = doric_dataset[key]
                        act_data = act_dataset['Values']
                        act_data = np.array(act_data)

            ttl_keys = doric_dataset['AnalogIn'].keys()

            for key in ttl_keys:
                if ttl_in in key:
                    ttl_time = doric_dataset['AnalogIn']['Time']
                    ttl_time = np.array(ttl_time)
                    ttl_data = doric_dataset['AnalogIn'][key]
                    ttl_data = np.array(ttl_data)
        elif software_version in ('6.2.5.0',''):
            print('The HDF5 File contains unresolvable Time Series Data, moving to next session')
            return         
            
        self.doric_pandas = pd.DataFrame({'Time': lock_time, 'Control': iso_data, 'Active': act_data})
        self.ttl_pandas = pd.DataFrame({'Time': ttl_time, 'TTL': ttl_data})
        self.doric_pandas = self.doric_pandas.astype('float')
        self.ttl_pandas = self.ttl_pandas.astype('float')
        print(act_data)

        return

    def abet_trial_definition(self, start_event_group, end_event_group):
        """ abet_trial_definition - Defines a trial structure for the components of the ABET II unprocessed data.
        This method uses the Item names of Condition Events that represent the normal start and end of a trial epoch.
        This method was expanded in PhotometryBatch to allow for multiple start and end groups.
        Arguments:
        start_event_group = the name of an ABET II Condition Event that defines the start of a trial
        end_event_group = the name of an ABET II Condition Event that defines the end of a trial
        Photometry Analyzer currently only supports start group definitions.
        Photometry Batch supports multiple start and end group definitions
        MousePAD will eventually support all definitions as well as sessions with no definition"""
        if not self.abet_loaded:
            return None

        if isinstance(start_event_group, list) and isinstance(end_event_group, list):
            event_group_list = start_event_group + end_event_group
            filtered_abet = self.abet_pandas[self.abet_pandas.Item_Name.isin(event_group_list)]
        elif isinstance(start_event_group, list) and not (isinstance(end_event_group, list)):
            filtered_abet = self.abet_pandas.loc[((self.abet_pandas['Item_Name'].isin(start_event_group)) | (
                    self.abet_pandas['Item_Name'] == str(end_event_group))) & (self.abet_pandas['Evnt_ID'] == '1')]
        elif isinstance(end_event_group, list) and not (isinstance(start_event_group, list)):
            filtered_abet = self.abet_pandas.loc[((self.abet_pandas['Item_Name'] == str(start_event_group)) | (
                self.abet_pandas['Item_Name'].isin(end_event_group))) & (self.abet_pandas['Evnt_ID'] == '1')]
        else:
            filtered_abet = self.abet_pandas.loc[((self.abet_pandas['Item_Name'] == str(start_event_group)) | (
                    self.abet_pandas['Item_Name'] == str(end_event_group))) & (self.abet_pandas['Evnt_ID'] == '1')]

        filtered_abet = filtered_abet.reset_index(drop=True)
        if isinstance(start_event_group, list):
            if filtered_abet.iloc[0, 3] not in start_event_group:
                filtered_abet = filtered_abet.drop([0])
                print('First Trial Event Not Start Stage. Moving to Next Event.')
        elif isinstance(start_event_group, str):
            if filtered_abet.iloc[0, 3] != str(start_event_group):
                filtered_abet = filtered_abet.drop([0])
                print('First Trial Event Not Start Stage. Moving to Next Event.')
        trial_times = filtered_abet.loc[:, self.time_var_name]
        trial_times = trial_times.reset_index(drop=True)
        start_times = trial_times.iloc[::2]
        start_times = start_times.reset_index(drop=True)
        start_times = pd.to_numeric(start_times, errors='coerce')
        end_times = trial_times.iloc[1::2]
        end_times = end_times.reset_index(drop=True)
        end_times = pd.to_numeric(end_times, errors='coerce')
        self.trial_definition_times = pd.concat([start_times, end_times], axis=1)
        self.trial_definition_times.columns = ['Start_Time', 'End_Time']
        self.trial_definition_times = self.trial_definition_times.reset_index(drop=True)


    def abet_search_event(self, start_event_id='1', start_event_group='', start_event_item_name='',
                          start_event_position=None,
                          filter_event=False, filter_list=None, extra_prior_time=0, extra_follow_time=0,
                          exclusion_list=None, event_alias=''):
        """ abet_search_event - This function searches through the ABET unprocessed data 
        for events specified in the ABET GUI. These events can be Condition Events, Variable Events,
        Touch Up/Down Events, Input Transition On/Off Events. This function can filter primary
        events with an unlimited number of filters. The output of this function is a pandas dataframe with the
        start and end times for all the identified events with the user specified padding.
        Arguments:
        start_event_id = The numerical value in the ABET II unprocessed file denoting the type of event.
        E.g. Condition Event, Variable Event
        start_event_group = The numerical value denoting the group number as defined by the ABET II
        schedule designer
        start_event_item name = The name of the specific event in the Item Name column.
        start_event_position = A numerical value denoting the positional argument of the event in the case of a
        Touch Up/Down event
        filter_list = A list of any filters present in the batch list. List stores dictionaries for each filter.
        extra_prior_time = A float value denoting the amount of time prior to the main event to pad it by
        extra_follow_time = A float value denoting the amount of time following the maine vent to pad it by
        """

        if filter_list is None:
            filter_list = []

        def filter_event_data(event_data, abet_data, filter_type='', filter_name='', filter_group='', filter_arg='',
                              filter_before=1,filter_eval=''):
            condition_event_names = ['Condition Event']
            variable_event_names = ['Variable Event']
            display_event_names = ['Whisker - Display Image']
            if filter_before == 'True':
                filter_before = 1
            elif filter_before == 'False':
                filter_before = 0

            if filter_type in condition_event_names:
                filter_event_abet = abet_data.loc[(abet_data[self.event_name_col] == str(filter_type)) & (
                            abet_data['Group_ID'] == str(int(filter_group))), :]
                # remove any rows that are in the exclusion list
                filter_event_abet = filter_event_abet[~filter_event_abet.isin(exclusion_list)]
                filter_event_abet = filter_event_abet.dropna(subset=['Item_Name'])
                for index, value in event_data.items():
                    sub_values = filter_event_abet.loc[:, self.time_var_name]
                    sub_values = sub_values.astype(dtype='float64')
                    sub_values = sub_values.sub(float(value))
                    filter_before = int(float(filter_before))
                    if filter_before == 1:
                        sub_values[sub_values > 0] = np.nan
                    elif filter_before == 0:
                        sub_values[sub_values < 0] = np.nan
                    sub_index = sub_values.abs().idxmin(skipna=True)
                    sub_null = sub_values.isnull().sum()
                    if sub_null >= sub_values.size:
                        event_data[index] = np.nan
                        continue

                    filter_value = filter_event_abet.loc[sub_index, 'Item_Name']
                    if filter_value != filter_name:
                        event_data[index] = np.nan

                event_data = event_data.dropna()
                event_data = event_data.reset_index(drop=True)
            elif filter_type in variable_event_names:
                filter_event_abet = abet_data.loc[(abet_data[self.event_name_col] == str(filter_type)) & (
                            abet_data['Item_Name'] == str(filter_name)), :]
                filter_event_abet = filter_event_abet[~filter_event_abet.isin(exclusion_list)]
                filter_event_abet = filter_event_abet.dropna(subset=['Item_Name'])
                for index, value in event_data.items():
                    sub_values = filter_event_abet.loc[:, self.time_var_name]
                    sub_values = sub_values.astype(dtype='float64')
                    sub_values = sub_values.sub(float(value))
                    sub_null = sub_values.isnull().sum()
                    filter_before = int(float(filter_before))
                    if sub_null >= sub_values.size:
                        continue
                    if filter_before == 1:
                        sub_values[sub_values > 0] = np.nan
                    elif filter_before == 0:
                        sub_values[sub_values < 0] = np.nan
                    sub_index = sub_values.abs().idxmin(skipna=True)
                    filter_value = filter_event_abet.loc[sub_index, 'Arg1_Value']
                    if isinstance(filter_arg,str):
                        filter_arg_test = filter_arg.replace(".","",1)
                        if not filter_arg_test.isdigit():
                            filter_val_abet = abet_data.loc[(abet_data[self.event_name_col] == 'Variable Event') & (
                                abet_data['Item_Name'] == str(filter_arg)), :]
                            filter_index = filter_val_abet.index
                            arg_index = filter_index.get_indexer([sub_index], method='pad')
                            print(arg_index[0])
                            filter_arg = filter_val_abet.loc[filter_val_abet.index[arg_index[0]],'Arg1_Value']
                    
                    # Equals
                    if ',' in filter_arg:
                        filter_arg = filter_arg.split(',')
                        
                    if filter_eval == "inlist":
                        if filter_value not in filter_arg:
                            event_data[index] = np.nan
                    if filter_eval == "notinlist":
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
                    if filter_eval == '>':
                        if float(filter_value) < float(filter_arg):
                            event_data[index] = np.nan

                event_data = event_data.dropna()
                event_data = event_data.reset_index(drop=True)
            elif filter_type in display_event_names:
                return

            return event_data

        touch_event_names = ['Touch Up Event', 'Touch Down Event', 'Whisker - Clear Image by Position']

        if start_event_id in touch_event_names:
            filtered_abet = self.abet_pandas.loc[(self.abet_pandas[self.event_name_col] == str(start_event_id)) & (
                        self.abet_pandas['Group_ID'] == str(start_event_group)) &
                                                 (self.abet_pandas['Item_Name'] == str(start_event_item_name)) & (
                                                             self.abet_pandas['Arg1_Value'] ==
                                                             str(start_event_position)), :]

        else:
            filtered_abet = self.abet_pandas.loc[(self.abet_pandas[self.event_name_col] == str(start_event_id)) & (
                        self.abet_pandas['Group_ID'] == str(start_event_group)) &
                                                 (self.abet_pandas['Item_Name'] == str(start_event_item_name)), :]

        self.abet_event_times = filtered_abet.loc[:, self.time_var_name]
        self.abet_event_times = self.abet_event_times.reset_index(drop=True)
        self.abet_event_times = pd.to_numeric(self.abet_event_times, errors='coerce')

        if filter_event:
            for fil in filter_list:
                self.abet_event_times = filter_event_data(self.abet_event_times, self.abet_pandas,
                                                          str(fil['Type']), str(fil['Name']),
                                                          str(fil['Group']), str(fil['Arg']),
                                                          str(fil['Prior']), str(fil['Eval']))

        abet_start_times = self.abet_event_times - extra_prior_time
        abet_end_times = self.abet_event_times + extra_follow_time
        self.abet_event_times = pd.concat([abet_start_times, abet_end_times], axis=1)
        self.abet_event_times.columns = ['Start_Time', 'End_Time']
        print(self.abet_event_times)
        self.event_name = start_event_item_name
        self.event_alias = event_alias if event_alias else start_event_item_name
        self.extra_follow = extra_follow_time
        self.extra_prior = extra_prior_time

    def abet_doric_synchronize(self):
        """ abet_doric_synchronize - Uses cross-correlation to find the optimal global lag between
        Doric TTL pulses and ABET TTL timestamps, then pairs nearest neighbours within tolerance.
        This approach is robust to dropped / missing TTL pulses on either side.
        The paired times are fed into a linear regression whose affine transform is applied to
        self.doric_pandas['Time'] so that all subsequent operations use ABET-referenced time."""
        if not self.abet_loaded:
            return None
        if not self.doric_loaded:
            return None
        try:
            doric_ttl_active = self.ttl_pandas.loc[(self.ttl_pandas['TTL'] >= 3.00), ]
        except KeyError:
            print('No TTL Signal Detected. Ending Analysis.')
            return
        try:
            abet_ttl_active = self.abet_pandas.loc[(self.abet_pandas['Item_Name'] == 'TTL #1'), ]
        except KeyError:
            print('ABET II File missing TTL Pulse Output. Ending Analysis.')
            return

        doric_ttl_times_all = doric_ttl_active['Time'].values.astype(float)
        # Filter doric_ttl_times to only keep the first TTL in each pulse (100ms tolerance)
        pulse_tol = 0.1  # 100 ms
        filtered_doric_ttl_times = []
        last_time = None
        for t in doric_ttl_times_all:
            if last_time is None or (t - last_time) > pulse_tol:
                filtered_doric_ttl_times.append(t)
                last_time = t
        doric_ttl_times = np.array(filtered_doric_ttl_times)

        abet_ttl_times = abet_ttl_active.iloc[:, 0].values.astype(float)
        print(f"Doric TTL count: {len(doric_ttl_times)}, ABET TTL count: {len(abet_ttl_times)}")

        if len(doric_ttl_times) < 2 or len(abet_ttl_times) < 2:
            print("Not enough TTL pulses for cross-correlation synchronization.")
            return

        # --- Step 1: Build binary event vectors on a shared time grid ---
        all_times = np.concatenate([doric_ttl_times, abet_ttl_times])
        t_min, t_max = all_times.min(), all_times.max()
        # Grid resolution: 10% of the median inter-pulse interval (whichever stream is denser)
        median_ipi = min(np.median(np.diff(doric_ttl_times)), np.median(np.diff(abet_ttl_times)))
        grid_res = median_ipi * 0.1
        grid = np.arange(t_min - median_ipi, t_max + median_ipi, grid_res)

        def times_to_binary(times, grid):
            vec = np.zeros(len(grid), dtype=float)
            indices = np.searchsorted(grid, times, side='left')
            indices = np.clip(indices, 0, len(grid) - 1)
            vec[indices] = 1.0
            return vec

        doric_vec = times_to_binary(doric_ttl_times, grid)
        abet_vec = times_to_binary(abet_ttl_times, grid)

        # --- Step 2: Cross-correlate to find optimal global lag ---
        correlation = signal.correlate(doric_vec, abet_vec, mode='full')
        lags = np.arange(-(len(abet_vec) - 1), len(doric_vec))
        best_lag_idx = np.argmax(correlation)
        best_lag_samples = lags[best_lag_idx]
        best_lag_seconds = best_lag_samples * grid_res
        print(f"Cross-correlation optimal lag: {best_lag_seconds:.4f} seconds ({best_lag_samples} grid steps)")

        # --- Step 3: Shift ABET times by estimated offset, pair nearest within tolerance ---
        abet_shifted = abet_ttl_times + best_lag_seconds
        pair_tolerance = median_ipi * 0.5  # half the median inter-pulse interval

        paired_doric_times = []
        paired_abet_times = []
        used_doric = set()

        for i, at in enumerate(abet_shifted):
            # Find the closest Doric TTL to this shifted ABET time
            diffs = np.abs(doric_ttl_times - at)
            closest_idx = np.argmin(diffs)
            if diffs[closest_idx] <= pair_tolerance and closest_idx not in used_doric:
                paired_doric_times.append(doric_ttl_times[closest_idx])
                paired_abet_times.append(abet_ttl_times[i])  # use original (un-shifted) ABET time
                used_doric.add(closest_idx)

        paired_doric_times = np.array(paired_doric_times)
        paired_abet_times = np.array(paired_abet_times)

        print(f"Paired {len(paired_doric_times)} of {len(doric_ttl_times)} Doric / {len(abet_ttl_times)} ABET TTL pulses")

        if len(paired_doric_times) < 2:
            print("Not enough paired TTL pulses for linear regression. Synchronization failed.")
            return

        # --- Step 4: Linear regression on paired times ---
        ttl_model = LinearRegression()
        ttl_model.fit(paired_doric_times.reshape(-1, 1), paired_abet_times)

        # Calculate sync diagnostics: Residual Time Error
        predicted_abet = ttl_model.predict(paired_doric_times.reshape(-1, 1))
        residual_error = np.abs(paired_abet_times - predicted_abet).mean()
        print(f"Synchronization Residual Error (Mean Abs): {residual_error:.4f} seconds")
        if residual_error > 0.1:
            print("WARNING: High synchronization error detected. Sync may be mismatched.")

        self.doric_pandas['Time'] = (self.doric_pandas['Time'] * ttl_model.coef_[0]) + ttl_model.intercept_
        print(self.doric_pandas['Time'].head(20))

    def doric_crop(self, start_time_remove=0, end_time_remove=0):
        """ doric_crop - This function crops the doric data to remove unwanted time at the start and end of the recording. 
        This is done after synchronization to ensure correct time reference.
        Arguments:
        start_time_remove: float, optional
            Time in seconds to remove from the start of the recording.
        end_time_remove: float, optional
            Time in seconds to remove from the end of the recording.
        """
        if not self.doric_loaded:
            return None
        # Crop the doric data to remove unwanted time at the start and end of the recording. This is done after synchronization to ensure correct time reference.
        doric_pandas_cut = self.doric_pandas[self.doric_pandas['Time'] >= 0]
        if start_time_remove > 0:
            doric_pandas_cut = doric_pandas_cut[doric_pandas_cut['Time'] >= start_time_remove]
        if end_time_remove > 0:
            max_time = doric_pandas_cut['Time'].max()
            doric_pandas_cut = doric_pandas_cut[doric_pandas_cut['Time'] <= (max_time - end_time_remove)]
        self.doric_pandas = doric_pandas_cut.reset_index(drop=True)

    def despike_signal(self, sig_array, window=2001, threshold=5.0):
            # Calculate rolling median
            med = median_filter(sig_array, size=window)
            # Calculate rolling Median Absolute Deviation (MAD)
            mad = median_filter(np.abs(sig_array - med), size=window)
            
            # Prevent division by zero if mad is perfectly 0 in flat regions
            mad[mad == 0] = np.min(mad[mad > 0]) if np.any(mad > 0) else 1e-6
            
            # Flag outliers
            outliers = np.abs(sig_array - med) > (threshold * mad)
            
            # Interpolate outliers
            cleaned = sig_array.copy()
            if np.any(outliers):
                valid_idx = np.flatnonzero(~outliers)
                outlier_idx = np.flatnonzero(outliers)
                cleaned[outliers] = np.interp(outlier_idx, valid_idx, sig_array[valid_idx])
            
            return cleaned

    def doric_filter(self, filter_type='lowpass', filter_name='butterworth', filter_order=4, filter_cutoff=6,
                     fs_method='median', despike=True, despike_window=2001, despike_threshold=5.0,
                     cheby_ripple=1.0):
        # Prepare data and apply selected filter (returns time and filtered signals)
        doric_pandas_cut = self.doric_pandas[self.doric_pandas['Time'] >= 0]

        time_data = doric_pandas_cut['Time'].to_numpy()
        f0_data = doric_pandas_cut['Control'].to_numpy()
        f_data = doric_pandas_cut['Active'].to_numpy()

        time_data = time_data.astype(float)
        f0_data = f0_data.astype(float)
        f_data = f_data.astype(float)

        if despike:
            f0_data = self.despike_signal(f0_data, window=int(despike_window), threshold=float(despike_threshold))
            f_data = self.despike_signal(f_data, window=int(despike_window), threshold=float(despike_threshold))

        # --- Interpolation step: resample to a uniform time grid ---
        # Equipment may have intermittent sample loss, producing non-uniform spacing.
        # Filtering assumes uniform spacing, so we interpolate onto a regular grid
        # derived from the median sample interval.
        time_diffs = np.diff(time_data)
        median_dt = np.median(time_diffs)
        uniform_time = np.arange(time_data[0], time_data[-1], median_dt)

        if len(uniform_time) >= 2:
            f0_interp = interp1d(time_data, f0_data, kind='linear', fill_value='extrapolate')
            f_interp = interp1d(time_data, f_data, kind='linear', fill_value='extrapolate')
            f0_data = f0_interp(uniform_time)
            f_data = f_interp(uniform_time)
            time_data = uniform_time
            print(f"Interpolated to uniform grid: {len(time_data)} samples, dt={median_dt:.6f}s")

        # compute sample frequency (Hz) from the (now uniform) time grid
        self.sample_frequency = 1.0 / median_dt

        # Default: no filtering applied if parameters are invalid
        filtered_f0 = f0_data.copy()
        filtered_f = f_data.copy()

        filter_type_lower = str(filter_type).lower()

        if filter_type_lower in ('lowpass', 'low_pass', 'low'):
            # Design a digital low-pass filter. scipy functions accept cutoff in Hz when fs is provided.
            filt_name = str(filter_name).lower()
            order = int(filter_order) if filter_order is not None else 4
            cutoff = float(filter_cutoff)

            if filt_name == 'butterworth' or filt_name == 'butter':
                sos = signal.butter(N=order, Wn=cutoff, btype='lowpass', analog=False, output='sos', fs=self.sample_frequency)
            elif filt_name == 'bessel' or filt_name == 'bess':
                # bessel supports sos output in recent scipy
                sos = signal.bessel(N=order, Wn=cutoff, btype='lowpass', analog=False, output='sos', fs=self.sample_frequency)
            elif filt_name in ('chebychev', 'cheby', 'cheby1'):
                # use Chebyshev type I with caller-supplied ripple (dB)
                sos = signal.cheby1(N=order, rp=float(cheby_ripple), Wn=cutoff, btype='lowpass', analog=False, output='sos', fs=self.sample_frequency)
            else:
                # Unknown filter name, fallback to Butterworth
                sos = signal.butter(N=order, Wn=cutoff, btype='lowpass', analog=False, output='sos', fs=self.sample_frequency)

            filtered_f0 = signal.sosfiltfilt(sos, f0_data)
            filtered_f = signal.sosfiltfilt(sos, f_data)

        elif filter_type_lower in ('smoothing', 'smooth', 'savgol', 'savgolay'):
            # Use Savitzky-Golay smoothing filter. Interpret filter_order as window length.
            window_length = int(filter_order) if filter_order is not None else 5
            # Window length must be odd and >= 3
            if window_length < 3:
                window_length = 3
            if window_length % 2 == 0:
                window_length += 1
            # Choose polynomial order smaller than window_length
            polyorder = min(3, window_length - 1)

            try:
                filtered_f0 = signal.savgol_filter(f0_data, window_length, polyorder)
                filtered_f = signal.savgol_filter(f_data, window_length, polyorder)
            except (ValueError, TypeError):
                # If savgol fails for any reason, fall back to original data
                filtered_f0 = f0_data.copy()
                filtered_f = f_data.copy()

        else:
            # Unknown filter type: leave data unfiltered
            filtered_f0 = f0_data.copy()
            filtered_f = f_data.copy()

        return time_data, filtered_f0, filtered_f


    def doric_fit(self, fit_type, filtered_f0, filtered_f, time_data=None, robust_fit=True,
                  arpls_lambda=1e5, arpls_max_iter=50, arpls_tol=1e-6, arpls_eps=1e-8,
                  arpls_weight_scale=2.0, huber_epsilon='auto'):
        """ doric_fit - This function fits the filtered photometry signals to compute delta-F/F. 
        The fit can be linear regression, exponential decay, or arPLS baseline fitting. 
        The fitted baseline is used to compute delta-F/F for the active channel.
        Arguments:
        fit_type: str The type of fit to apply. Options include 'linear', 'expodecay', 'arpls'.
        filtered_f0: np.array The filtered isobestic channel data.
        filtered_f: np.array The filtered active channel data. 
        time_data: np.array The time data corresponding to the filtered signals. If None, it will be extracted from self.doric_pandas.
        robust_fit: bool Whether to use a robust fitting method (HuberRegressor) for linear fits.
        huber_epsilon: str or float  Controls the epsilon parameter for HuberRegressor.
            - 'auto' or 'mad': calculate epsilon from the Median Absolute Deviation of the
              session's noise floor (residuals between isobestic and active channels).
            - A numeric value (float): use the value directly as epsilon.
            Default: 'auto'.
        """
        
        # Fit filtered signals, compute delta-F and populate self.doric_pd
        fit_type_lower = str(fit_type).lower() if fit_type is not None else 'linear'

        if time_data is None:
            doric_pandas_cut = self.doric_pandas[self.doric_pandas['Time'] >= 0]
            time_data = doric_pandas_cut['Time'].to_numpy().astype(float)

        # Linear fit: regress active on isobestic
        if fit_type_lower in ('linear', 'lin'):
            if robust_fit:
                # Reshape 1D Arrays to 2D for Scikit-learn
                f0_reshapped = filtered_f0.reshape(-1, 1)

                # Determine epsilon for HuberRegressor
                epsilon_str = str(huber_epsilon).strip().lower()
                if epsilon_str in ('auto', 'mad'):
                    # Calculate epsilon from the Median Absolute Deviation of the noise floor
                    residuals = filtered_f - np.median(filtered_f)
                    mad_val = np.median(np.abs(residuals - np.median(residuals)))
                    # Scale MAD to approximate standard deviation: sigma ≈ 1.4826 * MAD
                    epsilon_val = 1.4826 * mad_val
                    # HuberRegressor requires epsilon > 1.0
                    epsilon_val = max(1.01, epsilon_val)
                    print(f"Huber epsilon (auto/MAD): {epsilon_val:.4f}")
                else:
                    try:
                        epsilon_val = float(huber_epsilon)
                        epsilon_val = max(1.01, epsilon_val)
                    except (ValueError, TypeError):
                        epsilon_val = 1.35  # fallback default
                    print(f"Huber epsilon (user-specified): {epsilon_val:.4f}")

                # Fit a HuberRegressor which is more robust to outlying values
                huber = HuberRegressor(epsilon=epsilon_val)
                huber.fit(f0_reshapped, filtered_f)
                fitted = huber.predict(f0_reshapped)
            else:
                filtered_poly = np.polyfit(filtered_f0, filtered_f, 1)
                fitted = np.multiply(filtered_poly[0], filtered_f0) + filtered_poly[1]

        elif fit_type_lower in ('expodecay', 'exp_decay', 'exp'):
            # Fit an exponential decay to the active signal over time: y = A * exp(-k*t) + C
            y = filtered_f
            t = time_data.astype(float)
            try:
                A0 = np.max(y) - np.min(y)
                k0 = 1.0 / max((t[-1] - t[0]), 1.0)
                C0 = np.min(y)
                popt, _ = curve_fit(lambda tt, A, k, C: A * np.exp(-k * tt) + C, t, y, p0=[A0, k0, C0], maxfev=10000)
                fitted = popt[0] * np.exp(-popt[1] * t) + popt[2]
            except (RuntimeError, TypeError, ValueError):
                # Fallback to linear regression between channels if exponential fit fails
                filtered_poly = np.polyfit(filtered_f0, filtered_f, 1)
                fitted = np.multiply(filtered_poly[0], filtered_f0) + filtered_poly[1]

        elif fit_type_lower in ('arpls', 'ar_pls', 'ar-pls'):
            # Asymmetrically reweighted penalized least squares baseline fitting (arPLS)
            # Implements the algorithm by Baek et al. (2015) / variants using iterative reweighting.
            y = filtered_f.astype(float)
            n = y.size
            lam = float(arpls_lambda)
            ratio = float(arpls_tol)
            max_iter = int(arpls_max_iter)
            eps = float(arpls_eps)
            weight_scale = float(arpls_weight_scale)

            # Construct second-difference matrix D (size (n-2) x n)
            e = np.ones(n)
            D = sparse.diags([e, -2*e, e], [0, 1, 2], shape=(n-2, n))
            H = lam * (D.transpose().dot(D))  # (n x n) sparse

            # initial weights
            w = np.ones(n)
            W = sparse.diags(w, 0)

            z = np.zeros(n)
            for i in range(max_iter):
                W = sparse.diags(w, 0)
                C = (W + H).tocsc()  # CSC required by spilu
                # Build a sparse ILU preconditioner once per iteration
                try:
                    ilu = spilu(C, fill_factor=2)
                    M = LinearOperator(C.shape, ilu.solve)
                except RuntimeError:
                    M = None  # fall back to unpreconditioned CG
                # Solve C z ≈ W y with Conjugate Gradient (stays fully sparse)
                z, info = cg(C, w * y, x0=z, M=M, atol=1e-8)
                if info != 0:
                    print(f"arPLS CG solver did not converge at iteration {i} (info={info})")

                d = y - z
                # statistics of negative residuals
                d_neg = d[d < 0]
                if d_neg.size == 0:
                    break
                m = d_neg.mean()
                s = d_neg.std()
                if s <= 0:
                    break
                # logistic weighting as described in arPLS paper
                # weight_scale controls sharpness of the logistic transition around the baseline
                wt = 1.0 / (1.0 + np.exp(weight_scale * (d - (2.0 * s - m)) / s))

                # enforce bounds [eps, 1-eps]
                wt = np.clip(wt, eps, 1.0)

                # convergence check
                if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
                    w = wt
                    break

                w = wt

            fitted = z

        else:
            # Unknown fit type: fallback to linear
            filtered_poly = np.polyfit(filtered_f0, filtered_f, 1)
            fitted = np.multiply(filtered_poly[0], filtered_f0) + filtered_poly[1]

        # Compute delta-F relative to the fitted baseline
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            delta_f = (filtered_f - fitted) / fitted
            delta_f = np.nan_to_num(delta_f)
            print(delta_f)

        self.doric_pd = pd.DataFrame(time_data)
        self.doric_pd['DeltaF'] = delta_f
        self.doric_pd = self.doric_pd.rename(columns={0: 'Time', 1: 'DeltaF'})

        import gc
        self.doric_pandas = pd.DataFrame()
        self.ttl_pandas = pd.DataFrame()
        gc.collect()

    def extract_trial_data(self, doric_time_array, doric_deltaf_array, start_time, end_time, expected_samples=None):
        """
        Helper method to replace the while loop logic in trial_separator.
        Extracts data strictly within time bounds.
        Arguments:
        doric_time_array: np.array of time points from doric_pd
        doric_deltaf_array: np.array of delta-F values from doric_pd
        start_time: float, start time of the trial
        end_time: float, end time of the trial
        expected_samples: int, optional, expected number of samples in the trial
        """
        # 1. Fast boundary lookup using binary search
        start_index = np.searchsorted(doric_time_array, start_time, side='left')
        end_index = np.searchsorted(doric_time_array, end_time, side='right')

        # 2. Safety bounds check
        if start_index >= len(doric_time_array) or end_index <= 0 or start_index >= end_index:
            return np.array([]), np.array([])

        # 3. Slice the actual data
        trial_time = doric_time_array[start_index:end_index]
        trial_deltaf = doric_deltaf_array[start_index:end_index]

        # 4. Enforce uniform array lengths via interpolation (No while loops)
        if expected_samples is not None and len(trial_time) != expected_samples:
            # Generate a perfect linear time grid for the exact window
            target_time = np.linspace(start_time, end_time, expected_samples)
            
            # Interpolate the delta_f values to match the target time grid
            trial_deltaf = np.interp(target_time, trial_time, trial_deltaf)
            trial_time = target_time

        return trial_time, trial_deltaf


    def doric_process(self, filter_frequency=6):
        # Backwards-compatible wrapper: run standard lowpass (butterworth) filter then linear fit
        time_data, filtered_f0, filtered_f = self.doric_filter(filter_cutoff=filter_frequency)
        self.doric_fit('linear', filtered_f0, filtered_f, time_data)

    """trial_separator - This function takes the extracted photometry data and parses it using the event data obtained
        from the previous functions. This function will check to make sure the events are the same length. 
        This function will also calculate the z-scores using either the entire event or the time prior to the start of a 
        trial (i.e. iti). The result of this function are a series of pandas dataframes corresponding to the different
        output types in the write_data function.
        Arguments:
        whole_trial_normalize = A boolean value to determine whether to use the whole event to generate z-scores
        normalize_side = Denotes whether to use the pre or post trial data to normalize if not using whole trial.
        trial_definition = A flag to indicate whether a trial definition exists or not
        trial_iti_pad = How long in the pre-trial time space for normalization
        center_method = Determine whether median or mean is used to generate z-values"""

    def trial_separator(self, trial_normalize='whole', normalize_side='Left',
                        trial_iti_pad=0,
                        center_method='mean'):
        if not self.abet_loaded:
            return
        left_selection_list = ['Left', 'Before', 'L', 'l', 'left', 'before', 1]
        right_selection_list = ['Right', 'right', 'R', 'r', 'After', 'after', 2]

        trial_num = 1

        self.abet_time_list = self.abet_event_times
        # doric_time_series not used further; keep doric_time_array and delta arrays for slicing
        doric_time_series = self.doric_pd[['Time']]

        zscore_trials = []
        deltaf_trials = []
        time_trials = []

        doric_time_array = self.doric_pd['Time'].values  # For fast searchsorted
        doric_deltaf_array = self.doric_pd['DeltaF'].values

        # Convert doric_pd columns to numpy arrays for fast slicing
        doric_time_array = self.doric_pd['Time'].values
        doric_deltaf_array = self.doric_pd['DeltaF'].values

        # Prepare lists to collect trial data
        time_trials = []
        zscore_trials = []
        deltaf_trials = []

        for row in self.abet_time_list.itertuples():
            index = row.Index
            try:
                start_time = float(row.Start_Time)
                start_index = np.searchsorted(doric_time_array, start_time, side='left')
                if start_index >= len(doric_time_array):
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
            except (IndexError, ValueError, TypeError):
                print('Trial Start Out of Bounds or invalid, Skipping Event')
                continue
            try:
                end_time = float(row.End_Time)
                end_index = np.searchsorted(doric_time_array, end_time, side='right')
                if end_index < 0 or end_index >= len(doric_time_array):
                    print('Trial End Out of Bounds, Skipping Event')
                    continue
            except (IndexError, ValueError, TypeError):
                print('Trial End Out of Bounds or invalid, Skipping Event')
                continue


            # Compute the expected number of measurements for this trial from its duration
            try:
                length_time = end_time - start_time
                measurements_per_interval = max(1, int(round(length_time * self.sample_frequency)))
            except (ValueError, TypeError, ArithmeticError):
                # Fallback to at least a single measurement
                measurements_per_interval = 1

            trial_time, trial_deltaf = self.extract_trial_data(doric_time_array, 
                                                               doric_deltaf_array, 
                                                               start_time, 
                                                               end_time, 
                                                               expected_samples=measurements_per_interval)

            # Compute z-score normalization as before
            if trial_normalize == 'iti':
                if normalize_side in left_selection_list:
                    trial_start_index_diff = self.trial_definition_times.loc[:, 'Start_Time'].sub(
                        (self.abet_time_list.loc[index, 'Start_Time'] + self.extra_prior))
                    trial_start_index_diff[trial_start_index_diff > 0] = np.nan
                    trial_start_index = trial_start_index_diff.abs().idxmin(skipna=True)
                    trial_start_window = self.trial_definition_times.iloc[trial_start_index, 0]
                    trial_iti_window = trial_start_window - float(trial_iti_pad)
                    iti_mask = (doric_time_array >= trial_iti_window) & (doric_time_array <= trial_start_window)
                    iti_data = doric_deltaf_array[iti_mask]
                else:
                    trial_end_index = self.trial_definition_times.loc[:, 'End_Time'].sub(
                        self.abet_time_list.loc[index, 'End_Time']).abs().idxmin()
                    trial_end_window = self.trial_definition_times.iloc[trial_end_index, 0]
                    trial_iti_window = trial_end_window + trial_iti_pad
                    iti_mask = (doric_time_array >= trial_end_window) & (doric_time_array <= trial_iti_window)
                    iti_data = doric_deltaf_array[iti_mask]

                if center_method == 'mean':
                    z_mean = iti_data.mean() if len(iti_data) > 0 else 0
                    z_sd = iti_data.std() if len(iti_data) > 0 else 1
                elif center_method == 'median':
                    z_mean = np.median(iti_data) if len(iti_data) > 0 else 0
                    z_dev = np.abs(iti_data - z_mean)
                    z_sd = np.median(z_dev) if len(z_dev) > 0 else 1
            elif trial_normalize == 'prior':
                if normalize_side in left_selection_list:
                    baseline_mask = (trial_time >= self.abet_time_list.loc[index, 'Start_Time']) & \
                                    (trial_time <= (self.abet_time_list.loc[index, 'Start_Time']) + self.extra_prior)
                    baseline_data = trial_deltaf[baseline_mask]
                else:
                    baseline_mask = ((trial_time >= self.abet_time_list.loc[index, 'End_Time']) - self.extra_follow) & \
                                    (trial_time <= trial_iti_window)
                    baseline_data = trial_deltaf[baseline_mask]

                if center_method == 'mean':
                    z_mean = baseline_data.mean() if len(baseline_data) > 0 else 0
                    z_sd = baseline_data.std() if len(baseline_data) > 0 else 1
                elif center_method == 'median':
                    z_mean = np.median(baseline_data) if len(baseline_data) > 0 else 0
                    z_dev = np.abs(baseline_data - z_mean)
                    z_sd = np.median(z_dev) if len(z_dev) > 0 else 1
            elif trial_normalize == 'whole':
                if center_method == 'mean':
                    z_mean = trial_deltaf.mean() if len(trial_deltaf) > 0 else 0
                    z_sd = trial_deltaf.std() if len(trial_deltaf) > 0 else 1
                elif center_method == 'median':
                    z_mean = np.median(trial_deltaf) if len(trial_deltaf) > 0 else 0
                    z_dev = np.abs(trial_deltaf - z_mean)
                    z_sd = np.median(z_dev) if len(z_dev) > 0 else 1

            # Calculate zscore for this trial
            zscore = (trial_deltaf - z_mean) / z_sd if z_sd != 0 else trial_deltaf - z_mean

            # Store for later DataFrame construction
            time_trials.append(trial_time)
            zscore_trials.append(zscore)
            deltaf_trials.append(trial_deltaf)

            trial_num += 1

        # After loop, build DataFrames from lists
        # Pad arrays to the same length for DataFrame construction
        max_len = max(len(arr) for arr in time_trials) if time_trials else 0
        def pad(arr):
            return np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
        time_df = pd.DataFrame({f'Time Trial {i+1}': pad(arr) for i, arr in enumerate(time_trials)})
        zscore_df = pd.DataFrame({f'Z-Score Trial {i+1}': pad(arr) for i, arr in enumerate(zscore_trials)})
        deltaf_df = pd.DataFrame({f'Delta-F Trial {i+1}': pad(arr) for i, arr in enumerate(deltaf_trials)})

        # Set self.partial_dataframe, self.final_dataframe, etc. as before
        self.partial_dataframe = zscore_df
        self.partial_deltaf = deltaf_df
        self.final_dataframe = pd.concat([time_df, zscore_df], axis=1)
        self.final_deltaf = pd.concat([time_df, deltaf_df], axis=1)

    def calculate_max_peak(self):
        if not self.partial_dataframe.empty:
            # Calculate the max peak for each event (column) and then average these peaks.
            return self.partial_dataframe.max().mean()
        return 0

    def calculate_auc(self, event_start, event_end):
        if not self.partial_dataframe.empty:
            # Calculate the area under the curve (sum) for each event (column) and then average these AUCs.
            auc_values = []
            time_series = np.linspace(event_start, event_end, num=len(self.partial_dataframe))
            for col in self.partial_dataframe.columns:
                auc_values.append(np.trapezoid(y=self.partial_dataframe[col], x=time_series))
            return np.mean(auc_values)
        return 0

    def get_peri_event_data(self):
        return self.partial_dataframe

    def write_data(self, output_data, filename_override='', format='wide'):
        partial_list = [1, 'SimpleZ', 'simple']
        final_list = [2, 'TimedZ', 'timed']
        partialf_list = [3, 'SimpleF', 'simplef']
        finalf_list = [4, 'TimedF', 'timedf']
        processed_list = [5, 'Full', 'full']

        output_string_list = ['SimpleZ','TimedZ','SimpleF','TimedF','Raw']

        output_string = output_string_list[output_data-1]


        output_folder = self.main_folder_path / 'Output'
        output_folder.mkdir(exist_ok=True)
        if self.abet_loaded is True and self.anymaze_loaded is False:
            label = self.event_alias if self.event_alias else self.event_name
            file_name_str = f"{output_string}-{self.animal_id} {self.date} {label}.csv"
            file_path_string = str(output_folder / file_name_str)
        else:
            current_time = datetime.now()
            current_time_string = current_time.strftime('%d-%m-%Y %H-%M-%S')
            file_name_str = f"{output_string}-{current_time_string}.csv"
            file_path_string = str(output_folder / file_name_str)

        if filename_override != '':
            file_path_string = filename_override + '-' + output_string + '.csv'

        print(file_path_string)
        if format == 'long':
            output_long = None # Hold variable for converted data (default is wide format)
            if output_data in processed_list:
                output_long = self.doric_pd.melt(var_name='Time', value_name='DeltaF')
            elif output_data in partial_list:
                output_long = self.partial_dataframe.melt(var_name='Trial', value_name='Z-Score')
            elif output_data in final_list:
                output_long = self.final_dataframe.melt(var_name='Trial', value_name='Z-Score')
            elif output_data in partialf_list:
                output_long = self.partial_deltaf.melt(var_name='Trial', value_name='DeltaF')
            elif output_data in finalf_list:
                output_long = self.final_deltaf.melt(var_name='Trial', value_name='DeltaF')
            output_long.to_csv(file_path_string, index=False)
        else:
            if output_data in processed_list:
                self.doric_pd.to_csv(file_path_string, index=False)
            elif output_data in partial_list:
                self.partial_dataframe.to_csv(file_path_string, index=False)
            elif output_data in final_list:
                self.final_dataframe.to_csv(file_path_string, index=False)
            elif output_data in partialf_list:
                self.partial_deltaf.to_csv(file_path_string, index=False)
            elif output_data in finalf_list:
                self.final_deltaf.to_csv(file_path_string, index=False)

    def write_summary(self, output_data, summary_string, output_path, session_string):

        z_list = [6,'SummaryZ','summaryz']
        f_list = [7,'SummaryF', 'summaryf']
        p_list = [8,'SummaryP','summaryp']

        output_string_list = ['SummaryZ','SummaryF','SummaryP']

        output_string = output_string_list[output_data-1]

        summary_path = output_path + summary_string + 'Summary' + '-' + output_string + '-' +'.xlsx'

        if output_data in z_list:
            session_temp = self.partial_dataframe.transpose()
        if output_data in f_list:
            session_temp = self.partial_deltaf.transpose()
        if output_data in p_list:
            session_temp = self.partial_percent.transpose()

        session_mean = pd.DataFrame([[session_string] + session_temp.mean(axis=0, skipna=True).tolist()])
        session_std = pd.DataFrame([[session_string] + session_temp.std(axis=0, skipna=True).tolist()])
        session_sem = pd.DataFrame([[session_string] + session_temp.sem(axis=0, skipna=True).tolist()])


        if os.path.exists(summary_path):
            summary_xlsx = pd.read_excel(summary_path, sheet_name=None, header=None)
            xlsx_mean = summary_xlsx.get('Mean')
            xlsx_std = summary_xlsx.get('Std')
            xlsx_sem = summary_xlsx.get('Sem')
            
            xlsx_mean = pd.concat([xlsx_mean,session_mean])
            xlsx_std = pd.concat([xlsx_std,session_std])
            xlsx_sem = pd.concat([xlsx_sem,session_sem])

        else:

            xlsx_mean = session_mean
            xlsx_std = session_std
            xlsx_sem = session_sem

        with pd.ExcelWriter(summary_path) as writer:
            xlsx_mean.to_excel(writer, sheet_name='Mean', header=False, 
                               index=False)
            xlsx_std.to_excel(writer, sheet_name='Std', header=False, 
                               index=False)
            xlsx_sem.to_excel(writer, sheet_name='Sem', header=False, 
                               index=False)




# Functions
def _generate_event_alias(event_row):
    """Return a label that uniquely identifies this event row.

    Priority:
    1. The explicit ``event_alias`` column value (if the column exists and the
       value is non-empty / not NaN).
    2. An auto-generated string built from ``event_name`` and each active
       filter's name, evaluation operator, and argument, e.g.::

           "Display Sample [_Trial_Counter>9 & _Trial_Counter<=49]"

    The first filter uses bare column names (``filter_name``, ``filter_eval``,
    ``filter_arg``) and subsequent filters append an integer suffix
    (``filter_name2``, …).
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
        fname = str(event_row.get(f'filter_name{suffix}', '')).strip()
        feval = str(event_row.get(f'filter_eval{suffix}', '')).strip()
        farg  = str(event_row.get(f'filter_arg{suffix}',  '')).strip()
        if fname and feval and farg and fname.lower() not in ('nan', 'none'):
            filter_parts.append(f"{fname}{feval}{farg}")

    if filter_parts:
        return f"{base} [{' & '.join(filter_parts)}]"
    return base


def abet_extract_information(abet_file_path):
    animal_id = ''
    date = ''
    time = ''
    datetime_str = ''
    schedule = ''
    abet_file_path = abet_file_path
    event_time_colname = ['Evnt_Time', 'Event_Time']
    with open(abet_file_path, 'r', encoding='utf-8') as abet_file:
        abet_csv_reader = csv.reader(abet_file)
        for row in abet_csv_reader:
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
            elif (row[0] == 'Schedule') or (row[0] == 'Schedule Name'):
                schedule = str(row[1])
            elif row[0] in event_time_colname:
                break
    return animal_id, date, time, datetime_str, schedule

def _create_filter_list(event_row):
    """Create a list of filter dicts from the event_row, based on the num_filter column and corresponding filter_nameN, filter_evalN, filter_argN columns."""
    num_filters = event_row.get('num_filter', 0)
    # Loop across filter numbers and construct dicts for each valid filter
    filter_list = []
    for fil in range(1, num_filters + 1):
        # Legacy Check: Check if filter_type is existing column, if so modify fil_mod to '' for backward compatibility
        if event_row.get('filter_type', None) is not None and fil == 1:
            fil_mod = ''
        else:
            fil_mod = str(fil)
        fil_type_str = 'filter_type' + fil_mod
        fil_name_str = 'filter_name' + fil_mod
        fil_group_str = 'filter_group' + fil_mod
        fil_arg_str = 'filter_arg' + fil_mod
        fil_eval_str = 'filter_eval' + fil_mod
        fil_prior_str = 'filter_prior' + fil_mod

        fil_dict = {'Type': event_row[fil_type_str], 'Name': event_row[fil_name_str],
                            'Group': str(int(event_row[fil_group_str])), 'Arg': event_row[fil_arg_str],
                            'Prior': event_row[fil_prior_str], 'Eval': event_row[fil_eval_str]}
        filter_list.append(fil_dict)
    return filter_list

def _process_single_file(args):
    """Module-level worker for ProcessPoolExecutor. Processes one ABET/Doric file pair
    across all events and returns a list of per-event result dicts.

    Accepts a single tuple argument so it works cleanly with executor.map / executor.submit.
    All config values are passed as primitive Python types so the tuple is safely picklable
    across process boundaries.
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

    photometry_data = PhotometryData()
    photometry_data.load_abet_data(row['abet_path'])
    photometry_data.load_doric_data(
        row['doric_path'], row['ctrl_col_num'],
        row['act_col_num'], row['ttl_col_num'], row['mode']
    )
    if photometry_data.abet_loaded:
        photometry_data.abet_doric_synchronize()

    photometry_data.doric_crop(start_time_remove=crop_start, end_time_remove=crop_end)

    time_data, filtered_f0, filtered_f = photometry_data.doric_filter(
        filter_type=filter_type,
        filter_name=filter_name,
        filter_order=filter_order,
        filter_cutoff=filter_cutoff,
        despike=despike,
        despike_window=despike_window,
        despike_threshold=despike_threshold,
        cheby_ripple=cheby_ripple,
    )
    photometry_data.doric_fit(fit_type, filtered_f0, filtered_f, time_data,
                              robust_fit=robust_fit,
                              huber_epsilon=huber_epsilon,
                              arpls_lambda=arpls_lambda,
                              arpls_max_iter=arpls_max_iter,
                              arpls_tol=arpls_tol,
                              arpls_eps=arpls_eps,
                              arpls_weight_scale=arpls_weight_scale)
    photometry_data.abet_trial_definition(trial_start_stage, trial_end_stage)

    try:
        animal_id, date, time, datetime_str, _ = abet_extract_information(row['abet_path'])
    except (FileNotFoundError, OSError, IndexError, ValueError):
        animal_id, date, time, datetime_str = None, None, None, None

    event_sheet_df = pd.read_csv(event_sheet_path)

    for _, event_row in event_sheet_df.iterrows():
        # Resolve the display label for this event (user-provided or auto-generated).
        event_alias = _generate_event_alias(event_row)

        # Check if event_row[num_filters] is a valid integer > 0, otherwise treat as 0
        try:
            num_filters = int(float(event_row.get('num_filter', 0)))
            if num_filters < 0:
                num_filters = 0
        except (ValueError, TypeError):
            num_filters = 0

        # Conduct non-filtered search event if num_filters is 0
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
        # Conduct filtered search event if num_filters > 0, apply filters as list to abet_search_event
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
                filter_event = True
                )
        photometry_data.trial_separator(
            trial_normalize=center_z,
            trial_iti_pad=iti_prior_trial,
            center_method=center_method
        )

        for output in output_options:
            if output <= 7:
                photometry_data.write_data(output)
            else:
                photometry_data.write_summary(output)

        max_peak = photometry_data.calculate_max_peak()
        auc = photometry_data.calculate_auc(-event_window_prior, event_window_follow)

        plot_df = photometry_data.get_peri_event_data()
        try:
            plot_df_copy = plot_df.copy(deep=True)
        except (ValueError, TypeError):
            plot_df_copy = pd.DataFrame(plot_df)

        try:
            print(f"Processed file={row['abet_path']} behavior={event_alias} plot_shape={plot_df_copy.shape}")
        except (KeyError, AttributeError):
            print("Processed one result (unable to format debug info)")

        file_results.append({
            "file": os.path.basename(row['abet_path']),
            "behavior": event_alias,
            "max_peak": max_peak,
            "auc": auc,
            "plot_data": plot_df_copy,
            "animal_id": animal_id,
            "date": date,
            "time": time,
            "datetime": datetime_str
        })

    return file_results


def process_files(file_sheet_path, event_sheet_path, output_options, config, num_workers=1):
    """Process all file pairs defined in *file_sheet_path* and return
    ``(results, combined_results)``.

    Parameters
    ----------
    file_sheet_path : str
        Path to the CSV file-pair sheet.
    event_sheet_path : str
        Path to the CSV event sheet.
    output_options : list[int]
        Indices of selected output types.
    config : configparser.ConfigParser
        Loaded application configuration.
    num_workers : int
        Number of parallel worker processes.  Use 1 for sequential (default).
        Values > 1 spawn a ``ProcessPoolExecutor`` so each file pair is handled
        by an independent OS process, giving true CPU-level parallelism for the
        filtering, fitting, and trial-separation steps.

    Returns
    -------
    tuple[list[dict], dict[str, dict]]
        *results* – flat list of per-event result dicts (individual animals plus
        combined sentinel entries with ``animal_id=None``).
        *combined_results* – mapping from behavior name to its aggregated result
        dict, ready for direct use in the visualization tab.
    """
    file_pair_df = pd.read_csv(file_sheet_path)

    event_window_prior = float(config['Event_Window']['event_prior'])
    event_window_follow = float(config['Event_Window']['event_follow'])

    trial_start_stage = [item.strip() for item in config['ITI_Window']['trial_start_stage'].split(',')]
    trial_end_stage = [item.strip() for item in config['ITI_Window']['trial_end_stage'].split(',')]

    iti_prior_trial = float(config['ITI_Window']['iti_prior_trial'])
    center_z = config['ITI_Window']['center_z']
    center_method = config['ITI_Window']['center_method']

    filter_type = config['Photometry_Processing']['filter_type']
    filter_name = config['Photometry_Processing']['filter_name']
    filter_order = int(config['Photometry_Processing']['filter_order'])
    filter_cutoff = int(config['Photometry_Processing']['filter_cutoff'])

    despike_str = config['Photometry_Processing'].get('despike', 'true')
    despike = despike_str.lower() in ('true', '1', 'yes')
    try:
        despike_window = int(config['Photometry_Processing'].get('despike_window', '2001'))
    except ValueError:
        despike_window = 2001
    try:
        despike_threshold = float(config['Photometry_Processing'].get('despike_threshold', '5.0'))
    except ValueError:
        despike_threshold = 5.0
    try:
        cheby_ripple = float(config['Photometry_Processing'].get('cheby_ripple', '1.0'))
    except ValueError:
        cheby_ripple = 1.0

    fit_type = config['Photometry_Processing']['fit_type']

    robust_fit_str = config['Photometry_Processing'].get('robust_fit', 'true')
    robust_fit = robust_fit_str.lower() in ('true', '1', 'yes')

    huber_epsilon = config['Photometry_Processing'].get('huber_epsilon', 'auto')

    try:
        arpls_lambda = float(config['Photometry_Processing'].get('arpls_lambda', '1e5'))
    except ValueError:
        arpls_lambda = 1e5
    try:
        arpls_max_iter = int(config['Photometry_Processing'].get('arpls_max_iter', '50'))
    except ValueError:
        arpls_max_iter = 50
    try:
        arpls_tol = float(config['Photometry_Processing'].get('arpls_tol', '1e-6'))
    except ValueError:
        arpls_tol = 1e-6
    try:
        arpls_eps = float(config['Photometry_Processing'].get('arpls_eps', '1e-8'))
    except ValueError:
        arpls_eps = 1e-8
    try:
        arpls_weight_scale = float(config['Photometry_Processing'].get('arpls_weight_scale', '2.0'))
    except ValueError:
        arpls_weight_scale = 2.0
    
    try:
        crop_start = float(config['Photometry_Processing'].get('crop_start', '0.0'))
    except ValueError:
        crop_start = 0.0
        
    try:
        crop_end = float(config['Photometry_Processing'].get('crop_end', '0.0'))
    except ValueError:
        crop_end = 0.0

    exclusion_list = [item.strip() for item in config['Filter']['exclusion_list'].split(',')]

    # Build one args-tuple per file pair (all primitives – safe to pickle).
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
            exclusion_list, crop_start, crop_end
        )
        for _, row in file_pair_df.iterrows()
    ]

    results = []

    if num_workers > 1:
        # Use a process pool for true multi-core parallelism.
        # ProcessPoolExecutor runs each file pair in a separate OS process,
        # bypassing the Python GIL and allowing CPU-bound tasks (filtering, fitting)
        # to execute genuinely in parallel.
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_process_single_file, args): i
                           for i, args in enumerate(args_list)}
                for future in as_completed(futures):
                    try:
                        results.extend(future.result())
                    except Exception as e:
                        print(f"Worker error for file pair {futures[future]}: {e}")
        except Exception as e:
            print(f"ProcessPoolExecutor failed ({e}); falling back to sequential processing.")
            for args in args_list:
                try:
                    results.extend(_process_single_file(args))
                except Exception as ex:
                    print(f"Sequential fallback error: {ex}")
    else:
        # Sequential path – used when num_workers == 1 or as the fallback.
        for args in args_list:
            try:
                results.extend(_process_single_file(args))
            except Exception as e:
                print(f"Processing error: {e}")

    # ------------------------------------------------------------------
    # Pre-process sessions and dates based on original datetime
    # ------------------------------------------------------------------
    import dateutil.parser
    animal_sessions = {}
    for res in results:
        aid = res.get('animal_id')
        dt = res.get('datetime')
        if aid and dt:
            if aid not in animal_sessions:
                animal_sessions[aid] = set()
            animal_sessions[aid].add(dt)
            
    animal_session_map = {}
    for aid, dt_set in animal_sessions.items():
        try:
            sorted_dts = sorted(list(dt_set), key=lambda x: dateutil.parser.parse(x) if x else pd.Timestamp(0))
        except Exception:
            sorted_dts = sorted(list(dt_set)) # fallback to string sort
        animal_session_map[aid] = {dt: f"Session {i+1}" for i, dt in enumerate(sorted_dts)}
        
    for res in results:
        aid = res.get('animal_id')
        dt = res.get('datetime')
        if aid and dt and aid in animal_session_map and dt in animal_session_map[aid]:
            res['session'] = animal_session_map[aid][dt]
        else:
            res['session'] = "Session 1"

    # We removed predefined combined lists because they cause redundancy in the UI. 
    # The UI will dynamically generate any combinations and alignments required.
    combined_results = {}

    return results, combined_results