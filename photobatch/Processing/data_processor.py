# Imports
import os
import sys
import csv
import configparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from scipy import signal
from scipy.optimize import curve_fit
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LinearRegression, HuberRegressor
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
                          exclusion_list=None):
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
        self.extra_follow = extra_follow_time
        self.extra_prior = extra_prior_time

    def abet_doric_synchronize(self):
        """ abet_doric_synchronize - This function searches for TTL timestamps in the ABET II raw data and
        relates it to TTL pulses detected in the photometer. The adjusted sync value is calculated and the 
        doric photometry data time is adjusted to be in reference to the ABET II file."""
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
        tolerance = 0.1  # 100 ms
        filtered_doric_ttl_times = []
        last_time = None
        for t in doric_ttl_times_all:
            if last_time is None or (t - last_time) > tolerance:
                filtered_doric_ttl_times.append(t)
                last_time = t
        doric_ttl_times = np.array(filtered_doric_ttl_times)

        abet_ttl_times = abet_ttl_active.iloc[:, 0].values.astype(float)
        print(doric_ttl_times)
        print(abet_ttl_times)

        # Find corresponding TTL pulses.  Assume Doric has more pulses.
        paired_doric_times = []
        paired_abet_times = []
        doric_index = 0
        for abet_time in abet_ttl_times:
            while doric_index < len(doric_ttl_times) and doric_ttl_times[doric_index] < abet_time:
                doric_index += 1
                print([doric_ttl_times[doric_index], abet_time])
            if doric_index < len(doric_ttl_times):
                paired_doric_times.append(doric_ttl_times[doric_index])
                paired_abet_times.append(abet_time)
                doric_index += 1
            else:
                break

        if not paired_doric_times:
            print("No matching TTL pulses found. Synchronization failed.")
            return

        #  Use linear regression on the *paired* TTL pulses
        # print(paired_doric_times)
        # print(paired_abet_times)
        ttl_model = LinearRegression()
        ttl_model.fit(np.array(paired_doric_times).reshape(-1, 1), paired_abet_times)
        
        # Calculate sync diagnostics: Residual Time Error
        predicted_abet = ttl_model.predict(np.array(paired_doric_times).reshape(-1, 1))
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

    def doric_filter(self, filter_type='lowpass', filter_name='butterworth', filter_order=4, filter_cutoff=6,
                     fs_method='median'):
        # Prepare data and apply selected filter (returns time and filtered signals)
        doric_pandas_cut = self.doric_pandas[self.doric_pandas['Time'] >= 0]

        time_data = doric_pandas_cut['Time'].to_numpy()
        f0_data = doric_pandas_cut['Control'].to_numpy()
        f_data = doric_pandas_cut['Active'].to_numpy()

        time_data = time_data.astype(float)
        f0_data = f0_data.astype(float)
        f_data = f_data.astype(float)

        # compute sample frequency (Hz)
        if fs_method == 'median':
            # Calculate time diff between consecutive points and take median to estimate sampling interval
            time_diffs = np.diff(time_data)
            self.sample_frequency = 1.0 / np.median(time_diffs)
        else:
            # Use first and last point to approximate - good for uniform sampling but can be skewed by outliers
            self.sample_frequency = len(time_data) / (time_data[(len(time_data) - 1)] - time_data[0])

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
                # use Chebyshev type I with default ripple of 1 dB
                sos = signal.cheby1(N=order, rp=1, Wn=cutoff, btype='lowpass', analog=False, output='sos', fs=self.sample_frequency)
            else:
                # Unknown filter name, fallback to Butterworth
                sos = signal.butter(N=order, Wn=cutoff, btype='lowpass', analog=False, output='sos', fs=self.sample_frequency)

            filtered_f0 = signal.sosfilt(sos, f0_data)
            filtered_f = signal.sosfilt(sos, f_data)

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

        print(filtered_f)
        return time_data, filtered_f0, filtered_f


    def doric_fit(self, fit_type, filtered_f0, filtered_f, time_data=None, robust_fit=True):
        """ doric_fit - This function fits the filtered photometry signals to compute delta-F/F. 
        The fit can be linear regression, exponential decay, or arPLS baseline fitting. 
        The fitted baseline is used to compute delta-F/F for the active channel.
        Arguments:
        fit_type: str The type of fit to apply. Options include 'linear', 'expodecay', 'arpls'.
        filtered_f0: np.array The filtered isobestic channel data.
        filtered_f: np.array The filtered active channel data. 
        time_data: np.array The time data corresponding to the filtered signals. If None, it will be extracted from self.doric_pandas.
        robust_fit: bool Whether to use a robust fitting method (HuberRegressor) for linear fits.
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

                # epsilon control threshold where loss switches from squared to absolute
                epsilon_val = 1.35
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
            lam = 1e5  # smoothing parameter; could be exposed to the caller
            ratio = 1e-6
            max_iter = 50

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
                C = W + H
                # Solve C z = W y
                try:
                    z = spsolve(C, w * y)
                except (RuntimeError, ValueError):
                    # fallback to dense solve if sparse solve fails
                    C_dense = (W + H).toarray()
                    z = np.linalg.solve(C_dense, w * y)

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
                # logistic(d, m, s) = 1 / (1 + exp(2*(d - (2*s - m))/s))
                # this sets weight near 1 for values below baseline and near 0 for peaks above baseline
                wt = 1.0 / (1.0 + np.exp(2.0 * (d - (2.0 * s - m)) / s))

                # enforce bounds [eps, 1-eps]
                eps = 1e-8
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
                auc_values.append(np.trapz(y=self.partial_dataframe[col], x=time_series))
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
            file_name_str = f"{output_string}-{self.animal_id} {self.date} {self.event_name}.csv"
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
def abet_extract_information(abet_file_path):
    animal_id = ''
    date = ''
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
                date = str(row[1]).replace(':', '-').replace('/', '-')
            elif (row[0] == 'Schedule') or (row[0] == 'Schedule Name'):
                schedule = str(row[1])
            elif row[0] in event_time_colname:
                break
    return animal_id, date, schedule

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
     fit_type, exclusion_list, crop_start, crop_end) = args

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
        filter_cutoff=filter_cutoff
    )
    photometry_data.doric_fit(fit_type, filtered_f0, filtered_f, time_data)
    photometry_data.abet_trial_definition(trial_start_stage, trial_end_stage)

    try:
        animal_id, date, _ = abet_extract_information(row['abet_path'])
    except (FileNotFoundError, OSError, IndexError):
        animal_id, date = None, None

    event_sheet_df = pd.read_csv(event_sheet_path)

    for _, event_row in event_sheet_df.iterrows():
        photometry_data.abet_search_event(
            start_event_id=event_row['event_type'],
            start_event_item_name=event_row['event_name'],
            start_event_group=event_row['event_group'],
            extra_prior_time=event_window_prior,
            extra_follow_time=event_window_follow,
            exclusion_list=exclusion_list
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
            print(f"Processed file={row['abet_path']} behavior={event_row['event_name']} plot_shape={plot_df_copy.shape}")
        except (KeyError, AttributeError):
            print("Processed one result (unable to format debug info)")

        file_results.append({
            "file": os.path.basename(row['abet_path']),
            "behavior": event_row['event_name'],
            "max_peak": max_peak,
            "auc": auc,
            "plot_data": plot_df_copy,
            "animal_id": animal_id,
            "date": date
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

    fit_type = config['Photometry_Processing']['fit_type']
    
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
            fit_type, exclusion_list, crop_start, crop_end
        )
        for _, row in file_pair_df.iterrows()
    ]

    results = []

    if num_workers > 1:
        # Use a thread pool for I/O and GIL-releasing numerical work.
        # ThreadPoolExecutor runs in the same process as the GUI so no new
        # windows are ever spawned.  numpy/scipy release the GIL during their
        # C-extension calls, so multiple file pairs can execute their filtering,
        # fitting, and trial-separation steps genuinely in parallel.
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_process_single_file, args): i
                           for i, args in enumerate(args_list)}
                for future in as_completed(futures):
                    try:
                        results.extend(future.result())
                    except Exception as e:
                        print(f"Worker error for file pair {futures[future]}: {e}")
        except Exception as e:
            print(f"ThreadPoolExecutor failed ({e}); falling back to sequential processing.")
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
    # Aggregate per-behavior combined results
    # ------------------------------------------------------------------
    # We will compute three types of combined results:
    # 1. Combined across ALL animals (existing behavior) -> key: 'behavior'
    # 2. Combined across ALL animals FOR each date -> key: 'behavior (Date: YYYY-MM-DD)'
    # 3. Longitudinal (difference between sorted dates for SAME animal) -> added natively to results
    
    combined_results = {}
    date_combined = {} # Group by (behavior, date)
    animal_date_map = {} # Group by (behavior, animal_id) -> list of results sorted by date
    
    for res in results:
        behavior = res['behavior']
        animal_id = res['animal_id']
        date = res['date']
        
        # 1. Base combination (All Animals)
        if behavior not in combined_results:
            combined_results[behavior] = {
                "file": "Combined",
                "behavior": behavior,
                "animal_id": None,
                "date": None,
                "auc_list": [],
                "max_peak_list": [],
                "plot_data_list": []
            }
        
        combined_results[behavior]["auc_list"].append(res["auc"])
        combined_results[behavior]["max_peak_list"].append(res["max_peak"])
        if not res["plot_data"].empty:
            mean_plot = res["plot_data"].mean(axis=1)
            combined_results[behavior]["plot_data_list"].append(mean_plot)
            
        # 2. Date grouping (Combined Animals by Date)
        if date:
            date_key = f"{behavior} (Date: {date})"
            if date_key not in date_combined:
                date_combined[date_key] = {
                    "file": f"Combined {date}",
                    "behavior": behavior,
                    "animal_id": None,
                    "date": date,
                    "auc_list": [],
                    "max_peak_list": [],
                    "plot_data_list": []
                }
            date_combined[date_key]["auc_list"].append(res["auc"])
            date_combined[date_key]["max_peak_list"].append(res["max_peak"])
            if not res["plot_data"].empty:
                date_combined[date_key]["plot_data_list"].append(mean_plot)
                
        # 3. Longitudinal preparation (Group by Animal and Behavior)
        if animal_id and date:
            anim_beh_key = (behavior, animal_id)
            if anim_beh_key not in animal_date_map:
                animal_date_map[anim_beh_key] = []
            animal_date_map[anim_beh_key].append(res)

    # Process base combinations
    for behavior, data in combined_results.items():
        if data["auc_list"]:
            data["auc"] = float(np.mean(data["auc_list"]))
            data["max_peak"] = float(np.mean(data["max_peak_list"]))
            if data["plot_data_list"]:
                data["plot_data"] = pd.concat(data["plot_data_list"], axis=1)
            else:
                data["plot_data"] = pd.DataFrame()
        else:
            data["auc"] = 0
            data["max_peak"] = 0
            data["plot_data"] = pd.DataFrame()
        del data["auc_list"], data["max_peak_list"], data["plot_data_list"]
        results.append(data)
        
    # Process date combinations
    for date_key, data in date_combined.items():
        if data["auc_list"]:
            data["auc"] = float(np.mean(data["auc_list"]))
            data["max_peak"] = float(np.mean(data["max_peak_list"]))
            if data["plot_data_list"]:
                data["plot_data"] = pd.concat(data["plot_data_list"], axis=1)
            else:
                data["plot_data"] = pd.DataFrame()
        else:
            data["auc"] = 0
            data["max_peak"] = 0
            data["plot_data"] = pd.DataFrame()
        del data["auc_list"], data["max_peak_list"], data["plot_data_list"]
        results.append(data)
        
    # Process longitudinal changes (Day N - Day N-1)
    for (behavior, animal_id), anim_results in animal_date_map.items():
        if len(anim_results) > 1:
            # Sort by date
            anim_results.sort(key=lambda x: str(x['date']))
            for i in range(1, len(anim_results)):
                prev_res = anim_results[i-1]
                curr_res = anim_results[i]
                
                diff_auc = curr_res['auc'] - prev_res['auc']
                diff_peak = curr_res['max_peak'] - prev_res['max_peak']
                
                # Difference the plot data
                diff_plot_data = pd.DataFrame()
                if not curr_res['plot_data'].empty and not prev_res['plot_data'].empty:
                    # try to subtract the mean traces
                    curr_mean = curr_res['plot_data'].mean(axis=1)
                    prev_mean = prev_res['plot_data'].mean(axis=1)
                    diff_series = curr_mean - prev_mean
                    diff_plot_data = pd.DataFrame({f"Diff {curr_res['date']} - {prev_res['date']}": diff_series})
                    
                diff_behavior_key = f"{behavior} (Δ {curr_res['date']} minus {prev_res['date']})"
                
                longitudinal_res = {
                    "file": f"Diff {curr_res['file']} - {prev_res['file']}",
                    "behavior": diff_behavior_key,
                    "animal_id": animal_id,
                    "date": f"{curr_res['date']} - {prev_res['date']}",
                    "auc": diff_auc,
                    "max_peak": diff_peak,
                    "plot_data": diff_plot_data
                }
                results.append(longitudinal_res)

    return results, combined_results