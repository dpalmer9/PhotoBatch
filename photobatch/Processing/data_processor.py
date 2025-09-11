# Imports
import os
import sys
import csv
import configparser
from datetime import datetime
from scipy import signal
from scipy.optimize import curve_fit
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LinearRegression

pd.set_option('mode.chained_assignment', None)


# Classes
class PhotometryData:
    def __init__(self):

        # Initialize Folder Path Variables
        self.curr_dir = os.getcwd()
        if sys.platform == 'linux' or sys.platform == 'darwin':
            self.folder_symbol = '/'
        elif sys.platform == 'win32':
            self.folder_symbol = '\\'
        self.main_folder_path = os.getcwd()
        self.data_folder_path = self.main_folder_path + self.folder_symbol + 'Data' + self.folder_symbol

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
        self.anymaze_event_times = []

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

    """ load_abet_data - Loads in the ABET unprocessed data to the PhotometryData object. Also
    extracts the animal ID and date. csv reader import necessary due to unusual structure of
    ABET II/ABET Cognition data structures. Once the standard data table is detected, a curated
    subset of columns is collected . Output is moved to pandas dataframe.
     Arguments:
     filepath = The filepath for the ABET unprocessed csv. Generated from GUI path """

    def load_abet_data(self, filepath):
        self.abet_file_path = filepath
        self.abet_loaded = True
        abet_file = open(self.abet_file_path)
        abet_csv_reader = csv.reader(abet_file)
        abet_data_list = list()
        abet_name_list = list()
        event_time_colname = ['Evnt_Time', 'Event_Time']
        colnames_found = False
        for row_csv in abet_csv_reader:
            if not colnames_found:
                if len(row_csv) == 0:
                    continue
                if row_csv[0] == 'Animal ID':
                    self.animal_id = str(row_csv[1])
                    continue
                if row_csv[0] == 'Date/Time':
                    self.date = str(row_csv[1])
                    self.date = self.date.replace(':', '-')
                    self.date = self.date.replace('/', '-')
                    continue
                if row_csv[0] in event_time_colname:
                    colnames_found = True
                    self.time_var_name = row_csv[0]
                    self.event_name_col = row_csv[2]
                    # Columns are 0-time, 1-Event ID, 2-Event name, 3-Item Name, 5-Group ID, 8-Arg-1 Value
                    abet_name_list = [row_csv[0], row_csv[1], row_csv[2], row_csv[3], row_csv[5], row_csv[8]]
                else:
                    continue
            else:
                abet_data_list.append([row_csv[0], row_csv[1], row_csv[2], row_csv[3], row_csv[5], row_csv[8]])
        abet_file.close()
        abet_numpy = np.array(abet_data_list)
        self.abet_pandas = pd.DataFrame(data=abet_numpy, columns=abet_name_list)

    def load_doric_data(self, filepath, ch1_col, ch2_col, ttl_col, mode=''):
        self.doric_loaded = True
        if '.csv' in filepath:
            self.load_doric_data_csv(filepath, ch1_col, ch2_col, ttl_col, mode)
        elif '.doric' in filepath:
            self.load_doric_data_h5(filepath, ch1_col, ch2_col, ttl_col, mode)

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

    def load_doric_data_csv(self, filepath, ch1_col, ch2_col, ttl_col, mode=''):
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

    """ abet_trial_definition - Defines a trial structure for the components of the ABET II unprocessed data.
        This method uses the Item names of Condition Events that represent the normal start and end of a trial epoch.
        This method was expanded in PhotometryBatch to allow for multiple start and end groups.
        Arguments:
        start_event_group = the name of an ABET II Condition Event that defines the start of a trial
        end_event_group = the name of an ABET II Condition Event that defines the end of a trial
        Photometry Analyzer currently only supports start group definitions.
        Photometry Batch supports multiple start and end group definitions
        MousePAD will eventually support all definitions as well as sessions with no definition"""

    def abet_trial_definition(self, start_event_group, end_event_group):
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

    def abet_search_event(self, start_event_id='1', start_event_group='', start_event_item_name='',
                          start_event_position=None,
                          filter_event=False, filter_list=None, extra_prior_time=0, extra_follow_time=0,
                          exclusion_list=None):

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
                filter_event_abet = filter_event_bet[~filter_event_abet.isin(exclusion_list)]
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

    """ abet_doric_synchronize - This function searches for TTL timestamps in the ABET II raw data and
        relates it to TTL pulses detected in the photometer. The adjusted sync value is calculated and the 
        doric photometry data time is adjusted to be in reference to the ABET II file."""

    def abet_doric_synchronize(self):
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
        self.doric_pandas['Time'] = (self.doric_pandas['Time'] * ttl_model.coef_[0]) + ttl_model.intercept_
        print(self.doric_pandas['Time'].head(20))

    """doric_process - This function calculates the delta-f value based on the isobestic and active channel data.
        The two channels are first put through a 2nd order low-pass butterworth filter with a user-specified cutoff. 
        Following filtering, the data is fit with least squares regression to a linear function. Finally, the fitted
        data is used to calculate a delta-F value. A pandas dataframe with the time and delta-f values is created.
        Arguments:
        filter_frequency = The cut-off frequency used for the low-pass filter"""

    def doric_filter(self, filter_type='lowpass', filter_name='butterworth', filter_order=4, filter_cutoff=6):
        # Prepare data and apply selected filter (returns time and filtered signals)
        doric_pandas_cut = self.doric_pandas[self.doric_pandas['Time'] >= 0]

        time_data = doric_pandas_cut['Time'].to_numpy()
        f0_data = doric_pandas_cut['Control'].to_numpy()
        f_data = doric_pandas_cut['Active'].to_numpy()

        time_data = time_data.astype(float)
        f0_data = f0_data.astype(float)
        f_data = f_data.astype(float)

        # compute sample frequency (Hz)
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
            except Exception:
                # If savgol fails for any reason, fall back to original data
                filtered_f0 = f0_data.copy()
                filtered_f = f_data.copy()

        else:
            # Unknown filter type: leave data unfiltered
            filtered_f0 = f0_data.copy()
            filtered_f = f_data.copy()

        print(filtered_f)
        return time_data, filtered_f0, filtered_f


    def doric_fit(self, fit_type, filtered_f0, filtered_f, time_data=None):
        # Fit filtered signals, compute delta-F and populate self.doric_pd
        fit_type_lower = str(fit_type).lower() if fit_type is not None else 'linear'

        if time_data is None:
            doric_pandas_cut = self.doric_pandas[self.doric_pandas['Time'] >= 0]
            time_data = doric_pandas_cut['Time'].to_numpy().astype(float)

        # Linear fit: regress active on isobestic
        if fit_type_lower in ('linear', 'lin'):
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
            except Exception:
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
                except Exception:
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

        length_time = self.abet_time_list.iloc[0, 1] - self.abet_time_list.iloc[0, 0]
        measurements_per_interval = length_time * self.sample_frequency
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

        for index, row in self.abet_time_list.iterrows():
            try:
                start_time = row['Start_Time']
                start_index = np.searchsorted(doric_time_array, start_time, side='left')
                if start_index >= len(doric_time_array):
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
            except IndexError:
                print('Trial Start Out of Bounds, Skipping Event')
                continue
            try:
                end_time = row['End_Time']
                end_index = np.searchsorted(doric_time_array, end_time, side='right') - 1
                if end_index < 0 or end_index >= len(doric_time_array):
                    print('Trial End Out of Bounds, Skipping Event')
                    continue
            except IndexError:
                print('Trial End Out of Bounds, Skipping Event')
                continue

            try:
                while doric_time_array[start_index] > self.abet_time_list.loc[index, 'Start_Time'] and start_index > 0:
                    start_index -= 1
            except IndexError:
                continue

            try:
                while doric_time_array[end_index] < self.abet_time_list.loc[index, 'End_Time'] and end_index < len(doric_time_array) - 1:
                    end_index += 1
            except IndexError:
                continue

            while (end_index - start_index + 1) < measurements_per_interval:
                end_index += 1
            while (end_index - start_index + 1) > measurements_per_interval:
                end_index -= 1

            # Slice numpy arrays for this trial
            trial_time = doric_time_array[start_index:end_index]
            trial_deltaf = doric_deltaf_array[start_index:end_index]

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

    def write_data(self, output_data, filename_override=''):
        partial_list = [1, 'SimpleZ', 'simple']
        final_list = [2, 'TimedZ', 'timed']
        partialf_list = [3, 'SimpleF', 'simplef']
        finalf_list = [4, 'TimedF', 'timedf']
        processed_list = [5, 'Full', 'full']

        output_string_list = ['SimpleZ','TimedZ','SimpleF','TimedF','Raw']

        output_string = output_string_list[output_data-1]


        output_folder = self.main_folder_path + self.folder_symbol + 'Output'
        if not (os.path.isdir(output_folder)):
            os.mkdir(output_folder)
        if self.abet_loaded is True and self.anymaze_loaded is False:
            file_path_string = output_folder + self.folder_symbol + output_string + '-' + self.animal_id + ' ' + \
                               self.date + ' ' + self.event_name + '.csv'
        else:
            current_time = datetime.now()
            current_time_string = current_time.strftime('%d-%m-%Y %H-%M-%S')
            file_path_string = output_folder + self.folder_symbol + output_string + '-' + current_time_string + '.csv'

        if filename_override != '':
            file_path_string = filename_override + '-' + output_string + '.csv'

        print(file_path_string)
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

        output_string_list = ['SummaryZ','SummaryF','SummaryP']

        output_string = output_string_list[output_data-1]

        summary_path = output_path + summary_string + 'Summary' + '-' + output_string + '-' +'.xlsx'

        if output_data in z_list:
            session_temp = self.partial_dataframe.transpose()
        if output_data in f_list:
            session_temp = self.partial_deltaf.transpose()
        if output_data in p_list:
            session_temp = self.partial_percent.transpose()

        session_mean = session_temp.mean(axis=0, skipna=True)
        session_mean = [session_string] + session_mean.tolist()
        session_mean = pd.Series(session_mean)
        session_mean = pd.DataFrame(session_mean)
        session_mean = session_mean.transpose()
        session_std = session_temp.std(axis=0, skipna=True)
        session_std = [session_string] + session_std.tolist()
        session_std = pd.Series(session_std)
        session_std = pd.DataFrame(session_std)
        session_std = session_std.transpose()
        session_sem = session_temp.sem(axis=0, skipna=True)
        session_sem = [session_string] + session_sem.tolist()
        session_sem = pd.Series(session_sem)
        session_sem = pd.DataFrame(session_sem)
        session_sem = session_sem.transpose()


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
    abet_file = open(abet_file_path)
    abet_csv_reader = csv.reader(abet_file)
    event_time_colname = ['Evnt_Time', 'Event_Time']
    colnames_found = False
    for row in abet_csv_reader:
        if not colnames_found:
            if len(row) == 0:
                continue
            if row[0] == 'Animal ID':
                animal_id = str(row[1])
                continue
            if row[0] == 'Date/Time':
                date = str(row[1])
                date = date.replace(':', '-')
                date = date.replace('/', '-')
                continue
            if (row[0] == 'Schedule') or (row[0] == 'Schedule Name'):
                schedule = str(row[1])
            if row[0] in event_time_colname:
                colnames_found = True
        else:
            break
    abet_file.close()
    return animal_id, date, schedule

def process_files(file_sheet_path, event_sheet_path, output_options, config):
    file_pair_df = pd.read_csv(file_sheet_path)    
    num_rows = len(file_pair_df)

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

    exclusion_list = [item.strip() for item in config['Filter']['exclusion_list'].split(',')]
    
    results = []
    # Sequential processing instead of multiprocessing
    for _, row in file_pair_df.iterrows():
        photometry_data = PhotometryData()
        photometry_data.load_abet_data(row['abet_path'])
        photometry_data.load_doric_data(row['doric_path'], row['ctrl_col_num'], row['act_col_num'], row['ttl_col_num'], row['mode'])
        photometry_data.abet_doric_synchronize()
        # Use the split filter and fit functions
        time_data, filtered_f0, filtered_f = photometry_data.doric_filter(filter_cutoff)
        photometry_data.doric_fit(fit_type, filtered_f0, filtered_f, time_data)
        
        # Extract values from config object
        photometry_data.abet_trial_definition(trial_start_stage, trial_end_stage)

        
        event_sheet_df = pd.read_csv(event_sheet_path)

        for _, event_row in event_sheet_df.iterrows():
            photometry_data.abet_search_event(
                start_event_id=event_row['event_type'],
                start_event_item_name=event_row['event_name'],
                start_event_group=event_row['event_group'],
                extra_prior_time = event_window_prior,
                extra_follow_time = event_window_follow,
                exclusion_list = exclusion_list
            )
            photometry_data.trial_separator(trial_normalize=center_z,
                                            trial_iti_pad=iti_prior_trial,
                                            center_method=center_method)

            for output in output_options:
                if output <= 7:
                    photometry_data.write_data(output)
                else:
                    photometry_data.write_summary(output)

            max_peak = photometry_data.calculate_max_peak()
            auc = photometry_data.calculate_auc(-event_window_prior, event_window_follow)

            results.append({
                "file": os.path.basename(row['abet_path']),
                "behavior": event_row['event_name'],
                "max_peak": max_peak,
                "auc": auc,
                "plot_data": photometry_data.get_peri_event_data()
            })

    # Create a combined result for each behavior
    combined_results = {}
    for res in results:
        behavior = res['behavior']
        if behavior not in combined_results:
            combined_results[behavior] = {
                "file": "Combined",
                "behavior": behavior,
                "plot_data": pd.DataFrame()
            }
        combined_results[behavior]["plot_data"] = pd.concat([combined_results[behavior]["plot_data"], res["plot_data"]], axis=1)

    for behavior, data in combined_results.items():
        if not data["plot_data"].empty:
            data["max_peak"] = data["plot_data"].max().mean()

            time_series = np.linspace(-event_window_prior, event_window_follow, num=len(data["plot_data"]))
            auc_values = []
            for col in data["plot_data"].columns:
                auc_values.append(np.trapz(y=data["plot_data"][col], x=time_series))
            data["auc"] = np.mean(auc_values)
        else:
            data["max_peak"] = 0
            data["auc"] = 0

        results.append(data)

    return results