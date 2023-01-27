# Imports
import os
import sys
import csv
import configparser
from datetime import datetime
from scipy import signal
import numpy as np
import pandas as pd

pd.set_option('mode.chained_assignment', None)


# Classes
class PhotometryData:
    def __init__(self):

        # Initialize Folder Path Variables
        self.curr_cpu_core_count = os.cpu_count()
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
        for row in abet_csv_reader:
            if not colnames_found:
                if len(row) == 0:
                    continue
                if row[0] == 'Animal ID':
                    self.animal_id = str(row[1])
                    continue
                if row[0] == 'Date/Time':
                    self.date = str(row[1])
                    self.date = self.date.replace(':', '-')
                    self.date = self.date.replace('/', '-')
                    continue
                if row[0] in event_time_colname:
                    colnames_found = True
                    self.time_var_name = row[0]
                    self.event_name_col = row[2]
                    # Columns are 0-time, 1-Event ID, 2-Event name, 3-Item Name, 5-Group ID, 8-Arg-1 Value
                    abet_name_list = [row[0], row[1], row[2], row[3], row[5], row[8]]
                else:
                    continue
            else:
                abet_data_list.append([row[0], row[1], row[2], row[3], row[5], row[8]])
        abet_file.close()
        abet_numpy = np.array(abet_data_list)
        self.abet_pandas = pd.DataFrame(data=abet_numpy, columns=abet_name_list)

    """ load_doric_data - Loads in the doric data to the PhotometryData object. This method uses a
        simple pandas read csv function to import the data. User specified column indexes are used to grab only
        the relevant columns.
         Arguments:
         filepath = The filepath for the doric photometry csv. Generated from GUI path
         ch1_col = The column index for the isobestic channel data
         ch2_col = The column index for the active channel data
         ttl_col = The column index for the TTL data """

    def load_doric_data(self, filepath, ch1_col, ch2_col, ttl_col):
        self.doric_file_path = filepath
        self.doric_loaded = True
        colnames = ['Time', 'Control', 'Active', 'TTL']
        doric_data = pd.read_csv(self.doric_file_path, header=1)
        self.doric_pandas = doric_data.iloc[:, [0, ch1_col, ch2_col, ttl_col]]
        self.doric_pandas.columns = colnames
        self.doric_pandas = self.doric_pandas.astype('float')

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
            if filter_type in condition_event_names:
                filter_event_abet = abet_data.loc[(abet_data[self.event_name_col] == str(filter_type)) & (
                            abet_data['Group_ID'] == str(int(filter_group))), :]
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
            doric_ttl_active = self.doric_pandas.loc[(self.doric_pandas['TTL'] > 1.00), ]
        except KeyError:
            print('No TTL Signal Detected. Ending Analysis.')
            return
        try:
            abet_ttl_active = self.abet_pandas.loc[(self.abet_pandas['Item_Name'] == 'TTL #1'), ]
        except KeyError:
            print('ABET II File missing TTL Pulse Output. Ending Analysis.')
            return

        doric_time = doric_ttl_active.iloc[0, 0]
        doric_time = doric_time.astype(float)
        doric_time = doric_time.item(0)
        abet_time = abet_ttl_active.iloc[0, 0]
        abet_time = float(abet_time)

        self.abet_doric_sync_value = doric_time - abet_time

        self.doric_time = pd.to_numeric(self.doric_pandas['Time'])

        self.doric_pandas['Time'] = self.doric_time - self.abet_doric_sync_value

    """doric_process - This function calculates the delta-f value based on the isobestic and active channel data.
        The two channels are first put through a 2nd order low-pass butterworth filter with a user-specified cutoff. 
        Following filtering, the data is fit with least squares regression to a linear function. Finally, the fitted
        data is used to calculate a delta-F value. A pandas dataframe with the time and delta-f values is created.
        Arguments:
        filter_frequency = The cut-off frequency used for the low-pass filter"""

    def doric_process(self, filter_frequency=6):
        doric_pandas_cut = self.doric_pandas[self.doric_pandas['Time'] >= 0]

        time_data = doric_pandas_cut['Time'].to_numpy()
        f0_data = doric_pandas_cut['Control'].to_numpy()
        f_data = doric_pandas_cut['Active'].to_numpy()

        time_data = time_data.astype(float)
        f0_data = f0_data.astype(float)
        f_data = f_data.astype(float)

        self.sample_frequency = len(time_data) / (time_data[(len(time_data) - 1)] - time_data[0])
        filter_frequency / (self.sample_frequency / 2)
        butter_filter = signal.butter(N=2, Wn=filter_frequency, btype='lowpass', analog=False, output='sos',
                                      fs=self.sample_frequency)
        filtered_f0 = signal.sosfilt(butter_filter, f0_data)
        filtered_f = signal.sosfilt(butter_filter, f_data)

        filtered_poly = np.polyfit(filtered_f0, filtered_f, 1)
        filtered_lobf = np.multiply(filtered_poly[0], filtered_f0) + filtered_poly[1]

        delta_f = (filtered_f - filtered_lobf) / filtered_lobf

        self.doric_pd = pd.DataFrame(time_data)
        self.doric_pd['DeltaF'] = delta_f
        self.doric_pd = self.doric_pd.rename(columns={0: 'Time', 1: 'DeltaF'})

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

    def trial_separator(self, whole_trial_normalize=True, normalize_side='Left', trial_definition=False,
                        trial_iti_pad=0,
                        center_method='mean'):
        if not self.abet_loaded:
            return
        left_selection_list = ['Left', 'Before', 'L', 'l', 'left', 'before', 1]
        right_selection_list = ['Right', 'right', 'R', 'r', 'After', 'after', 2]

        trial_definition_none_list = ['None', 0, '0', 'No', False]
        trial_definition_ind_list = ['Individual', 1, '1', 'Ind', 'Indv']
        trial_definition_overall_list = ['Overall', 2, '2']

        trial_num = 1

        self.abet_time_list = self.abet_event_times

        length_time = self.abet_time_list.iloc[0, 1] - self.abet_time_list.iloc[0, 0]
        measurements_per_interval = length_time * self.sample_frequency
        if trial_definition in trial_definition_none_list:
            for index, row in self.abet_time_list.iterrows():

                try:
                    start_index = self.doric_pd['Time'].sub(self.abet_time_list.loc[index, 'Start_Time']).abs().idxmin()
                except IndexError:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    end_index = self.doric_pd['Time'].sub(self.abet_time_list.loc[index, 'End_Time']).abs().idxmin()
                except IndexError:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

                while self.doric_pd.iloc[start_index, 0] > self.abet_time_list.loc[index, 'Start_Time']:
                    start_index -= 1

                while self.doric_pd.iloc[end_index, 0] < self.abet_time_list.loc[index, 'End_Time']:
                    end_index += 1

                while len(range(start_index, (end_index + 1))) < measurements_per_interval:
                    end_index += 1

                while len(range(start_index, (end_index + 1))) > measurements_per_interval:
                    end_index -= 1

                trial_deltaf = self.doric_pd.iloc[start_index:end_index]
                iti_deltaf = np.empty([1, 1])
                z_mean = np.empty([1, 1])
                z_sd = np.empty([1, 1])

                if not whole_trial_normalize:
                    if normalize_side in left_selection_list:
                        norm_end_time = self.abet_time_list.loc[index, 'Start_Time'] + trial_iti_pad
                        iti_deltaf = trial_deltaf.loc[
                            trial_deltaf['Time'] < norm_end_time, 'DeltaF']
                    elif normalize_side in right_selection_list:
                        norm_start_time = self.abet_time_list.loc[index, 'End_Time'] - trial_iti_pad
                        iti_deltaf = trial_deltaf.loc[
                            trial_deltaf['Time'] > norm_start_time, 'DeltaF']
                    if center_method == 'mean':
                        z_mean = iti_deltaf.mean()
                        z_sd = iti_deltaf.std()
                    elif center_method == 'median':
                        z_mean = iti_deltaf.median()
                        z_dev = np.absolute(np.subtract(iti_deltaf, z_mean))
                        z_sd = z_dev.median()
                else:
                    deltaf_split = trial_deltaf.loc[:, 'DeltaF']
                    if center_method == 'mean':
                        z_mean = deltaf_split.mean()
                        z_sd = deltaf_split.std()
                    elif center_method == 'median':
                        z_mean = deltaf_split.median()
                        z_dev = np.absolute(np.subtract(deltaf_split, z_mean))
                        z_sd = z_dev.median()

                trial_deltaf.loc[:, 'zscore'] = (trial_deltaf.loc[:, 'DeltaF'] - z_mean) / z_sd
                trial_deltaf.loc[:, 'percent_change'] = trial_deltaf.loc[:, 'DeltaF'].map(
                    lambda x: ((x - z_mean) / abs(z_mean)) * 100)

                colname_1 = 'Time Trial ' + str(trial_num)
                colname_2 = 'Z-Score Trial ' + str(trial_num)
                colname_3 = 'Delta-F Trial ' + str(trial_num)
                colname_4 = 'Percent-Change Trial ' + str(trial_num)

                if trial_num == 1:
                    self.final_dataframe = trial_deltaf.loc[:, ('Time', 'zscore')]
                    self.final_dataframe = self.final_dataframe.reset_index(drop=True)
                    self.final_dataframe = self.final_dataframe.rename(
                        columns={'Time': colname_1, 'zscore': colname_2})

                    self.partial_dataframe = trial_deltaf.loc[:, 'zscore']
                    self.partial_dataframe = self.partial_dataframe.to_frame()
                    self.partial_dataframe = self.partial_dataframe.reset_index(drop=True)
                    self.partial_dataframe = self.partial_dataframe.rename(columns={'zscore': colname_2})

                    self.partial_deltaf = trial_deltaf.loc[:, 'DeltaF']
                    self.partial_deltaf = self.partial_deltaf.to_frame()
                    self.partial_deltaf = self.partial_deltaf.reset_index(drop=True)
                    self.partial_deltaf = self.partial_deltaf.rename(columns={'DeltaF': colname_2})

                    self.final_deltaf = trial_deltaf.loc[:, ('Time', 'DeltaF')]
                    self.final_deltaf = self.final_deltaf.reset_index(drop=True)
                    self.final_deltaf = self.final_deltaf.rename(columns={'Time': colname_1, 'DeltaF': colname_2})

                    self.partial_percent = trial_deltaf.loc[:, 'percent_change']
                    self.partial_percent = self.partial_percent.to_frame()
                    self.partial_percent = self.partial_percent.reset_index(drop=True)
                    self.partial_percent = self.partial_percent.rename(columns={'percent_change': colname_2})

                    self.final_percent = trial_deltaf.loc[:, ('Time', 'percent_change')]
                    self.final_percent = self.final_percent.reset_index(drop=True)
                    self.final_percent = self.final_percent.rename(
                        columns={'Time': colname_1, 'percent_change': colname_2})

                    trial_num += 1
                else:
                    trial_deltaf = trial_deltaf.reset_index(drop=True)
                    dataframe_len = len(self.final_dataframe.index)
                    trial_len = len(trial_deltaf.index)
                    if trial_len > dataframe_len:
                        len_diff = trial_len - dataframe_len
                        new_index = list(range(dataframe_len, (dataframe_len + len_diff)))
                        self.final_dataframe = self.final_dataframe.reindex(
                            self.final_dataframe.index.union(new_index))
                        self.partial_dataframe = self.partial_dataframe.reindex(
                            self.partial_dataframe.index.union(new_index))
                        self.partial_deltaf = self.partial_deltaf.reindex(
                            self.partial_deltaf.index.union(new_index))
                        self.final_deltaf = self.final_deltaf.reindex(
                            self.final_deltaf.index.union(new_index))
                        self.partial_percent = self.partial_percent.reindex(
                            self.partial_percent.index.union(new_index))
                        self.final_percent = self.final_percent.reindex(
                            self.final_percent.index.union(new_index))

                    trial_deltaf = trial_deltaf.rename(columns={'Time': colname_1, 'zscore': colname_2,
                                                                'DeltaF': colname_3, 'percent_change': colname_4})

                    self.partial_dataframe = pd.concat([self.partial_dataframe, trial_deltaf[colname_2]],
                                                       axis=1)
                    self.partial_deltaf = pd.concat([self.partial_deltaf, trial_deltaf[colname_3]],
                                                    axis=1)
                    self.final_dataframe = pd.concat(
                        [self.final_dataframe, trial_deltaf[colname_1], trial_deltaf[colname_2]],
                        axis=1)
                    self.final_deltaf = pd.concat([self.final_deltaf, trial_deltaf[colname_1], trial_deltaf[colname_3]],
                                                  axis=1)
                    self.partial_percent = pd.concat([self.partial_percent, trial_deltaf[colname_4]],
                                                     axis=1)
                    self.final_percent = pd.concat(
                        [self.final_percent, trial_deltaf[colname_1], trial_deltaf[colname_4]],
                        axis=1)
                    trial_num += 1

        elif trial_definition in trial_definition_ind_list:
            for index, row in self.abet_time_list.iterrows():
                try:
                    start_index = self.doric_pd.loc[:, 'Time'].sub(
                        self.abet_time_list.loc[index, 'Start_Time']).abs().idxmin()
                except IndexError:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    end_index = self.doric_pd.loc[:, 'Time'].sub(
                        self.abet_time_list.loc[index, 'End_Time']).abs().idxmin()
                except IndexError:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

                try:
                    while self.doric_pd.iloc[start_index, 0] > self.abet_time_list.loc[index, 'Start_Time']:
                        start_index -= 1
                except IndexError:
                    continue

                try:
                    while self.doric_pd.iloc[end_index, 0] < self.abet_time_list.loc[index, 'End_Time']:
                        end_index += 1
                except IndexError:
                    continue

                while len(range(start_index, (end_index + 1))) < measurements_per_interval:
                    end_index += 1

                while len(range(start_index, (end_index + 1))) > measurements_per_interval:
                    end_index -= 1

                trial_deltaf = self.doric_pd.iloc[start_index:end_index]
                if center_z_on_iti == 1:
                    if normalize_side in left_selection_list:
                        trial_start_index_diff = self.trial_definition_times.loc[:, 'Start_Time'].sub(
                            (self.abet_time_list.loc[index, 'Start_Time'] + self.extra_prior))  # .abs().idxmin()
                        trial_start_index_diff[trial_start_index_diff > 0] = np.nan
                        trial_start_index = trial_start_index_diff.abs().idxmin(skipna=True)
                        trial_start_window = self.trial_definition_times.iloc[trial_start_index, 0]
                        trial_iti_window = trial_start_window - float(trial_iti_pad)
                        iti_data = self.doric_pd.loc[(self.doric_pd.loc[:, 'Time'] >= trial_iti_window) & (
                                    self.doric_pd.loc[:, 'Time'] <= trial_start_window), 'DeltaF']
                    else:
                        trial_end_index = self.trial_definition_times.loc[:, 'End_Time'].sub(
                            self.abet_time_list.loc[index, 'End_Time']).abs().idxmin()
                        trial_end_window = self.trial_definition_times.iloc[trial_end_index, 0]
                        trial_iti_window = trial_end_window + trial_iti_pad
                        iti_data = self.doric_pd.loc[(self.doric_pd['Time'] >= trial_end_window) & (
                                    self.doric_pd['Time'] <= trial_iti_window), 'DeltaF']

                    if center_method == 'mean':
                        z_mean = iti_data.mean()
                        z_sd = iti_data.std()
                    elif center_method == 'median':
                        z_mean = iti_data.median()
                        z_dev = np.absolute(np.subtract(iti_data, z_mean))
                        z_sd = z_dev.median()
                else:
                    deltaf_split = trial_deltaf.loc[:, 'DeltaF']
                    if center_method == 'mean':
                        z_mean = deltaf_split.mean()
                        z_sd = deltaf_split.std()
                    elif center_method == 'median':
                        z_mean = deltaf_split.median()
                        z_dev = np.absolute(np.subtract(deltaf_split, z_mean))
                        z_sd = z_dev.median()

                trial_deltaf.loc[:, 'zscore'] = trial_deltaf.loc[:, 'DeltaF'].map(lambda x: ((x - z_mean) / z_sd))
                trial_deltaf.loc[:, 'percent_change'] = trial_deltaf.loc[:, 'DeltaF'].map(
                    lambda x: ((x - z_mean) / abs(z_mean)) * 100)

                colname_1 = 'Time Trial ' + str(trial_num)
                colname_2 = 'Z-Score Trial ' + str(trial_num)
                colname_3 = 'Delta-F Trial ' + str(trial_num)
                colname_4 = 'Percent-Change Trial ' + str(trial_num)

                if trial_num == 1:
                    self.final_dataframe = trial_deltaf.loc[:, ('Time', 'zscore')]
                    self.final_dataframe = self.final_dataframe.reset_index(drop=True)
                    self.final_dataframe = self.final_dataframe.rename(
                        columns={'Time': colname_1, 'zscore': colname_2})

                    self.partial_dataframe = trial_deltaf.loc[:, 'zscore']
                    self.partial_dataframe = self.partial_dataframe.to_frame()
                    self.partial_dataframe = self.partial_dataframe.reset_index(drop=True)
                    self.partial_dataframe = self.partial_dataframe.rename(columns={'zscore': colname_2})

                    self.partial_deltaf = trial_deltaf.loc[:, 'DeltaF']
                    self.partial_deltaf = self.partial_deltaf.to_frame()
                    self.partial_deltaf = self.partial_deltaf.reset_index(drop=True)
                    self.partial_deltaf = self.partial_deltaf.rename(columns={'DeltaF': colname_3})

                    self.final_deltaf = trial_deltaf.loc[:, ('Time', 'DeltaF')]
                    self.final_deltaf = self.final_deltaf.reset_index(drop=True)
                    self.final_deltaf = self.final_deltaf.rename(columns={'Time': colname_1, 'DeltaF': colname_3})

                    self.partial_percent = trial_deltaf.loc[:, 'percent_change']
                    self.partial_percent = self.partial_percent.to_frame()
                    self.partial_percent = self.partial_percent.reset_index(drop=True)
                    self.partial_percent = self.partial_percent.rename(columns={'percent_change': colname_4})

                    self.final_percent = trial_deltaf.loc[:, ('Time', 'percent_change')]
                    self.final_percent = self.final_percent.reset_index(drop=True)
                    self.final_percent = self.final_percent.rename(
                        columns={'Time': colname_1, 'percent_change': colname_4})

                    trial_num += 1
                else:
                    trial_deltaf = trial_deltaf.reset_index(drop=True)
                    dataframe_len = len(self.final_dataframe.index)
                    trial_len = len(trial_deltaf.index)
                    if trial_len > dataframe_len:
                        len_diff = trial_len - dataframe_len
                        new_index = list(range(dataframe_len, (dataframe_len + len_diff)))
                        self.final_dataframe = self.final_dataframe.reindex(
                            self.final_dataframe.index.union(new_index))
                        self.partial_dataframe = self.partial_dataframe.reindex(
                            self.partial_dataframe.index.union(new_index))
                        self.partial_deltaf = self.partial_deltaf.reindex(
                            self.partial_deltaf.index.union(new_index))
                        self.final_deltaf = self.final_deltaf.reindex(
                            self.final_deltaf.index.union(new_index))
                        self.partial_percent = self.partial_percent.reindex(
                            self.partial_percent.index.union(new_index))
                        self.final_percent = self.final_percent.reindex(
                            self.final_percent.index.union(new_index))

                    trial_deltaf = trial_deltaf.rename(columns={'Time': colname_1, 'zscore': colname_2,
                                                                'DeltaF': colname_3, 'percent_change': colname_4})

                    self.partial_dataframe = pd.concat([self.partial_dataframe, trial_deltaf[colname_2]],
                                                       axis=1)
                    self.partial_deltaf = pd.concat([self.partial_deltaf, trial_deltaf[colname_3]],
                                                    axis=1)
                    self.final_dataframe = pd.concat(
                        [self.final_dataframe, trial_deltaf[colname_1], trial_deltaf[colname_2]],
                        axis=1)
                    self.final_deltaf = pd.concat([self.final_deltaf, trial_deltaf[colname_1], trial_deltaf[colname_3]],
                                                  axis=1)
                    self.partial_percent = pd.concat([self.partial_percent, trial_deltaf[colname_4]],
                                                     axis=1)
                    self.final_percent = pd.concat(
                        [self.final_percent, trial_deltaf[colname_1], trial_deltaf[colname_4]],
                        axis=1)
                    trial_num += 1

        elif trial_definition in trial_definition_overall_list:
            mod_trial_times = self.trial_definition_times
            mod_trial_times.iloc[-1, 1] = np.nan
            mod_trial_times.iloc[0, 0] = np.nan
            mod_trial_times['Start_Time'] = mod_trial_times['Start_Time'].shift(-1)
            mod_trial_times = mod_trial_times[:-1]
            for index, row in mod_trial_times.iterrows():
                try:
                    end_index = self.doric_pd.loc[:, 'Time'].sub(
                        mod_trial_times.loc[index, 'Start_Time']).abs().idxmin()
                except IndexError:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    start_index = self.doric_pd.loc[:, 'Time'].sub(
                        mod_trial_times.loc[index, 'End_Time']).abs().idxmin()
                except IndexError:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

                while self.doric_pd.iloc[start_index, 0] > mod_trial_times.loc[index, 'Start_Time']:
                    start_index -= 1

                while self.doric_pd.iloc[end_index, 0] < mod_trial_times.loc[index, 'End_Time']:
                    end_index += 1

                while len(range(start_index, (end_index + 1))) < measurements_per_interval:
                    end_index += 1

                while len(range(start_index, (end_index + 1))) > measurements_per_interval:
                    end_index -= 1

                iti_deltaf = self.doric_pd.iloc[start_index:end_index]
                iti_deltaf = iti_deltaf.loc[:, 'DeltaF']
                full_iti_deltaf = pd.DataFrame()
                if index == 0:
                    full_iti_deltaf = iti_deltaf
                else:
                    full_iti_deltaf = full_iti_deltaf.append(iti_deltaf)

            if center_method == 'mean':
                z_mean = full_iti_deltaf.mean()
                z_sd = full_iti_deltaf.std()
            elif center_method == 'median':
                z_mean = full_iti_deltaf.median()
                z_sd = full_iti_deltaf.std()

            for index, row in self.abet_time_list.iterrows():
                try:
                    start_index = self.doric_pd.loc[:, 'Time'].sub(
                        self.abet_time_list.loc[index, 'Start_Time']).abs().idxmin()
                except IndexError:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    end_index = self.doric_pd.loc[:, 'Time'].sub(
                        self.abet_time_list.loc[index, 'End_Time']).abs().idxmin()
                except IndexError:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

                while self.doric_pd.iloc[start_index, 0] > self.abet_time_list.loc[index, 'Start_Time']:
                    start_index -= 1

                while self.doric_pd.iloc[end_index, 0] < self.abet_time_list.loc[index, 'End_Time']:
                    end_index += 1

                while len(range(start_index, (end_index + 1))) < measurements_per_interval:
                    end_index += 1

                while len(range(start_index, (end_index + 1))) > measurements_per_interval:
                    end_index -= 1

                trial_deltaf = self.doric_pd.iloc[start_index:end_index]
                trial_deltaf.loc[:, 'zscore'] = trial_deltaf.loc[:, 'DeltaF'].map(lambda x: ((x - z_mean) / z_sd))
                trial_deltaf.loc[:, 'percent_change'] = trial_deltaf.loc[:, 'DeltaF'].map(
                    lambda x: ((x - z_mean) / abs(z_mean)) * 100)
                colname_1 = 'Time Trial ' + str(trial_num)
                colname_2 = 'Z-Score Trial ' + str(trial_num)
                colname_3 = 'Delta-F Trial ' + str(trial_num)
                colname_4 = 'Percent-Change Trial ' + str(trial_num)

                if trial_num == 1:
                    self.final_dataframe = trial_deltaf.loc[:, ('Time', 'zscore')]
                    self.final_dataframe = self.final_dataframe.reset_index(drop=True)
                    self.final_dataframe = self.final_dataframe.rename(
                        columns={'Time': colname_1, 'zscore': colname_2})

                    self.partial_dataframe = trial_deltaf.loc[:, 'zscore']
                    self.partial_dataframe = self.partial_dataframe.to_frame()
                    self.partial_dataframe = self.partial_dataframe.reset_index(drop=True)
                    self.partial_dataframe = self.partial_dataframe.rename(columns={'zscore': colname_2})

                    self.partial_deltaf = trial_deltaf.loc[:, 'DeltaF']
                    self.partial_deltaf = self.partial_deltaf.to_frame()
                    self.partial_deltaf = self.partial_deltaf.reset_index(drop=True)
                    self.partial_deltaf = self.partial_deltaf.rename(columns={'DeltaF': colname_2})

                    self.final_deltaf = trial_deltaf.loc[:, ('Time', 'DeltaF')]
                    self.final_deltaf = self.final_deltaf.to_frame()
                    self.final_deltaf = self.final_deltaf.reset_index(drop=True)
                    self.final_deltaf = self.final_deltaf.rename(columns={'Time': colname_1, 'DeltaF': colname_2})

                    self.partial_percent = trial_deltaf.loc[:, 'percent_change']
                    self.partial_percent = self.partial_percent.to_frame()
                    self.partial_percent = self.partial_percent.reset_index(drop=True)
                    self.partial_percent = self.partial_percent.rename(columns={'percent_change': colname_2})

                    self.final_percent = trial_deltaf.loc[:, ('Time', 'percent_change')]
                    self.final_percent = self.final_percent.reset_index(drop=True)
                    self.final_percent = self.final_percent.rename(
                        columns={'Time': colname_1, 'percent_change': colname_2})

                    trial_num += 1
                else:
                    trial_deltaf = trial_deltaf.reset_index(drop=True)
                    dataframe_len = len(self.final_dataframe.index)
                    trial_len = len(trial_deltaf.index)
                    if trial_len > dataframe_len:
                        len_diff = trial_len - dataframe_len
                        new_index = list(range(dataframe_len, (dataframe_len + len_diff)))
                        self.final_dataframe = self.final_dataframe.reindex(
                            self.final_dataframe.index.union(new_index))
                        self.partial_dataframe = self.partial_dataframe.reindex(
                            self.partial_dataframe.index.union(new_index))
                        self.partial_deltaf = self.partial_deltaf.reindex(
                            self.partial_deltaf.index.union(new_index))
                        self.final_deltaf = self.final_deltaf.reindex(
                            self.final_deltaf.index.union(new_index))
                        self.partial_percent = self.partial_percent.reindex(
                            self.partial_percent.index.union(new_index))
                        self.final_percent = self.final_percent.reindex(
                            self.final_percent.index.union(new_index))

                    trial_deltaf = trial_deltaf.rename(columns={'Time': colname_1, 'zscore': colname_2,
                                                                'DeltaF': colname_3, 'percent_change': colname_4})

                    self.partial_dataframe = pd.concat([self.partial_dataframe, trial_deltaf[colname_2]],
                                                       axis=1)
                    self.partial_deltaf = pd.concat([self.partial_deltaf, trial_deltaf[colname_3]],
                                                    axis=1)
                    self.final_dataframe = pd.concat(
                        [self.final_dataframe, trial_deltaf[colname_1], trial_deltaf[colname_2]],
                        axis=1)
                    self.final_deltaf = pd.concat([self.final_deltaf, trial_deltaf[colname_1], trial_deltaf[colname_3]],
                                                  axis=1)
                    self.partial_percent = pd.concat([self.partial_percent, trial_deltaf[colname_4]],
                                                     axis=1)
                    self.final_percent = pd.concat(
                        [self.final_percent, trial_deltaf[colname_1], trial_deltaf[colname_4]],
                        axis=1)
                    trial_num += 1

    def write_data(self, output_data, filename_override=''):
        processed_list = [1, 'Full', 'full']
        partial_list = [4, 'SimpleZ', 'simple']
        final_list = [7, 'TimedZ', 'timed']
        partialf_list = [2, 'SimpleF', 'simplef']
        finalf_list = [5, 'TimedF', 'timedf']
        partialp_list = [3, 'SimpleP', 'simplep']
        finalp_list = [6, 'TimedP', 'timedp']

        output_folder = self.main_folder_path + self.folder_symbol + 'Output'
        if not (os.path.isdir(output_folder)):
            os.mkdir(output_folder)
        if self.abet_loaded is True and self.anymaze_loaded is False:
            file_path_string = output_folder + self.folder_symbol + output_data + '-' + self.animal_id + ' ' + \
                               self.date + ' ' + self.event_name + '.csv'
        else:
            current_time = datetime.now()
            current_time_string = current_time.strftime('%d-%m-%Y %H-%M-%S')
            file_path_string = output_folder + self.folder_symbol + output_data + '-' + current_time_string + '.csv'

        if filename_override != '':
            file_path_string = filename_override + '-' + output_data + '.csv'

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
        elif output_data in partialp_list:
            self.partial_percent.to_csv(file_path_string, index=False)
        elif output_data in finalp_list:
            self.final_percent.to_csv(file_path_string, index=False)


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


# Config Load
curr_dirr = os.getcwd()

config_ini = curr_dirr + '\\config.ini'
config_file = configparser.ConfigParser()
config_file.read(config_ini)

file_list_path = config_file['Filepath']['file_list_path']
file_csv = pd.read_csv(file_list_path)

event_path = config_file['Filepath']['event_list_path']
event_csv = pd.read_csv(event_path)

output_path = config_file['Filepath']['output_path']

start_group_name = config_file['ITI_Window']['trial_start_stage']
start_group_name = start_group_name.split(',')
end_group_name = config_file['ITI_Window']['trial_end_stage']
end_group_name = end_group_name.split(',')

event_prior = float(config_file['Event_Window']['event_prior'])
event_follow = float(config_file['Event_Window']['event_follow'])

iti_prior = float(config_file['ITI_Window']['iti_prior_trial'])

filter_frequency = float(config_file['Photometry_Processing']['filter_frequency'])

run_simplez = int(config_file['Output']['create_simplez'])
run_timedz = int(config_file['Output']['create_timedz'])
run_simplep = int(config_file['Output']['create_simplep'])
run_timedp = int(config_file['Output']['create_timedp'])
run_simplef = int(config_file['Output']['create_simplef'])
run_timedf = int(config_file['Output']['create_timedf'])
run_raw = int(config_file['Output']['create_raw'])

center_z_on_iti = int(config_file['ITI_Window']['center_z_on_iti'])
if center_z_on_iti == 1:
    whole_trial_normalize = True
elif center_z_on_iti == 0:
    whole_trial_normalize = False
else:
    whole_trial_normalize = True
center_method = config_file['ITI_Window']['center_method']

exclusion_list = config_file['Filter']['exclusion_list']
exclusion_list = exclusion_list.split(',')

# Run the Batch

for row_index, row in file_csv.iterrows():
    analyzer = PhotometryData()

    abet_path = row.loc['abet_path']
    doric_path = row.loc['doric_path']

    ctrl_col_index = row.loc['ctrl_col_num']
    active_col_index = row.loc['act_col_num']
    ttl_col_index = row.loc['ttl_col_num']

    animal_id, date, schedule = abet_extract_information(abet_path)

    analyzer.load_abet_data(abet_path)
    analyzer.load_doric_data(doric_path, ctrl_col_index, active_col_index, ttl_col_index)
    analyzer.abet_trial_definition(start_group_name, end_group_name)

    analyzer.abet_doric_synchronize()

    analyzer.doric_process(filter_frequency=filter_frequency)

    raw_processed = False

    legacy_code = False
    if 'num_filter' not in event_csv:
        event_csv['num_filter'] = 1
        legacy_code = True

    for row_index2, row2 in event_csv.iterrows():
        filter_list = []

        if row2['num_filter'] == 0:
            file_string = animal_id + '-' + schedule + '-' + row2.loc['event_name'] + '-' + date
            file_dir = output_path + file_string
        elif row2['num_filter'] >= 1:
            file_string = animal_id + '-' + schedule + '-' + row2.loc['event_name'] + '-'
            for fil in range(0, row2['num_filter']):
                if fil == 0:
                    fil_mod = ''
                else:
                    fil_mod = str(fil + 1)

                if not legacy_code:
                    fil_type_str = 'filter_type' + fil_mod
                    fil_name_str = 'filter_name' + fil_mod
                    fil_group_str = 'filter_group' + fil_mod
                    fil_arg_str = 'filter_arg' + fil_mod
                    fil_eval_str = 'filter_eval' + fil_mod
                    fil_prior_str = 'filter_prior' + fil_mod
                else:
                    fil_type_str = 'filter_type'
                    fil_name_str = 'filter_name'
                    fil_group_str = 'filter_group'
                    fil_arg_str = 'filter_arg'
                    fil_eval_str = 'filter_eval'
                    fil_prior_str = 'filter_prior'

                fil_dict = {'Type': row2[fil_type_str], 'Name': row2[fil_name_str],
                            'Group': str(int(row2[fil_group_str])), 'Arg': row2[fil_arg_str],
                            'Prior': row2[fil_prior_str], 'Eval': row2[fil_eval_str]}
                filter_list.append(fil_dict)

                if pd.isnull(row2[fil_arg_str]):
                    file_string = file_string + row2.loc[fil_name_str] + '-'
                else:
                    file_string = file_string + row2.loc[fil_name_str] + '-' + str(row2.loc[fil_arg_str]) + '-'
            file_string = file_string + date
            file_dir = output_path + file_string
        if os.path.exists(file_dir):
            continue

        null_check = row2.isnull()

        if row2['num_filter'] == 0:
            analyzer.abet_search_event(start_event_id=row2.loc['event_type'],
                                       start_event_item_name=row2.loc['event_name'],
                                       start_event_group=row2.loc['event_group'],
                                       extra_prior_time=event_prior, extra_follow_time=event_follow)

        elif row2['num_filter'] >= 1:
            analyzer.abet_search_event(start_event_id=row2.loc['event_type'],
                                       start_event_item_name=row2.loc['event_name'],
                                       start_event_group=row2.loc['event_group'],
                                       extra_prior_time=event_prior, extra_follow_time=event_follow, filter_event=True,
                                       filter_list=filter_list, exclusion_list=exclusion_list)

        if analyzer.abet_event_times.shape[0] < 1:
            print('no events located')
            continue

        analyzer.trial_separator(whole_trial_normalize=whole_trial_normalize, trial_definition=1,
                                 trial_iti_pad=iti_prior, center_method=center_method)

        if run_simplez == 1:
            analyzer.write_data('SimpleZ', filename_override=file_dir)
        if run_timedz == 1:
            analyzer.write_data('TimedZ', filename_override=file_dir)
        if run_simplep == 1:
            analyzer.write_data('SimpleP', filename_override=file_dir)
        if run_timedp == 1:
            analyzer.write_data('TimedP', filename_override=file_dir)
        if run_simplef == 1:
            analyzer.write_data('SimpleF', filename_override=file_dir)
        if run_timedf == 1:
            analyzer.write_data('TimedF', filename_override=file_dir)
        if run_raw == 1 and not raw_processed:
            file_path_raw = output_path + animal_id + '-' + schedule + '-' + date
            analyzer.write_data('Full', filename_override=file_path_raw)
            raw_processed = True
