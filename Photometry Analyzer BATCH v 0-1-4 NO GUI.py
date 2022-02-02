## Imports ##
import os
import sys
import csv
import configparser
from datetime import datetime
from scipy import signal
import numpy as np
import pandas as pd

pd.set_option('mode.chained_assignment',None)

## Classes ##
class Photometry_Data:
    def __init__(self):

        self.curr_cpu_core_count = os.cpu_count()
        self.curr_dir = os.getcwd()
        if sys.platform == 'linux'or sys.platform == 'darwin':
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

        self.abet_loaded = False
        self.abet_searched = False
        self.anymaze_loaded = False

        self.abet_doric_sync_value = 0

        self.extra_prior = 0
        self.extra_follow = 0

    def load_abet_data(self,filepath):
        self.abet_file_path = filepath
        self.abet_loaded = True
        #abet_file = open(self.abet_file_path)
        abet_file = open(self.abet_file_path)
        #abet_csv_reader = csv.reader(abet_file)
        abet_csv_reader = csv.reader(abet_file)
        abet_data_list = list()
        abet_name_list = list()
        event_time_colname = ['Evnt_Time','Event_Time']
        colnames_found = False
        for row in abet_csv_reader:
            #yield [unicode(cell, 'utf-8') for cell in row]
            if colnames_found == False:
                if len(row) == 0:
                    continue
                if row[0] == 'Animal ID':
                    self.animal_id = str(row[1])
                    continue
                if row[0] == 'Date/Time':
                    self.date = str(row[1])
                    self.date = self.date.replace(':','-')
                    self.date = self.date.replace('/','-')
                    continue
                if row[0] in event_time_colname:
                    colnames_found = True
                    self.time_var_name = row[0]
                    self.event_name_col = row[2]
                    abet_name_list = [row[0],row[1],row[2],row[3],row[5],row[8]]
                else:
                    continue
            else:
                abet_data_list.append([row[0],row[1],row[2],row[3],row[5],row[8]])
        abet_file.close()
        abet_numpy = np.array(abet_data_list)
        self.abet_pandas = pd.DataFrame(data=abet_numpy,columns=abet_name_list)

    def load_doric_data(self,filepath,ch1_col,ch2_col,ttl_col):
        self.doric_file_path = filepath
        self.doric_loaded = True
        doric_file = open(self.doric_file_path)
        doric_csv_reader = csv.reader(doric_file)
        first_row_read = False
        second_row_read = False
        doric_name_list = list()
        doric_list = list()
        for row in doric_csv_reader:
            if first_row_read == False:
                first_row_read = True
                continue
            if second_row_read == False and first_row_read == True:
                doric_name_list = [row[0],row[ch1_col],row[ch2_col],row[ttl_col]]
                second_row_read = True
                continue
            else:
                if row[0] == '':
                    break
                try:
                    if row[ch1_col] == '' or row[ch2_col] == '' or row[ttl_col] == '':
                        continue
                except:
                    print(row)
                
                doric_list.append([row[0],row[ch1_col],row[ch2_col],row[ttl_col]])
        doric_file.close()
        doric_numpy = np.array(doric_list)
        self.doric_pandas = pd.DataFrame(data=doric_numpy,columns=doric_name_list)
        self.doric_pandas.columns = ['Time','Control','Active','TTL']
        self.doric_pandas = self.doric_pandas.astype('float')
                
                

    def load_anymaze_data(self,filepath):
        self.anymaze_file_path = filepath
        self.anymaze_loaded = True
        anymaze_file = open(self.anymaze_file_path)
        anymaze_csv = csv.reader(anymaze_file)
        colname_found = False
        anymaze_data = list()
        anymaze_colnames = list()
        for row in anymaze_csv:
            if colname_found == False:
                anymaze_colnames = row
                colname_found = True
            else:
                anymaze_data.append(row)
        anymaze_file.close()
        anymaze_numpy = np.array(anymaze_data)
        self.anymaze_pandas = pd.DataFrame(data=anymaze_numpy,columns=anymaze_colnames)
        self.anymaze_pandas = self.anymaze_pandas.replace(r'^\s*$', np.nan, regex=True)
        self.anymaze_pandas = self.anymaze_pandas.astype('float')

    def abet_trial_definition(self,start_event_group,end_event_group,extra_prior_time=0,extra_follow_time=0):
        if self.abet_loaded == False:
            return None

        if isinstance(start_event_group,list) and isinstance(end_event_group,list):
            #filtered_abet = self.abet_pandas.loc[((self.abet_pandas['Item_Name'].isin(start_event_group)) | (self.abet_pandas['Item_Name'].isin(end_event_group))) & (self.abet_pandas['Evnt_ID'] == '1')]
            event_group_list = start_event_group + end_event_group
            filtered_abet = self.abet_pandas[self.abet_pandas.Item_Name.isin(event_group_list)]
        elif isinstance(start_event_group,list) and not(isinstance(end_event_group,list)):
            filtered_abet = self.abet_pandas.loc[((self.abet_pandas['Item_Name'].isin(start_event_group)) | (self.abet_pandas['Item_Name'] == str(end_event_group))) & (self.abet_pandas['Evnt_ID'] == '1')]
            end_event_group = [end_event_group]
        elif isinstance(end_event_group,list) and not(isinstance(start_event_group,list)):
             filtered_abet = self.abet_pandas.loc[((self.abet_pandas['Item_Name'] == str(start_event_group)) | (self.abet_pandas['Item_Name'].isin(end_event_group))) & (self.abet_pandas['Evnt_ID'] == '1')]
             start_event_group = [start_event_group]
        else:
            filtered_abet = self.abet_pandas.loc[((self.abet_pandas['Item_Name'] == str(start_event_group)) | (self.abet_pandas['Item_Name'] == str(end_event_group))) & (self.abet_pandas['Evnt_ID'] == '1')]
            start_event_group = [start_event_group]
            end_event_group = [end_event_group]
            
        filtered_abet = filtered_abet.reset_index(drop=True)
        if filtered_abet.iloc[0,3] not in start_event_group:
            filtered_abet = filtered_abet.drop([0]) # OCCURS IF FIRST INSTANCE IS THE END OF A TRIAL (COMMON WITH ITI)
        trial_times = filtered_abet.loc[:,self.time_var_name]
        trial_times = trial_times.reset_index(drop=True)
        start_times = trial_times.iloc[::2]
        start_times = start_times.reset_index(drop=True)
        start_times = pd.to_numeric(start_times,errors='coerce')
        end_times = trial_times.iloc[1::2]
        end_times = end_times.reset_index(drop=True)
        end_times = pd.to_numeric(end_times,errors='coerce')
        self.trial_definition_times = pd.concat([start_times,end_times],axis=1)
        self.trial_definition_times.columns = ['Start_Time','End_Time']
        self.trial_definition_times = self.trial_definition_times.reset_index(drop=True)
        

    def abet_search_event(self,start_event_id='1',start_event_group='',start_event_item_name='',start_event_position=[''],
                          filter_event_id='1',filter_event_group='',filter_event_item_name='',filter_event_position=[''],
                          filter_event_arg='',filter_event=False,filter_before=1,centered_event=False,
                          extra_prior_time=0,extra_follow_time=0, exclusion_list = []):
        touch_event_names = ['Touch Up Event','Touch Down Event','Whisker - Clear Image by Position']
        condition_event_names = ['Condition Event']
        variable_event_names = ['Variable Event']
        
        if start_event_id in touch_event_names:
            filtered_abet = self.abet_pandas.loc[(self.abet_pandas[self.event_name_col] == str(start_event_id)) & (self.abet_pandas['Group_ID'] == str(start_event_group)) & 
                                                 (self.abet_pandas['Item_Name'] == str(start_event_item_name)) & (self.abet_pandas['Arg1_Value'] == str(start_event_position)),:] 
    
        else:
            filtered_abet = self.abet_pandas.loc[(self.abet_pandas[self.event_name_col] == str(start_event_id)) & (self.abet_pandas['Group_ID'] == str(start_event_group)) &
                                                (self.abet_pandas['Item_Name'] == str(start_event_item_name)),:]
         
        if filter_event == True:
            if filter_event_id in condition_event_names:
                filter_event_abet = self.abet_pandas.loc[(self.abet_pandas[self.event_name_col] == str(filter_event_id)) & (self.abet_pandas['Group_ID'] == str(int(filter_event_group))),:]
            elif filter_event_id in variable_event_names:
                filter_event_abet = self.abet_pandas.loc[(self.abet_pandas[self.event_name_col] == str(filter_event_id)) & (self.abet_pandas['Item_Name'] == str(filter_event_item_name)),:]


        self.abet_event_times = filtered_abet.loc[:,self.time_var_name]
        self.abet_event_times = self.abet_event_times.reset_index(drop=True)
        self.abet_event_times = pd.to_numeric(self.abet_event_times, errors='coerce')
        
        if filter_event == True:
            filter_event_abet = filter_event_abet[~filter_event_abet.isin(exclusion_list)]
            
            
            filter_before = int(filter_before)
            if filter_event_id in condition_event_names:
                for index, value in self.abet_event_times.items():
                    sub_values = filter_event_abet.loc[:,self.time_var_name]
                    sub_values = sub_values.astype(dtype='float64')
                    sub_values = sub_values.sub(float(value))
                    filter_before = int(filter_before)
                    if filter_before == 1:
                        sub_values[sub_values > 0] = np.nan
                    elif filter_before == 0:
                        sub_values[sub_values < 0] = np.nan
                    sub_index = sub_values.abs().idxmin(skipna=True)
                    sub_null = sub_values.isnull().sum()
                    if sub_null >= sub_values.size:
                        continue
                    
                    filter_value = filter_event_abet.loc[sub_index,'Item_Name']
                    if filter_value != filter_event_item_name:
                        self.abet_event_times[index] = np.nan
                
                self.abet_event_times = self.abet_event_times.dropna()
                self.abet_event_times = self.abet_event_times.reset_index(drop=True)
            elif filter_event_id in variable_event_names:
                for index, value in self.abet_event_times.items():
                    sub_values = filter_event_abet.loc[:,self.time_var_name]
                    sub_values = sub_values.astype(dtype='float64')
                    sub_values = sub_values.sub(float(value))
                    sub_null = sub_values.isnull().sum()
                    filter_before = int(filter_before)
                    if sub_null >= sub_values.size:
                        continue
                    if filter_before == 1:
                        sub_values[sub_values > 0] = np.nan
                    elif filter_before == 0:
                        sub_values[sub_values < 0] = np.nan
                    sub_index = sub_values.abs().idxmin(skipna=True)
                    
                    filter_value = filter_event_abet.loc[sub_index,'Arg1_Value']
                    if float(filter_value) != float(filter_event_arg):
                        self.abet_event_times[index] = np.nan
                        
                self.abet_event_times = self.abet_event_times.dropna()
                self.abet_event_times = self.abet_event_times.reset_index(drop=True)
                
        
        abet_start_times = self.abet_event_times - extra_prior_time
        abet_end_times = self.abet_event_times + extra_follow_time
        self.abet_event_times = pd.concat([abet_start_times,abet_end_times],axis=1)
        self.abet_event_times.columns = ['Start_Time','End_Time']
        self.event_name = start_event_item_name
        self.extra_follow = extra_follow_time
        self.extra_prior = extra_prior_time
        

    def anymaze_search_event_OR(self,event1_name,event1_operation,event1_value=0,event2_name='None',event2_operation='None',event2_value=0,event3_name='None',event3_operation='None',event3_value=0,
                                event_tolerance = 1.00,extra_prior_time=0,extra_follow_time=0,event_definition='Event Start'):
        def operation_search(event,operation,value=0):
            if operation == 'Active':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] == 1,:]
            elif operation == 'Inactive':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] == 0,:]
            elif operation == 'Less Than':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] < value,:]
            elif operation == 'Less Than or Equal':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] <= value,:]
            elif operation == 'Equal':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] == value,:]
            elif operation == 'Greater Than or Equal':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] >= value,:]
            elif operation == 'Greater Than':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] > value,:]
            elif operation == 'Not Equal':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] != value,:]

            search_index = search_data.index
            search_index = search_index.tolist()
            return search_index

        
        anymaze_boolean_list = ['Active','Inactive']
        anymaze_operation_list = ['Less Than', 'Less Than or Equal', 'Equal', 'Greater Than or Equal', 'Greater Than','Not Equal']

        if event1_name == 'None' and event2_name == 'None' and event3_name == 'None':
            return
        
        elif event2_name == 'None' and event3_name == 'None' and event1_name != 'None':
            event_index = operation_search(event1_name,event1_operation,event1_value)
        elif event3_name == 'None' and event2_name != 'None' and event1_name != 'None':
            event1_index = operation_search(event1_name,event1_operation,event1_value)
            event2_index = operation_search(event2_name,event2_operation,event2_value)
            event_index_hold = event1_index + event2_index
            event_index = list()
            for item in event_index_hold:
                if event_index_hold.count(item) >= 2:
                    if item not in event_index:
                        event_index.append(item)
            
        else:
            event1_index = operation_search(event1_name,event1_operation,event1_value)
            event2_index = operation_search(event2_name,event2_operation,event2_value)
            event3_index = operation_search(event2_name,event2_operation,event2_value)
            event_index_hold = event1_index + event2_index + event3_index
            event_index = list()
            for item in event_index_hold:
                if event_index_hold.count(item) >= 3:
                    if item not in event_index:
                        event_index.append(item)
            

        #event_index = event_index.sort()


        search_times = self.anymaze_pandas.loc[event_index,'Time']
        search_times = search_times.reset_index(drop=True)

        event_start_times = list()
        event_end_times = list()

        event_start_time = self.anymaze_pandas.loc[0,'Time']

        current_time = self.anymaze_pandas.loc[0,'Time']

        previous_time = self.anymaze_pandas.loc[0,'Time']

        event_end_time = 0

        for index,value in search_times.items():
            previous_time = current_time
            current_time = value
            if event_start_time == self.anymaze_pandas.loc[0,'Time']:
                event_start_time = current_time
                event_start_times.append(event_start_time)
                continue
            if (current_time - previous_time) >= event_tolerance:
                event_end_time = previous_time
                event_start_time = current_time
                event_start_times.append(event_start_time)
                event_end_times.append(event_end_time)
                continue
            if index >= (search_times.size - 1):
                event_end_time = current_time
                event_end_times.append(event_end_time)
                break

        final_start_times = list()
        final_end_times = list()
        if event_definition == "Event Start":
            for time in event_start_times:
                final_start_time = time - extra_prior_time
                if final_start_time <= 0:
                    continue
                final_start_times.append(final_start_time)
                final_end_time = time + extra_follow_time
                final_end_times.append(final_end_time)

        elif event_definition == "Event Center":
            center_times = list()
            for index in range(0,(len(event_start_times) -1)):
                center_time = event_end_times[index] - event_start_times[index]
                center_times.append(center_time)
            for time in center_times:
                final_start_time = time - extra_prior_time
                if final_start_time <= 0:
                    continue
                final_start_times.append(final_start_time)
                final_end_time = time + extra_follow_time
                final_end_times.append(final_end_time)

        elif event_definition == "Event End":
            for time in event_end_times:
                final_start_time = time - extra_prior_time
                if final_start_time <= 0:
                    continue
                final_start_times.append(final_start_time)
                final_end_time = time + extra_follow_time
                final_end_times.append(final_end_time)
        self.anymaze_event_times = pd.DataFrame(final_start_times)
        self.anymaze_event_times['End_Time'] = final_end_times
        self.anymaze_event_times.columns = ['Start_Time','End_Time']
        self.abet_event_times = self.anymaze_event_times
                
                
            
        
        


    def abet_doric_synchronize(self):
        if self.abet_loaded == False:
            return None
        if self.doric_loaded == False:
            return None
        try:
            doric_ttl_active = self.doric_pandas.loc[(self.doric_pandas['TTL'] > 1.00),]
        except:
            print('No TTL Signal Detected. Ending Analysis')
            return
        try:
            abet_ttl_active = self.abet_pandas.loc[(self.abet_pandas['Item_Name'] == 'TTL #1'),]
        except:
            print('ABET II File missing TTL Pulse Output')
            return

        doric_time = doric_ttl_active.iloc[0,0]
        doric_time = doric_time.astype(float)
        doric_time = doric_time.item()
        abet_time = abet_ttl_active.iloc[0,0]
        abet_time = float(abet_time)

        self.abet_doric_sync_value = doric_time - abet_time
        
        self.doric_time = pd.to_numeric(self.doric_pandas['Time'])

        self.doric_pandas['Time'] = self.doric_time - self.abet_doric_sync_value
                                        

    def anymaze_doric_synchronize_OR(self):
        if self.anymaze_loaded == False:
            return None
        if self.doric_loaded == False:
            return None

        try:
            doric_ttl_active = self.doric_pandas.loc[(self.doric_pandas['TTL'] > 1.00),]
        except:
            print('No TTL Signal Detected. Ending Analysis')
            return

        try:
            anymaze_ttl_active = self.anymaze_pandas.loc[(self.anymaze_pandas['TTL Pulse active'] > 0),]
        except:
            print('Anymaze File missing TTL Pulse Output')

        doric_time = doric_ttl_active.iloc[0,0]
        doric_time = doric_time.astype(float)
        doric_time = np.asscalar(doric_time)
        anymaze_time = anymaze_ttl_active.iloc[0,0]
        anymaze_time = float(anymaze_time)

        self.anymaze_doric_sync_value = doric_time - anymaze_time
        
        self.doric_time = pd.to_numeric(self.doric_pandas['Time'])

        self.doric_pandas['Time'] = self.doric_time - self.anymaze_doric_sync_value

    def doric_process(self,filter_frequency=6):
        time_data = self.doric_pandas['Time'].to_numpy()
        f0_data = self.doric_pandas['Control'].to_numpy()
        f_data = self.doric_pandas['Active'].to_numpy()

        time_data = time_data.astype(float)
        f0_data = f0_data.astype(float)
        f_data = f_data.astype(float)

        self.sample_frequency = len(time_data) / (time_data[(len(time_data) - 1)] - time_data[0])
        filter_frequency_normalized = filter_frequency / (self.sample_frequency/2)
        butter_filter = signal.butter(N=2,Wn=filter_frequency,
                                           btype='lowpass',analog=False,
                                           output='sos',fs=self.sample_frequency)
        filtered_f0 = signal.sosfilt(butter_filter,f0_data)
        filtered_f = signal.sosfilt(butter_filter,f_data)
        
        filtered_poly = np.polyfit(filtered_f0,filtered_f,1)
        filtered_lobf = np.multiply(filtered_poly[0],filtered_f0) + filtered_poly[1]
        
        delta_f = (filtered_f - filtered_lobf) / filtered_lobf

        self.doric_pd = pd.DataFrame(time_data)
        self.doric_pd['DeltaF'] = delta_f
        self.doric_pd = self.doric_pd.rename(columns={0:'Time',1:'DeltaF'})

    def trial_separator(self,normalize=True,center_z_on_iti=1,normalize_side = 'Left',trial_definition = False,trial_iti_pad=0,event_location='None', center_method = 'mean'):
        if self.abet_loaded == False and self.anymaze_loaded == False:
            return
        left_selection_list = ['Left','Before','L','l','left','before',1]
        right_selection_list = ['Right','right','R','r','After','after',2]
        
        trial_definition_none_list = ['None',0,'0','No',False]
        trial_definition_ind_list = ['Individual',1,'1','Ind','Indv']
        trial_definition_overall_list = ['Overall',2,'2']

        trial_num = 1
        
        self.abet_time_list = self.abet_event_times
        
        
        length_time = self.abet_time_list.iloc[0,1]- self.abet_time_list.iloc[0,0]
        measurements_per_interval = length_time * self.sample_frequency
        if trial_definition in trial_definition_none_list:
            for index, row in self.abet_time_list.iterrows():

                try:
                    start_index = self.doric_pd['Time'].sub(self.abet_time_list.loc[index,'Start_Time']).abs().idxmin()
                except:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    end_index = self.doric_pd['Time'].sub(self.abet_time_list.loc[index,'End_Time']).abs().idxmin()
                except:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

                while self.doric_pd.iloc[start_index, 0] > self.abet_time_list.loc[index,'Start_Time']:
                    start_index -= 1

                while self.doric_pd.iloc[end_index, 0] < self.abet_time_list.loc[index,'End_Time']:
                    end_index += 1

                while len(range(start_index,(end_index + 1))) < measurements_per_interval:
                    end_index += 1
                    
                while len(range(start_index,(end_index + 1))) > measurements_per_interval:
                    end_index -= 1

                trial_deltaf = self.doric_pd.iloc[start_index:end_index]
                if center_z_on_iti == 1:
                    if normalize_side in left_selection_list:
                        norm_start_time = self.abet_time_list.loc[index,'Start_Time']
                        norm_end_time = self.abet_time_list.loc[index,'Start_Time'] + trial_iti_pad
                        iti_deltaf = trial_deltaf.loc[
                            trial_deltaf['Time'] < norm_end_time, 'DeltaF']
                    elif normalize_side in right_selection_list:
                        norm_start_time = self.abet_time_list.loc[index,'End_Time'] - trial_iti_pad
                        norm_end_time = self.abet_time_list.loc[index,'End_Time']
                        iti_deltaf = trial_deltaf.loc[
                            trial_deltaf['Time'] > norm_start_time, 'DeltaF']
                    if center_method == 'mean':
                        z_mean = iti_deltaf.mean()
                        z_sd = iti_deltaf.std()
                    elif center_method == 'median':
                        z_mean = iti_deltaf.median()
                        z_dev = np.absolute(np.subtract(iti_deltaf,z_mean))
                        z_sd = z_dev.median()
                else:
                    deltaf_split = trial_deltaf.loc[:, 'DeltaF']
                    if center_method == 'mean':
                        z_mean = deltaf_split.mean()
                        z_sd = deltaf_split.std()
                    elif center_method == 'median':
                        z_mean = deltaf_split.median()
                        z_dev = np.absolute(np.subtract(deltaf_split,z_mean))
                        z_sd = z_dev.median()

                trial_deltaf.loc[:,'zscore'] = (trial_deltaf.loc[:,'DeltaF'] - z_mean) / z_sd
                trial_deltaf.loc[:,'percent_change'] = trial_deltaf.loc[:,'DeltaF'].map(lambda x: ((x - z_mean) / abs(z_mean)) * 100)

                colname_1 = 'Time Trial ' + str(trial_num)
                colname_2 = 'Z-Score Trial ' + str(trial_num)

                if trial_num == 1:
                    self.final_dataframe = trial_deltaf.loc[:, ('Time', 'zscore')]
                    self.final_dataframe = self.final_dataframe.reset_index(drop=True)
                    self.final_dataframe = self.final_dataframe.rename(
                        columns={'Time': colname_1, 'zscore': colname_2})

                    self.partial_dataframe = trial_deltaf.loc[:, 'zscore']
                    self.partial_dataframe = self.partial_dataframe.to_frame()
                    self.partial_dataframe = self.partial_dataframe.reset_index(drop=True)
                    self.partial_dataframe = self.partial_dataframe.rename(columns={'zscore': colname_2})
                    
                    self.partial_deltaf = trial_deltaf.loc[:,'DeltaF']
                    self.partial_deltaf = self.partial_deltaf.to_frame()
                    self.partial_deltaf = self.partial_deltaf.reset_index(drop=True)
                    self.partial_deltaf = self.partial_deltaf.rename(columns={'DeltaF': colname_2})
                    
                    self.final_deltaf = trial_deltaf.loc[:, ('Time', 'DeltaF')]
                    self.final_deltaf = self.final_deltaf.reset_index(drop=True)
                    self.final_deltaf = self.final_deltaf.rename(columns={'Time': colname_1, 'DeltaF': colname_2})
                    
                    self.partial_percent = trial_deltaf.loc[:,'percent_change']
                    self.partial_percent = self.partial_percent.to_frame()
                    self.partial_percent = self.partial_percent.reset_index(drop=True)
                    self.partial_percent = self.partial_percent.rename(columns={'percent_change': colname_2})
                    
                    self.final_percent = trial_deltaf.loc[:, ('Time', 'percent_change')]
                    self.final_percent = self.final_percent.reset_index(drop=True)
                    self.final_percent = self.final_percent.rename(columns={'Time': colname_1, 'percent_change': colname_2})
                    
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

                    self.partial_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    self.final_dataframe.loc[:,colname_1] = trial_deltaf.loc[:,'Time']
                    self.final_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    self.partial_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    self.partial_deltaf.loc[:,colname_2] = trial_deltaf.loc[:,'DeltaF']
                    self.final_dataframe.loc[:,colname_1] = trial_deltaf.loc[:,'Time']
                    self.final_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    self.final_deltaf.loc[:,colname_1] = trial_deltaf.loc[:,'Time']
                    self.final_deltaf.loc[:,colname_2] = trial_deltaf.loc[:,'DeltaF']
                    self.partial_percent.loc[:,colname_2] = trial_deltaf.loc[:,'percent_change']
                    self.final_percent.loc[:,colname_1] = trial_deltaf.loc[:,'Time']
                    self.final_percent.loc[:,colname_2] = trial_deltaf.loc[:,'percent_change']
                    trial_num += 1
                    
        elif trial_definition in trial_definition_ind_list:
            for index, row in self.abet_time_list.iterrows():
                try:
                    start_index = self.doric_pd.loc[:,'Time'].sub(self.abet_time_list.loc[index,'Start_Time']).abs().idxmin()
                except:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    end_index = self.doric_pd.loc[:,'Time'].sub(self.abet_time_list.loc[index,'End_Time']).abs().idxmin()
                except:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

                while self.doric_pd.iloc[start_index, 0] > self.abet_time_list.loc[index,'Start_Time']:
                    start_index -= 1

                while self.doric_pd.iloc[end_index, 0] < self.abet_time_list.loc[index,'End_Time']:
                    end_index += 1

                while len(range(start_index,(end_index + 1))) < measurements_per_interval:
                    end_index += 1
                    
                while len(range(start_index,(end_index + 1))) > measurements_per_interval:
                    end_index -= 1

                trial_deltaf = self.doric_pd.iloc[start_index:end_index]
                if center_z_on_iti == 1:
                    if normalize_side in left_selection_list:
                        trial_start_index_diff = self.trial_definition_times.loc[:,'Start_Time'].sub((self.abet_time_list.loc[index,'Start_Time'] + self.extra_prior))#.abs().idxmin()
                        trial_start_index_diff[trial_start_index_diff > 0] = np.nan
                        trial_start_index = trial_start_index_diff.abs().idxmin(skipna=True)
                        trial_start_window = self.trial_definition_times.iloc[trial_start_index,0]
                        trial_iti_window = trial_start_window - float(trial_iti_pad)
                        iti_data = self.doric_pd.loc[(self.doric_pd.loc[:,'Time'] >= trial_iti_window) & (self.doric_pd.loc[:,'Time'] <= trial_start_window),'DeltaF']
                    elif normalize_side in right_selection_list:
                        trial_end_index = self.trial_definition_times.loc[:,'End_Time'].sub(self.abet_time_list.loc[index,'End_Time']).abs().idxmin()
                        trial_end_window = self.trial_definition_times.iloc[trial_end_index,0]
                        trial_iti_window = trial_end_window + trial_iti_pad
                        iti_data = self.doric_pd.loc[(self.doric_pd['Time'] >= trial_end_window) & (self.doric_pd['Time'] <= trial_iti_window),'DeltaF']
                    
                    if center_method == 'mean':
                        z_mean = iti_data.mean()
                        z_sd = iti_data.std()
                    elif center_method == 'median':
                        z_mean = iti_data.median()
                        z_dev = np.absolute(np.subtract(iti_data,z_mean))
                        z_sd = z_dev.median()
                else:
                    deltaf_split = trial_deltaf.loc[:, 'DeltaF']
                    if center_method == 'mean':
                        z_mean = deltaf_split.mean()
                        z_sd = deltaf_split.std()
                    elif center_method == 'median':
                        z_mean = deltaf_split.median()
                        z_dev = np.absolute(np.subtract(deltaf_split,z_mean))
                        z_sd = z_dev.median()
                    
                    
                trial_deltaf.loc[:,'zscore'] = trial_deltaf.loc[:,'DeltaF'].map(lambda x: ((x - z_mean)/z_sd))
                trial_deltaf.loc[:,'percent_change'] = trial_deltaf.loc[:,'DeltaF'].map(lambda x: ((x - z_mean) / abs(z_mean)) * 100)
                #trial_deltaf.loc[:,'zscore'] = (trial_deltaf.loc[:,'DeltaF'] - z_mean) / z_sd

                colname_1 = 'Time Trial ' + str(trial_num)
                colname_2 = 'Z-Score Trial ' + str(trial_num)

                if trial_num == 1:
                    self.final_dataframe = trial_deltaf.loc[:, ('Time', 'zscore')]
                    self.final_dataframe = self.final_dataframe.reset_index(drop=True)
                    self.final_dataframe = self.final_dataframe.rename(
                        columns={'Time': colname_1, 'zscore': colname_2})

                    self.partial_dataframe = trial_deltaf.loc[:, 'zscore']
                    self.partial_dataframe = self.partial_dataframe.to_frame()
                    self.partial_dataframe = self.partial_dataframe.reset_index(drop=True)
                    self.partial_dataframe = self.partial_dataframe.rename(columns={'zscore': colname_2})
                    
                    self.partial_deltaf = trial_deltaf.loc[:,'DeltaF']
                    self.partial_deltaf = self.partial_deltaf.to_frame()
                    self.partial_deltaf = self.partial_deltaf.reset_index(drop=True)
                    self.partial_deltaf = self.partial_deltaf.rename(columns={'DeltaF': colname_2})
                    
                    self.final_deltaf = trial_deltaf.loc[:, ('Time', 'DeltaF')]
                    self.final_deltaf = self.final_deltaf.reset_index(drop=True)
                    self.final_deltaf = self.final_deltaf.rename(columns={'Time': colname_1, 'DeltaF': colname_2})
                    
                    self.partial_percent = trial_deltaf.loc[:,'percent_change']
                    self.partial_percent = self.partial_percent.to_frame()
                    self.partial_percent = self.partial_percent.reset_index(drop=True)
                    self.partial_percent = self.partial_percent.rename(columns={'percent_change': colname_2})
                    
                    self.final_percent = trial_deltaf.loc[:, ('Time', 'percent_change')]
                    self.final_percent = self.final_percent.reset_index(drop=True)
                    self.final_percent = self.final_percent.rename(columns={'Time': colname_1, 'percent_change': colname_2})
                    
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

                    self.partial_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    self.final_dataframe.loc[:,colname_1] = trial_deltaf.loc[:,'Time']
                    self.final_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    self.partial_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    self.partial_deltaf.loc[:,colname_2] = trial_deltaf.loc[:,'DeltaF']
                    self.final_dataframe.loc[:,colname_1] = trial_deltaf.loc[:,'Time']
                    self.final_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    self.final_deltaf.loc[:,colname_1] = trial_deltaf.loc[:,'Time']
                    self.final_deltaf.loc[:,colname_2] = trial_deltaf.loc[:,'DeltaF']
                    self.partial_percent.loc[:,colname_2] = trial_deltaf.loc[:,'percent_change']
                    self.final_percent.loc[:,colname_1] = trial_deltaf.loc[:,'Time']
                    self.final_percent.loc[:,colname_2] = trial_deltaf.loc[:,'percent_change']
                    trial_num += 1
                    
        elif trial_definition in trial_definition_overall_list:
            mod_trial_times = self.trial_definition_times
            mod_trial_times.iloc[-1,1] = np.nan
            mod_trial_times.iloc[0,0] = np.nan
            mod_trial_times['Start_Time'] = mod_trial_times['Start_Time'].shift(-1)
            mod_trial_times = mod_trial_times[:-1]
            for index, row in mod_trial_times.iterrows():
                try:
                    end_index = self.doric_pd.loc[:,'Time'].sub(mod_trial_times.loc[index,'Start_Time']).abs().idxmin()
                except:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    start_index = self.doric_pd.loc[:,'Time'].sub(mod_trial_times.loc[index,'End_Time']).abs().idxmin()
                except:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

                while self.doric_pd.iloc[start_index, 0] > mod_trial_times.loc[index,'Start_Time']:
                    start_index -= 1

                while self.doric_pd.iloc[end_index, 0] < mod_trial_times.loc[index,'End_Time']:
                    end_index += 1

                while len(range(start_index,(end_index + 1))) < measurements_per_interval:
                    end_index += 1
                    
                while len(range(start_index,(end_index + 1))) > measurements_per_interval:
                    end_index -= 1
                    
                iti_deltaf = self.doric_pd.iloc[start_index:end_index]
                iti_deltaf = iti_deltaf.loc[:,'DeltaF']
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
                    start_index = self.doric_pd.loc[:,'Time'].sub(self.abet_time_list.loc[index,'Start_Time']).abs().idxmin()
                except:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    end_index = self.doric_pd.loc[:,'Time'].sub(self.abet_time_list.loc[index,'End_Time']).abs().idxmin()
                except:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

                while self.doric_pd.iloc[start_index, 0] > self.abet_time_list.loc[index,'Start_Time']:
                    start_index -= 1

                while self.doric_pd.iloc[end_index, 0] < self.abet_time_list.loc[index,'End_Time']:
                    end_index += 1

                while len(range(start_index,(end_index + 1))) < measurements_per_interval:
                    end_index += 1
                    
                while len(range(start_index,(end_index + 1))) > measurements_per_interval:
                    end_index -= 1
                
                trial_deltaf = self.doric_pd.iloc[start_index:end_index]
                trial_deltaf.loc[:,'zscore'] = trial_deltaf.loc[:,'DeltaF'].map(lambda x: ((x - z_mean)/z_sd))
                trial_deltaf.loc[:,'percent_change'] = trial_deltaf.loc[:,'DeltaF'].map(lambda x: ((x - z_mean) / abs(z_mean)) * 100)
                colname_1 = 'Time Trial ' + str(trial_num)
                colname_2 = 'Z-Score Trial ' + str(trial_num)

                if trial_num == 1:
                    self.final_dataframe = trial_deltaf.loc[:, ('Time', 'zscore')]
                    self.final_dataframe = self.final_dataframe.reset_index(drop=True)
                    self.final_dataframe = self.final_dataframe.rename(
                        columns={'Time': colname_1, 'zscore': colname_2})

                    self.partial_dataframe = trial_deltaf.loc[:, 'zscore']
                    self.partial_dataframe = self.partial_dataframe.to_frame()
                    self.partial_dataframe = self.partial_dataframe.reset_index(drop=True)
                    self.partial_dataframe = self.partial_dataframe.rename(columns={'zscore': colname_2})
                    
                    self.partial_deltaf = trial_deltaf.loc[:,'DeltaF']
                    self.partial_deltaf = self.partial_deltaf.to_frame()
                    self.partial_deltaf = self.partial_deltaf.reset_index(drop=True)
                    self.partial_deltaf = self.partial_deltaf.rename(columns={'DeltaF': colname_2})
                    
                    self.final_deltaf = trial_deltaf.loc[:, ('Time', 'DeltaF')]
                    self.final_deltaf = self.final_deltaf.to_frame()
                    self.final_deltaf = self.final_deltaf.reset_index(drop=True)
                    self.final_deltaf = self.final_deltaf.rename(columns={'Time': colname_1, 'DeltaF': colname_2})

                    self.partial_percent = trial_deltaf.loc[:,'percent_change']
                    self.partial_percent = self.partial_percent.to_frame()
                    self.partial_percent = self.partial_percent.reset_index(drop=True)
                    self.partial_percent = self.partial_percent.rename(columns={'percent_change': colname_2})
                    
                    self.final_percent = trial_deltaf.loc[:, ('Time', 'percent_change')]
                    self.final_percent = self.final_percent.reset_index(drop=True)
                    self.final_percent = self.final_percent.rename(columns={'Time': colname_1, 'percent_change': colname_2})  
                    
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
                        
                        
                    self.partial_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    self.partial_deltaf.loc[:,colname_2] = trial_deltaf.loc[:,'DeltaF']
                    self.final_dataframe.loc[:,colname_1] = trial_deltaf.loc[:,'Time']
                    self.final_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    self.final_deltaf.loc[:,colname_1] = trial_deltaf.loc[:,'Time']
                    self.final_deltaf.loc[:,colname_2] = trial_deltaf.loc[:,'DeltaF']
                    self.partial_percent.loc[:,colname_2] = trial_deltaf.loc[:,'percent_change']
                    self.final_percent.loc[:,colname_1] = trial_deltaf.loc[:,'Time']
                    self.final_percent.loc[:,colname_2] = trial_deltaf.loc[:,'percent_change']
                    trial_num += 1
                    
                

    def write_data(self,output_data,include_abet=False,filename_override=''):
        processed_list = [1,'Full','full']
        partial_list = [4,'SimpleZ','simple']
        final_list = [7,'TimedZ','timed']
        partialf_list = [2,'SimpleF','simplef']
        finalf_list = [5,'TimedF','timedf']
        partialp_list = [3,'SimpleP','simplep']
        finalp_list = [6,'TimedP','timedp']

        if self.abet_loaded == True:
            if include_abet == True:
                #end_path = filedialog.asksaveasfilename(title='Save Output Data',
                                                        #filetypes=(('Excel File', '*.xlsx'), ('all files', '*.')))

                abet_file = open(self.abet_file_path)
                abet_csv_reader = csv.reader(abet_file)
                colnames_found = False
                colnames = list()
                abet_raw_data = list()

                for row in abet_csv_reader:
                    if colnames_found == False:
                        if len(row) == 0:
                            continue

                        if row[0] == 'Evnt_Time':
                            colnames_found = True
                            colnames = row
                            continue
                        else:
                            continue
                    else:
                        abet_raw_data.append(row)

                self.abet_pd = pd.DataFrame(self.abet_raw_data,columns=self.colnames)

                if output_data in processed_list:
                    with pd.ExcelWriter(self.end_path) as writer:
                        self.doric_pd.to_excel(writer, sheet_name='Photometry Data',index=False)
                        self.abet_pd.to_excel(writer, sheet_name='ABET Trial Data',index=False)
                elif output_data in partial_list:
                    with pd.ExcelWriter(self.end_path) as writer:
                        self.partial_dataframe.to_excel(writer, sheet_name='Photometry Data',index=False)
                        self.abet_pd.to_excel(writer, sheet_name='ABET Trial Data',index=False)
                elif output_data in final_list:
                    with pd.ExcelWriter(self.end_path) as writer:
                        self.final_dataframe.to_excel(writer, sheet_name='Photometry Data',index=False)
                        self.abet_pd.to_excel(writer, sheet_name='ABET Trial Data',index=False)

                return


        output_folder = self.main_folder_path + self.folder_symbol + 'Output'
        if (os.path.isdir(output_folder)) == False:
            os.mkdir(output_folder)
        if self.abet_loaded == True and self.anymaze_loaded == False:
            file_path_string = output_folder + self.folder_symbol +  output_data + '-' + self.animal_id + ' ' + self.date + ' ' + self.event_name + '.csv'
        else:
            current_time = datetime.now()
            current_time_string = current_time.strftime('%d-%m-%Y %H-%M-%S')
            file_path_string = output_folder + self.folder_symbol +  output_data + '-' + current_time_string + '.csv'
        
        if filename_override != '':
            file_path_string = filename_override + '-' + output_data + '.csv'

        print(file_path_string)
        if output_data in processed_list:
            self.doric_pd.to_csv(file_path_string,index=False)
        elif output_data in partial_list:
            self.partial_dataframe.to_csv(file_path_string,index=False)
        elif output_data in final_list:
            self.final_dataframe.to_csv(file_path_string,index=False)
        elif output_data in partialf_list:
            self.partial_deltaf.to_csv(file_path_string,index=False)
        elif output_data in finalf_list:
            self.final_deltaf.to_csv(file_path_string,index=False)
        elif output_data in partialp_list:
            self.partial_percent.to_csv(file_path_string,index=False)
        elif output_data in finalp_list:
            self.final_percent.to_csv(file_path_string,index=False)

## Functions ##
def abet_extract_information(abet_file_path):
    abet_file_path = abet_file_path
    abet_file = open(abet_file_path)
    abet_csv_reader = csv.reader(abet_file)
    event_time_colname = ['Evnt_Time','Event_Time']
    colnames_found = False
    for row in abet_csv_reader:
        if colnames_found == False:
            if len(row) == 0:
                continue
            if row[0] == 'Animal ID':
                animal_id = str(row[1])
                continue
            if row[0] == 'Date/Time':
                date = str(row[1])
                date = date.replace(':','-')
                date = date.replace('/','-')
                continue
            if (row[0] =='Schedule') or (row[0] == 'Schedule Name'):
                schedule = str(row[1])
            if row[0] in event_time_colname:
                colnames_found = True
        else:
            break
    abet_file.close()
    return animal_id, date, schedule         
            
## Config Load ##
curr_dirr = os.getcwd()

config_ini = curr_dirr + '\config.ini'
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

event_prior= float(config_file['Event_Window']['event_prior'])
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
center_method = config_file['ITI_Window']['center_method']

exclusion_list = config_file['Filter']['exclusion_list']
exclusion_list = exclusion_list.split(',')


## Run the Batch ##


for row_index, row in file_csv.iterrows():
    analyzer = Photometry_Data()
    
    abet_path = row.loc['abet_path']
    doric_path = row.loc['doric_path']
    
    ctrl_col_index = row.loc['ctrl_col_num']
    active_col_index = row.loc['act_col_num']
    ttl_col_index = row.loc['ttl_col_num']
    
    
    animal_id, date, schedule = abet_extract_information(abet_path)
    
    analyzer.load_abet_data(abet_path)
    analyzer.load_doric_data(doric_path, ctrl_col_index, active_col_index, ttl_col_index)
    try:
        analyzer.abet_trial_definition(start_group_name,end_group_name)
    except:
        print('Fail to Identify Definition Structure')
        continue
    
    try:
        analyzer.abet_doric_synchronize()
    except:
        print('No Sync Event')
        continue
    analyzer.doric_process(filter_frequency=filter_frequency)
    
    raw_processed = False
    
    
    
    
    for row_index2, row2 in event_csv.iterrows():
        null_check = row2.isnull()
        if null_check['filter_type'] == True:
            file_string = animal_id + '-' + schedule + '-' + row2.loc['event_name'] + '-' + date 
            file_dir = output_path + file_string
        elif null_check['filter_type'] == False and null_check['filter_arg'] == True:
            file_string = animal_id + '-' + schedule + '-' + row2.loc['event_name'] + '-' + row2.loc['filter_name'] + '-' + date
            file_dir = output_path + file_string
        elif null_check['filter_type'] == False and null_check['filter_arg'] == False:
            file_string = animal_id + '-' + schedule + '-' + row2.loc['event_name'] + '-' + row2.loc['filter_name'] + '-' + str(row2.loc['filter_arg']) + '-' + date
            file_dir = output_path + file_string
        if os.path.exists(file_dir):
            continue

        
        if null_check['filter_type'] == True:
            try:
                analyzer.abet_search_event(start_event_id=row2.loc['event_type'],
                                           start_event_item_name=row2.loc['event_name'],
                                           start_event_group=row2.loc['event_group'],
                                           extra_prior_time=event_prior,extra_follow_time=event_follow,
                                           centered_event=True)
            except:
                print('failed to find event')
                continue
            
        elif null_check['filter_type'] == False and null_check['filter_arg'] == False:
            try:
                analyzer.abet_search_event(start_event_id=row2.loc['event_type'],
                                           start_event_item_name=row2.loc['event_name'],
                                           start_event_group=row2.loc['event_group'],
                                           extra_prior_time=event_prior,extra_follow_time=event_follow,
                                           centered_event=True,filter_event=True,filter_before=row2.loc['filter_prior'],
                                           filter_event_id=row2.loc['filter_type'],filter_event_group=row2.loc['filter_group'],
                                           filter_event_item_name=row2.loc['filter_name'], filter_event_arg = row2.loc['filter_arg'],
                                           exclusion_list = exclusion_list)
            except:
                print('failed to find event with filter')
                continue
        
        elif null_check['filter_type'] == False and null_check['filter_arg'] == True:
            try:
                analyzer.abet_search_event(start_event_id=row2.loc['event_type'],
                                           start_event_item_name=row2.loc['event_name'],
                                           start_event_group=row2.loc['event_group'],
                                           extra_prior_time=event_prior,extra_follow_time=event_follow,
                                           centered_event=True,filter_event=True,filter_before=row2.loc['filter_prior'],
                                           filter_event_id=row2.loc['filter_type'],filter_event_group=row2.loc['filter_group'],
                                           filter_event_item_name=row2.loc['filter_name'],
                                           exclusion_list = exclusion_list)
            except:
                print('failed to find event with filter')
                continue
        if analyzer.abet_event_times.shape[0] < 1:
            print('no events located')
            continue
        
        
        try:
            analyzer.trial_separator(normalize=True,center_z_on_iti=center_z_on_iti,
                                     trial_definition=1,trial_iti_pad=iti_prior, center_method = center_method)
            
        except:
            print('failed to separate trials')
            continue
        
        if run_simplez == 1:
            analyzer.write_data('SimpleZ',filename_override=file_dir)
        if run_timedz == 1:
            analyzer.write_data('TimedZ',filename_override=file_dir)
        if run_simplep == 1:
            analyzer.write_data('SimpleP',filename_override=file_dir)
        if run_timedp == 1:
            analyzer.write_data('TimedP',filename_override=file_dir)
        if run_simplef == 1:
            analyzer.write_data('SimpleF',filename_override=file_dir)
        if run_timedf == 1:
            analyzer.write_data('TimedF',filename_override=file_dir)
        if run_raw == 1 and raw_processed == False:
            file_path_raw = output_path + animal_id + '-' + schedule + '-' + date
            analyzer.write_data('Full',filename_override=file_path_raw)
            raw_processed = True
            
            
        
        
