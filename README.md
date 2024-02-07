# PhotoBatch
PhotoBatch is a specialized Python script for processing several pairs of behavioural data and fiber photometry data.

Currently, this is Legacy Code that supports ABET II raw output and Doric Lenses Fiber Photometry Data in csv format. 

Example templates are presented for file and event sheets.

# Requirements

## PC Requirements
Photobatch can run on any system that a standard Python 3 install can be run on. This will include Windows, Linux, and Mac OS operating systems. There is no hard requirement on CPU, however CPU's with higher clock speed and core count will have faster performance.

## Software Requirements

### Python
Photobatch requires an installation of Python 3.8 or newer. For simplier installation and usage, it is highly reccomended that users download the Anaconda distribution of Python 3, as most of the Python packages that are required will be included with the installation.

### Python packages
Photobatch requires the following packages:
Numpy
Scipy
Pandas
H5Py

These can be installed with the following code from the command prompt (Windows) or terminal (Linux, Mac OS):

```
  py -m pip install numpy scipy pandas h5py
```
# Instructions
## Step 1: Set up File Pairs
In order to use Photobatch, the user must match the ABET raw time series data (csv format) with the photometry data. Users can refer to the "File_Pair_Example_Sheet.csv" for a template to create a pair sheet. Users will need to specify a filepath to the photometry data and the behaviour data.

There are two procedures for specifying photometry depending on the data type used.

### Doric v5 csv Format
Users should specify the column index numbers (with 0 specifying the time column). Users will be able to specify an isobestic, active, and ttl column index.

### Doric v6 hdf5 Format
Users utilizing Doric Neuroscience Studio 6 and a lock-in method can specify the analog input/output channel values to identify the isobestic, active, and ttl data sources.

## Step 2: Set up an Event Sheet
Once the file pairs have been created, users can create an event sheet to dictate the specific behaviors that will be parsed in the ABET raw time series data. 

### Step 2.1: Establish the Behaviour of Interest
The primary behaviour can be one of the following event types in the time series output:
- Condition Event (stage)
- Input Transition On/Off Event
- Touch Down/Up Event
Users will be required to specify an event name (corresponding to Item_Name), an event group (same as time series), and an event argument (only if necessary and defined.)

### Step 2.2: Create Filters
In addition to setting the behaviour of interest, users may add an unlimited number of filters to further narrow events of interest. Currently, the following event types are supported as filters:
- Condition Event
- Variable Event

Users will be required to input an event type, event name, event group, and event argument. In addition to these parameters, users can define a logical operator for variable events (<, <=, ==, !=, >=, >) or list operator (inlist, notinlist). Finally, users can use a boolean flag to identify whether the filter should be checked before the primary event of interest or after.

## Step 3: Set up a Config file
Users will need to edit the configuration file to customize the data processing. The fields will be described below:
### Filepath
file_list_path: The filepath for the file list sheet (see example for structure)
event_list_path: The filepath for the event list sheet (see legacy and current examples)
output_path: The folder path for where output will be generated
### Event Window
event_prior: The amount of time (sec) that should be captured prior to an event
event_follow: The amount of time (sec) that should be captured following an event
### ITI Window
trial_start_stage: The ABET II Condition Events that defines the start of a trial structure. Is used to identify non-trial window prior to event.
trial_end_stage: The ABET II Condition Events that defines the end of a trial structure.
iti_prior_trial: If using the ITI to generate z-Scores, specify the amount of time (sec) to use prior to the start stages.
center_z_on_iti: A binary value to denote whether the iti period should be used for generating z-scores
center_method: Specifies the method used to calculate the z-score. Two methods currently exist. mean - uses the mean and standard deviation. median - use the median and median absolute deviation.
### Photometry Processing
filter_frequency: The frequency (hz) used as the cutoff for the Butterworth Low-Pass Filter
### Output
All values are binary
create_simplez: Creates an output with single columns for the z-scores for every event
create_timedz: Creates an output with two columns for the time and z-scores for every event
create_simplep: Creates an output with single columns for the percent changed for every event
create_timedp: Creates an output with two columns for the time and percent changed for every event
create_simplef: Creates an output with single columns for the delta-f values for every event
create_timedf: Creates an output with two columns for the time and delta-f values for every event
create_raw: Creates an output with the entire time and delta-f values for the entire recording

## Step 4: Run Script
Once your files are configured, run the Photometry Analyzer BATCH.py script to process the data files. All desired outputs will be moved to the output path.

#Notes
## Signal Processing
Photometry data from the isobestic and active channels are passed through a 2nd order Low Pass Butterworth Filter to remove noise. Following this, least squares regression is used to fit the isobestic and active channel data. Finally, Delta-F is calculated for the entire event.
## Integration
The two sources of data are synchronized through examining the pattern of TTL pulses. Events are searched using the criteria and filters present in the event file sheet. Once events are identified, all relevant data is separated and processed using the settings previously specified.
