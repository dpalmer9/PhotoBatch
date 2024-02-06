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

# Config File Settings
## Filepath
file_list_path: The filepath for the file list sheet (see example for structure)
event_list_path: The filepath for the event list sheet (see legacy and current examples)
output_path: The folder path for where output will be generated

## Event Window
event_prior: The amount of time (sec) that should be captured prior to an event
event_follow: The amount of time (sec) that should be captured following an event

## ITI Window
trial_start_stage: The ABET II Condition Events that defines the start of a trial structure. Is used to identify non-trial window prior to event.
trial_end_stage: The ABET II Condition Events that defines the end of a trial structure.
iti_prior_trial: If using the ITI to generate z-Scores, specify the amount of time (sec) to use prior to the start stages.
center_z_on_iti: A binary value to denote whether the iti period should be used for generating z-scores
center_method: Specifies the method used to calculate the z-score. Two methods currently exist. mean - uses the mean and standard deviation. median - use the median and median absolute deviation.

## Photometry Processing
filter_frequency: The frequency (hz) used as the cutoff for the Butterworth Low-Pass Filter

## Output
All values are binary
create_simplez: Creates an output with single columns for the z-scores for every event
create_timedz: Creates an output with two columns for the time and z-scores for every event
create_simplep: Creates an output with single columns for the percent changed for every event
create_timedp: Creates an output with two columns for the time and percent changed for every event
create_simplef: Creates an output with single columns for the delta-f values for every event
create_timedf: Creates an output with two columns for the time and delta-f values for every event
create_raw: Creates an output with the entire time and delta-f values for the entire recording


# Signal Processing

Photometry data from the isobestic and active channels are passed through a 2nd order Low Pass Butterworth Filter to remove noise. Following this, least squares regression is used to fit the isobestic and active channel data. Finally, Delta-F is calculated for the entire event.

# Integration

The two sources of data are synchronized through examining the pattern of TTL pulses. Events are searched using the criteria and filters present in the event file sheet. Once events are identified, all relevant data is separated and processed using the settings previously specified.
