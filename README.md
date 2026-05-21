# PhotoBatch

PhotoBatch is a Python package for batch processing paired behavioural and fibre photometry datasets. The current pipeline supports ABET II behavioural exports and Doric photometry recordings, with both GUI and headless command-line entry points.

Example templates are included at the repository root:

- `File_Pair_Example_Sheet.csv`
- `Event_Sheet_Example.csv`
- `Event_Sheet_Example_Legacy.csv`

# Requirements

## Platform

PhotoBatch runs anywhere Python 3.8 or newer is supported, including Windows, macOS, and Linux. Higher core counts improve throughput for multi-session batch processing.

## Installation

Standard installation installs the CLI-capable pipeline without GUI extras:

```bash
pip install .
```

GUI installation adds PySide6 and matplotlib:

```bash
pip install ".[gui]"
```

Developer installation keeps the package editable during local development:

```bash
pip install -e ".[gui]"
```

# Usage

PhotoBatch can be launched either through the packaged console script or directly through the repository entry point.

## GUI mode

Launch the desktop application with either of the following:

```bash
photobatch
```

```bash
python main.py
```

If GUI dependencies are not installed, PhotoBatch prints an explicit error telling you to install the `gui` extra.

## Headless CLI mode

Use headless mode for scripted runs, automation, or server environments without a display.

```bash
photobatch --headless \
  --file-sheet File_Pair_Example_Sheet.csv \
  --event-sheet Event_Sheet_Example.csv \
  --config photobatch/config.json \
  --workers 4
```

Equivalent repository-local invocation:

```bash
python main.py --headless \
  --file-sheet File_Pair_Example_Sheet.csv \
  --event-sheet Event_Sheet_Example.csv \
  --config photobatch/config.json \
  --workers 4
```

Available CLI flags:

- `--headless`: run the batch pipeline without launching the GUI.
- `--file-sheet`, `-f`: CSV file describing ABET and Doric file pairs.
- `--event-sheet`, `-e`: CSV file describing events and optional filters.
- `--config`, `-c`: path to a config JSON file or to the `photobatch/` config directory.
- `--output-dir`, `-o`: override the configured output directory.
- `--workers`, `-w`: number of worker processes for batch execution.

When running headless, PhotoBatch requires `--file-sheet` and `--event-sheet`. If `--workers` is omitted, the value from the config is used.

# Input Files

## File pair sheet

The file pair sheet matches each ABET export with its corresponding photometry file.

For Doric CSV exports in `col_index` mode, specify numeric column indices where column `0` is the time axis.

For Doric HDF5 exports, specify Doric analog input/output identifiers as configured by the lock-in acquisition format.

## Event sheet

The event sheet defines which behavioural events should be extracted and how they should be filtered.

Supported primary event families include:

- `Condition Event`
- `Input Transition On/Off Event`
- `Touch Down/Up Event`

Supported filter families include:

- `Condition Event`
- `Variable Event`

Each event row can include a primary event name, group, optional argument, and zero or more filters with comparison operators such as `<`, `<=`, `!=`, `>=`, `>`, `inlist`, and `notinlist`.

# Configuration

The default merged configuration starts at `photobatch/config.json` and pulls in section-specific JSON files from the package.

Common sections include:

## Filepath

- `file_list_path`: default path to the file pair sheet.
- `event_list_path`: default path to the event sheet.
- `output_path`: directory for CSV, Excel, and persisted results output.

## Event_Window

- `event_prior`: seconds captured before each event.
- `event_follow`: seconds captured after each event.

## Normalization

- `iti_prior_trial`: ITI window length used when normalising against ITI data.
- `center_z`: normalization mode, such as `whole`, `prior`, or `iti`.
- `center_method`: `mean` or `median`.
- `normalize_side`: side or segment used for `prior` and `iti` normalization.
- `scale_median`: scale median absolute deviation to a standard-deviation-like value.

## Signal_Filter

- `filter_type`: filtering mode such as `lowpass` or `smoothing`.
- `filter_name`: filter family such as `butterworth`, `bessel`, or `chebychev`.
- `filter_order`: filter order.
- `filter_cutoff`: cutoff frequency in Hz.

## Signal_Utilities

- `despike`: enable or disable pre-filter despiking.
- `despike_window`: despiking window size.
- `despike_threshold`: MAD multiplier used to detect spikes.
- `crop_start`: seconds to remove from the start of the signal.
- `crop_end`: seconds to remove from the end of the signal.

## Signal_Fitting

- `fit_type`: baseline fit mode such as `linear` or `expodecay`.
- `baseline_detrend`: optional drift-removal mode such as `arpls`.
- `robust_fit`: enable Huber regression for linear channel fitting.
- `huber_epsilon`: Huber threshold or `auto`.

## Output

Output flags are boolean values that determine which export files are written:

- `create_simplez`
- `create_timedz`
- `create_simplef`
- `create_timedf`
- `create_raw`
- `create_summaryz`
- `create_summaryf`

# Persisted Analysis Data

PhotoBatch writes analysis results to a project-level HDF5 file named `temp.hdf5` by default instead of keeping all processed results in memory.

The persisted store contains:

- a flat index of animals, dates, sessions, behaviours, and summary metrics
- per-event peri-event arrays stored as individual HDF5 groups for on-demand loading
- metadata describing the run inputs and serialized configuration snapshot

This allows the GUI to reopen previous analysis data, save a copy of the store, and regenerate plots or summary tables without rerunning the full batch pipeline.

# Processing Notes

Photometry signals are filtered, fitted to estimate baseline or channel relationship, converted to delta-F/F, synchronized to behavioural time through TTL pulse alignment, and then segmented into peri-event trials for z-score and summary calculations.
