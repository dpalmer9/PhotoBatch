"""IO/output.py
Output writing functions for PhotoBatch analysis results.

Functions
---------
write_data    – write per-event trial data to CSV (wide or long format).
write_summary – append per-session aggregate statistics to an Excel workbook.
"""

import os
import pandas as pd
from datetime import datetime
from pathlib import Path


def write_data(output_data, main_folder_path,
               animal_id, date, event_name, event_alias,
               doric_pd, partial_dataframe, final_dataframe,
               partial_deltaf, final_deltaf,
               filename_override='', format='wide'):
    """Write per-event trial data to a CSV file.

    Parameters
    ----------
    output_data : int
        Output type selector (1–5):
        1 / 'SimpleZ'  – z-score without time column.
        2 / 'TimedZ'   – z-score with time column.
        3 / 'SimpleF'  – delta-F without time column.
        4 / 'TimedF'   – delta-F with time column.
        5 / 'Full'     – raw doric_pd (Time + DeltaF).
    main_folder_path : str or Path
        Root folder; an 'Output' sub-folder is created automatically.
    animal_id : str
    date : str
    event_name : str
    event_alias : str
    doric_pd : pd.DataFrame   columns=['Time','DeltaF']
    partial_dataframe : pd.DataFrame   z-score trial columns.
    final_dataframe : pd.DataFrame     time + z-score trial columns.
    partial_deltaf : pd.DataFrame      delta-F trial columns.
    final_deltaf : pd.DataFrame        time + delta-F trial columns.
    filename_override : str
        If non-empty, used as the full path prefix (the output type string
        and '.csv' are appended).
    format : str
        'wide' (default) or 'long'.
    """
    partial_list    = [1, 'SimpleZ', 'simple']
    final_list      = [2, 'TimedZ',  'timed']
    partialf_list   = [3, 'SimpleF', 'simplef']
    finalf_list     = [4, 'TimedF',  'timedf']
    processed_list  = [5, 'Full',    'full']
    output_string_list = ['SimpleZ', 'TimedZ', 'SimpleF', 'TimedF', 'Raw']
    output_string = output_string_list[output_data - 1]

    output_folder = Path(main_folder_path) / 'Output'
    output_folder.mkdir(exist_ok=True)

    if animal_id:
        label = event_alias if event_alias else event_name
        file_name_str = f"{output_string}-{animal_id} {date} {label}.csv"
        file_path_string = str(output_folder / file_name_str)
    else:
        current_time_string = datetime.now().strftime('%d-%m-%Y %H-%M-%S')
        file_name_str = f"{output_string}-{current_time_string}.csv"
        file_path_string = str(output_folder / file_name_str)

    if filename_override:
        file_path_string = filename_override + '-' + output_string + '.csv'

    print(file_path_string)

    if format == 'long':
        if output_data in processed_list:
            output_long = doric_pd.melt(var_name='Time', value_name='DeltaF')
        elif output_data in partial_list:
            output_long = partial_dataframe.melt(var_name='Trial', value_name='Z-Score')
        elif output_data in final_list:
            output_long = final_dataframe.melt(var_name='Trial', value_name='Z-Score')
        elif output_data in partialf_list:
            output_long = partial_deltaf.melt(var_name='Trial', value_name='DeltaF')
        elif output_data in finalf_list:
            output_long = final_deltaf.melt(var_name='Trial', value_name='DeltaF')
        else:
            return
        output_long.to_csv(file_path_string, index=False)
    else:
        if output_data in processed_list:
            doric_pd.to_csv(file_path_string, index=False)
        elif output_data in partial_list:
            partial_dataframe.to_csv(file_path_string, index=False)
        elif output_data in final_list:
            final_dataframe.to_csv(file_path_string, index=False)
        elif output_data in partialf_list:
            partial_deltaf.to_csv(file_path_string, index=False)
        elif output_data in finalf_list:
            final_deltaf.to_csv(file_path_string, index=False)


def write_summary(output_data, summary_string, output_path, session_string,
                  partial_dataframe, partial_deltaf, partial_percent):
    """Append per-session aggregate statistics (mean, std, sem) to an Excel file.

    The workbook is created on first call and extended on subsequent calls,
    supporting batch accumulation across many sessions.

    Parameters
    ----------
    output_data : int
        Type selector (1=SummaryZ, 2=SummaryF, 3=SummaryP).
    summary_string : str
        Prefix used to build the output file name.
    output_path : str
        Directory where the Excel file is written.
    session_string : str
        Label for this session (used as the first column value in each row).
    partial_dataframe : pd.DataFrame   z-score trial columns.
    partial_deltaf : pd.DataFrame      delta-F trial columns.
    partial_percent : pd.DataFrame     percent-change trial columns.
    """
    z_list = [1, 'SummaryZ', 'summaryz']
    f_list = [2, 'SummaryF', 'summaryf']
    p_list = [3, 'SummaryP', 'summaryp']
    output_string_list = ['SummaryZ', 'SummaryF', 'SummaryP']
    output_string = output_string_list[output_data - 1]

    summary_path = (output_path + summary_string + 'Summary'
                    + '-' + output_string + '-' + '.xlsx')

    if output_data in z_list:
        session_temp = partial_dataframe.transpose()
    elif output_data in f_list:
        session_temp = partial_deltaf.transpose()
    elif output_data in p_list:
        session_temp = partial_percent.transpose()
    else:
        return

    session_mean = pd.DataFrame([[session_string] +
                                 session_temp.mean(axis=0, skipna=True).tolist()])
    session_std  = pd.DataFrame([[session_string] +
                                 session_temp.std(axis=0,  skipna=True).tolist()])
    session_sem  = pd.DataFrame([[session_string] +
                                 session_temp.sem(axis=0,  skipna=True).tolist()])

    if os.path.exists(summary_path):
        summary_xlsx = pd.read_excel(summary_path, sheet_name=None, header=None)
        xlsx_mean = pd.concat([summary_xlsx.get('Mean'), session_mean])
        xlsx_std  = pd.concat([summary_xlsx.get('Std'),  session_std])
        xlsx_sem  = pd.concat([summary_xlsx.get('Sem'),  session_sem])
    else:
        xlsx_mean = session_mean
        xlsx_std  = session_std
        xlsx_sem  = session_sem

    with pd.ExcelWriter(summary_path) as writer:
        xlsx_mean.to_excel(writer, sheet_name='Mean', header=False, index=False)
        xlsx_std.to_excel(writer,  sheet_name='Std',  header=False, index=False)
        xlsx_sem.to_excel(writer,  sheet_name='Sem',  header=False, index=False)
