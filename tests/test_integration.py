import csv
from pathlib import Path

import numpy as np
import pandas as pd

from photobatch.exceptions import SynchronizationError
from photobatch.Processing import hdf_store
from photobatch.Processing.Process.advanced_analysis import (
    run_flmm_analysis,
    run_glm_hmm_analysis,
    run_moa_hmm_analysis,
)
from photobatch.Processing.data_processor import _process_single_file, process_files
from photobatch.config_manager import ConfigManager


def _write_abet_csv(path: Path):
    rows = [
        ["Animal ID", "Mouse-1"],
        ["Date/Time", "2024/01/02 12:00:00"],
        ["Schedule", "Synthetic"],
        ["Evnt_Time", "Evnt_ID", "Evnt_Name", "Item_Name", "Group_ID", "Arg1_Value"],
        [1.0, "1", "Output Event", "TTL #1", "99", ""],
        [5.0, "1", "Output Event", "TTL #1", "99", ""],
        [9.0, "1", "Output Event", "TTL #1", "99", ""],
        [2.0, "1", "Condition Event", "Display Image", "20", ""],
        [3.0, "1", "Condition Event", "Reward Collected Start ITI", "8", ""],
        [6.0, "1", "Condition Event", "Display Image", "20", ""],
        [7.0, "1", "Condition Event", "Reward Collected Start ITI", "8", ""],
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _write_doric_csv(path: Path):
    time = np.arange(0.0, 12.0, 0.05)
    control = 10.0 + 0.2 * np.sin(2.0 * np.pi * 0.2 * time)
    active = control + 0.5 * np.sin(2.0 * np.pi * 1.0 * time)
    ttl = np.zeros_like(time)
    for pulse_time in (1.0, 5.0, 9.0):
        pulse_index = np.searchsorted(time, pulse_time)
        ttl[pulse_index:pulse_index + 2] = 5.0

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Doric export", "", "", ""])
        writer.writerow(["Time", "Control", "Active", "TTL"])
        for row in zip(time, control, active, ttl):
            writer.writerow(row)


def _write_event_sheet(path: Path):
    rows = [
        {
            "event_type": "Condition Event",
            "event_name": "Display Image",
            "event_group": 20,
            "num_filter": 0,
            "event_alias": "Display Image",
        }
    ]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["event_type", "event_name", "event_group", "num_filter", "event_alias"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_file_sheet(path: Path, abet_path: Path, doric_path: Path):
    rows = [
        {
            "abet_path": str(abet_path),
            "doric_path": str(doric_path),
            "ctrl_col_num": 1,
            "act_col_num": 2,
            "ttl_col_num": 3,
            "mode": "col_index",
        }
    ]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["abet_path", "doric_path", "ctrl_col_num", "act_col_num", "ttl_col_num", "mode"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_process_files_persists_results_to_hdf5(tmp_path, monkeypatch):
    abet_path = tmp_path / "session_abet.csv"
    doric_path = tmp_path / "session_doric.csv"
    event_sheet_path = tmp_path / "event_sheet.csv"
    file_sheet_path = tmp_path / "file_sheet.csv"
    hdf5_path = tmp_path / "results.hdf5"

    _write_abet_csv(abet_path)
    _write_doric_csv(doric_path)
    _write_event_sheet(event_sheet_path)
    _write_file_sheet(file_sheet_path, abet_path, doric_path)

    repo_root = Path(__file__).resolve().parents[1]
    config = ConfigManager(repo_root / "photobatch")
    config["Event_Window"]["event_prior"] = 0.5
    config["Event_Window"]["event_follow"] = 0.5
    config["Normalization"]["center_z"] = "whole"
    config["Normalization"]["center_method"] = "mean"
    config["Normalization"]["normalize_side"] = "Left"
    config["Normalization"]["iti_prior_trial"] = 0.5
    config["Signal_Filter"]["filter_type"] = "lowpass"
    config["Signal_Filter"]["filter_name"] = "butterworth"
    config["Signal_Filter"]["filter_order"] = 2
    config["Signal_Filter"]["filter_cutoff"] = 2
    config["Signal_Utilities"]["despike"] = False
    config["Signal_Utilities"]["despike_window"] = 11
    config["Signal_Fitting"]["fit_type"] = "linear"
    config["Signal_Fitting"]["baseline_detrend"] = None
    config["Signal_Fitting"]["robust_fit"] = False
    config["ABET"]["trial_start_stage"] = "Display Image"
    config["ABET"]["trial_end_stage"] = "Reward Collected Start ITI"
    config["ABET"]["exclusion_list"] = ""

    monkeypatch.setattr(hdf_store, "get_default_results_path", lambda: hdf5_path)

    config["Filepath"]["output_path"] = str(tmp_path)

    results_path = process_files(
        file_sheet_path=str(file_sheet_path),
        event_sheet_path=str(event_sheet_path),
        output_options=[1, 2, 3, 4, 5, 6, 7],
        config=config,
        num_workers=1,
    )

    assert Path(results_path).exists()

    # Check for wide/CSV output files
    output_dir = tmp_path / "Output"
    assert output_dir.exists()
    assert (output_dir / "SimpleZ-Mouse-1 2024-01-02 12-00-00 Display Image.csv").exists()
    assert (output_dir / "TimedZ-Mouse-1 2024-01-02 12-00-00 Display Image.csv").exists()
    assert (output_dir / "SimpleF-Mouse-1 2024-01-02 12-00-00 Display Image.csv").exists()
    assert (output_dir / "TimedF-Mouse-1 2024-01-02 12-00-00 Display Image.csv").exists()
    assert (output_dir / "Raw-Mouse-1 2024-01-02 12-00-00 Display Image.csv").exists()

    # Check for Excel summary files
    assert (tmp_path / "Display Image_Summary-SummaryZ-.xlsx").exists()
    assert (tmp_path / "Display Image_Summary-SummaryF-.xlsx").exists()

    index_df = hdf_store.load_results_index(results_path)
    assert len(index_df) == 1
    assert index_df.iloc[0]["behavior"] == "Display Image"
    assert index_df.iloc[0]["animal_id"] == "Mouse-1"
    assert index_df.iloc[0]["session"] == "Session 1"

    result_id = index_df.iloc[0]["result_id"]
    plot_df = hdf_store.load_plot_data(results_path, result_id)
    assert not plot_df.empty
    assert list(plot_df.columns) == ["Z-Score Trial 1", "Z-Score Trial 2"]

    session_payload = hdf_store.load_session_traces(results_path, "Mouse-1_2024-01-02")
    trace_table = session_payload["trace_table"]
    assert not trace_table.empty
    assert list(trace_table.columns) == [
        "Time",
        "Raw_Control",
        "Raw_Active",
        "Filtered_Control",
        "Filtered_Active",
        "DeltaF",
    ]
    event_table = session_payload["event_table"]
    assert not event_table.empty
    assert "event_type" in event_table.columns


def test_hdf_store_persists_advanced_analysis_payloads(tmp_path):
    hdf5_path = tmp_path / "advanced_results.hdf5"
    store_path = hdf_store.initialize_results_file(hdf5_path)

    result_record = {
        "file": "session.csv",
        "behavior": "Display Image",
        "animal_id": "Mouse-1",
        "date": "2024-01-02",
        "time": "12-00-00",
        "datetime": "2024-01-02 12:00:00",
        "session": "Session 1",
        "max_peak": 1.2,
        "auc": 0.4,
        "plot_data": pd.DataFrame({"Z-Score Trial 1": [0.1, 0.2]}),
        "advanced_flmm": {
            "model_type": "flmm",
            "status": "ok",
            "grid_points": [0.0, 1.0],
            "trial_names": ["Trial 1"],
            "coefficient_curves": pd.DataFrame({"intercept": [0.5, 0.4]}, index=pd.Index([0.0, 1.0], name="time")),
            "covariance_parameters": pd.DataFrame({"time": [0.0, 1.0], "residual_variance": [0.1, 0.2]}),
            "design_matrices": {
                "fixed_effects": pd.DataFrame({"intercept": [1.0], "trial_index": [0.0]}),
                "random_effects": pd.DataFrame({"animal_id": ["Mouse-1"]}),
            },
            "fdata_shape": (1, 2),
        },
        "advanced_glm_hmm": {
            "model_type": "glm_hmm",
            "status": "ok",
            "transition_matrix": np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=float),
            "glm_weights": pd.DataFrame({"state": [0, 1], "intercept": [0.1, -0.2]}),
            "viterbi_path": np.asarray([0, 1, 1, 0], dtype=int),
            "design_matrix": pd.DataFrame({"cue": [0.0, 1.0, 0.5]}),
            "emission_parameters": pd.DataFrame({"state": [0, 1], "mean": [0.0, 0.4]}),
        },
        "advanced_moa_hmm": {
            "model_type": "moa_hmm",
            "status": "ok",
            "transition_matrix": np.asarray([[0.7, 0.3], [0.25, 0.75]], dtype=float),
            "agent_weights": pd.DataFrame({"state": [0, 1], "bias": [0.1, -0.1], "agent_1": [0.5, 0.2]}),
            "viterbi_path": np.asarray([1, 1, 0, 0], dtype=int),
            "agent_predictions": pd.DataFrame({"agent_1": [0.2, 0.5, 0.7]}),
            "emission_parameters": pd.DataFrame({"state": [0, 1], "variance": [0.1, 0.2]}),
        },
    }

    hdf_store.append_result(store_path, "result_000001", result_record)

    flmm_payload = hdf_store.load_advanced_result(store_path, "flmm", "result_000001")
    assert flmm_payload["model_type"] == "flmm"
    assert np.allclose(flmm_payload["coefficient_curves"]["intercept"].to_numpy(dtype=float), np.asarray([0.5, 0.4]))
    assert "fixed_effects" in flmm_payload["design_matrices"]

    glm_payload = hdf_store.load_advanced_result(store_path, "glm_hmm", "result_000001")
    assert glm_payload["model_type"] == "glm_hmm"
    assert glm_payload["transition_matrix"].shape == (2, 2)
    assert glm_payload["glm_weights"].shape[0] == 2

    moa_payload = hdf_store.load_advanced_result(store_path, "moa_hmm", "result_000001")
    assert moa_payload["model_type"] == "moa_hmm"
    assert moa_payload["transition_matrix"].shape == (2, 2)
    assert moa_payload["agent_weights"].shape[0] == 2


def test_native_advanced_analysis_entry_points_return_expected_shapes():
    plot_data = pd.DataFrame(
        {
            "Trial 1": [0.1, 0.3, 0.25, 0.2, 0.15],
            "Trial 2": [0.05, 0.28, 0.22, 0.18, 0.1],
            "Trial 3": [0.12, 0.31, 0.24, 0.17, 0.11],
            "Trial 4": [0.08, 0.27, 0.21, 0.16, 0.09],
        },
        index=np.linspace(-1.0, 1.0, 5),
    )
    flmm_result = run_flmm_analysis(
        plot_data,
        covariates={"trial_index": [0, 1, 2, 3]},
        random_effects={"animal_id": ["A", "A", "B", "B"]},
    )

    assert flmm_result["model_type"] == "flmm"
    assert not flmm_result["coefficient_curves"].empty
    assert "intercept" in flmm_result["coefficient_curves"].columns

    session_time = np.linspace(0.0, 9.9, 100)
    session_signal = 0.4 * np.sin(session_time) + 0.15 * np.cos(0.4 * session_time)
    events = pd.DataFrame({"time": [2.0, 5.0, 7.5], "event_type": ["cue", "reward", "cue"]})

    glm_result = run_glm_hmm_analysis(session_time, session_signal, events, n_states=2)
    assert glm_result["model_type"] == "glm_hmm"
    assert glm_result["transition_matrix"].shape == (2, 2)
    assert glm_result["viterbi_path"].shape[0] == session_time.shape[0]

    moa_result = run_moa_hmm_analysis(session_time, session_signal, None, events, n_states=2)
    assert moa_result["model_type"] == "moa_hmm"
    assert moa_result["transition_matrix"].shape == (2, 2)
    assert moa_result["viterbi_path"].shape[0] == session_time.shape[0]


def test_process_single_file_skips_when_synchronization_fails(monkeypatch, caplog, tmp_path):
    class StubSignalEventData:
        def __init__(self):
            self.behaviour_loaded = True
            self.signal_loaded = True

        def load_behaviour_data(self, filepath, vendor="abet"):
            return None

        def load_signal_data(self, filepath, ch1_col, ch2_col, ttl_col, mode, vendor="doric"):
            return None

        def synchronize_time(self, behaviour_vendor="abet", signal_vendor="doric"):
            raise SynchronizationError("synthetic failure")

    monkeypatch.setattr("photobatch.Processing.data_processor.SignalEventData", StubSignalEventData)

    event_sheet_path = tmp_path / "event_sheet.csv"
    pd.DataFrame(
        [{"event_type": "Condition Event", "event_name": "Display Image", "event_group": 20, "num_filter": 0}]
    ).to_csv(event_sheet_path, index=False)

    row_dict = {
        "abet_path": "synthetic_abet.csv",
        "doric_path": "synthetic_doric.csv",
        "ctrl_col_num": 1,
        "act_col_num": 2,
        "ttl_col_num": 3,
        "mode": "col_index",
    }
    args = (
        row_dict,
        str(event_sheet_path),
        [],
        0.5,
        0.5,
        ["Display Image"],
        ["Reward Collected Start ITI"],
        0.5,
        "whole",
        "mean",
        False,
        "Left",
        "lowpass",
        "butterworth",
        2,
        2,
        False,
        11,
        5.0,
        1.0,
        "linear",
        None,
        False,
        "auto",
        1e5,
        50,
        1e-6,
        1e-8,
        2.0,
        [],
        0.0,
        0.0,
    )

    with caplog.at_level("ERROR"):
        result = _process_single_file(args)

    assert result == []
    assert "Time synchronization failed for file pair" in caplog.text


def test_process_files_with_progress_callback(tmp_path, monkeypatch):
    abet_path = tmp_path / "session_abet.csv"
    doric_path = tmp_path / "session_doric.csv"
    event_sheet_path = tmp_path / "event_sheet.csv"
    file_sheet_path = tmp_path / "file_sheet.csv"
    hdf5_path = tmp_path / "results_progress.hdf5"

    _write_abet_csv(abet_path)
    _write_doric_csv(doric_path)
    _write_event_sheet(event_sheet_path)
    _write_file_sheet(file_sheet_path, abet_path, doric_path)

    repo_root = Path(__file__).resolve().parents[1]
    config = ConfigManager(repo_root / "photobatch")
    config["Event_Window"]["event_prior"] = 0.5
    config["Event_Window"]["event_follow"] = 0.5
    config["Normalization"]["center_z"] = "whole"
    config["Normalization"]["center_method"] = "mean"
    config["Normalization"]["normalize_side"] = "Left"
    config["Normalization"]["iti_prior_trial"] = 0.5
    config["Signal_Filter"]["filter_type"] = "lowpass"
    config["Signal_Filter"]["filter_name"] = "butterworth"
    config["Signal_Filter"]["filter_order"] = 2
    config["Signal_Filter"]["filter_cutoff"] = 2
    config["Signal_Utilities"]["despike"] = False
    config["Signal_Utilities"]["despike_window"] = 11
    config["Signal_Fitting"]["fit_type"] = "linear"
    config["Signal_Fitting"]["baseline_detrend"] = None
    config["Signal_Fitting"]["robust_fit"] = False
    config["ABET"]["trial_start_stage"] = "Display Image"
    config["ABET"]["trial_end_stage"] = "Reward Collected Start ITI"
    config["ABET"]["exclusion_list"] = ""

    monkeypatch.setattr(hdf_store, "get_default_results_path", lambda: hdf5_path)
    config["Filepath"]["output_path"] = str(tmp_path)

    progress_calls = []
    def progress_cb(completed, total):
        progress_calls.append((completed, total))

    process_files(
        file_sheet_path=str(file_sheet_path),
        event_sheet_path=str(event_sheet_path),
        output_options=[1],
        config=config,
        num_workers=1,
        progress_callback=progress_cb
    )

    assert len(progress_calls) == 1
    assert progress_calls[0] == (1, 1)