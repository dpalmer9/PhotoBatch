import numpy as np

from photobatch.Processing.Signal.filter import signal_filter
from photobatch.Processing.Signal import fitting
from photobatch.Processing.Signal.fitting import _arpls_drift_fit, signal_fit
from photobatch.Processing.Signal.utilities import despike_signal


def test_despike_signal_replaces_transient_spikes():
    clean_signal = np.sin(np.linspace(0.0, 10.0, 1000))
    spiked_signal = clean_signal.copy()
    spiked_signal[100] = 50.0
    spiked_signal[500] = -40.0

    despiked = despike_signal(spiked_signal, window=51, threshold=5.0)

    assert np.abs(despiked[100] - clean_signal[100]) < 1.0
    assert np.abs(despiked[500] - clean_signal[500]) < 1.0


def test_signal_filter_resamples_to_uniform_grid_and_reduces_noise(synthetic_signal):
    irregular = synthetic_signal.copy()
    irregular["Time"] = np.cumsum(np.r_[0.0, np.diff(irregular["Time"]) * 1.02])

    time, control_filtered, active_filtered, sample_frequency = signal_filter(
        irregular,
        filter_type="lowpass",
        filter_name="butterworth",
        filter_order=4,
        filter_cutoff=2.0,
        despike=False,
    )

    assert len(time) == len(control_filtered) == len(active_filtered)
    assert np.allclose(np.diff(time), np.diff(time)[0], atol=1e-10)
    assert np.isclose(sample_frequency, 98.0392156862745)
    assert np.std(np.diff(active_filtered)) < np.std(np.diff(irregular["Active"]))


def test_signal_fit_linear_returns_deltaf_dataframe(synthetic_signal):
    result = signal_fit(
        "linear",
        synthetic_signal["Control"].to_numpy(),
        synthetic_signal["Active"].to_numpy(),
        synthetic_signal["Time"].to_numpy(),
        robust_fit=True,
        huber_epsilon=1.35,
    )

    assert list(result.columns) == ["Time", "DeltaF"]
    assert len(result) == len(synthetic_signal)
    assert np.abs(result["DeltaF"].mean()) < 0.05
    assert result["DeltaF"].std() > 0.005


def test_signal_fit_expodecay_tracks_oscillatory_component():
    time = np.linspace(0.0, 40.0, 4000)
    bleaching = 5.0 * np.exp(-0.08 * time) + 10.0
    oscillation = 0.4 * np.sin(2.0 * np.pi * 0.5 * time)
    active = bleaching + oscillation
    control = bleaching

    result = signal_fit("expodecay", control, active, time, robust_fit=False)

    recovered = result["DeltaF"].to_numpy()
    target = oscillation / bleaching
    correlation = np.corrcoef(recovered, target)[0, 1]

    assert correlation > 0.9


def test_signal_fit_irls_handles_outliers_better_than_plain_linear():
    rng = np.random.default_rng(42)
    control = np.linspace(0.0, 10.0, 500)
    true_signal = 1.8 * control + 4.0
    active = true_signal + rng.normal(0.0, 0.05, size=control.size)
    outlier_idx = np.arange(0, control.size, 25)
    active[outlier_idx] += 8.0
    time = np.linspace(0.0, 50.0, control.size)

    irls_result = signal_fit("irls", control, active, time)
    linear_result = signal_fit("linear", control, active, time, robust_fit=False)

    clean_mask = np.ones(control.size, dtype=bool)
    clean_mask[outlier_idx] = False
    irls_error = np.mean(np.abs(irls_result["DeltaF"].to_numpy()[clean_mask]))
    linear_error = np.mean(np.abs(linear_result["DeltaF"].to_numpy()[clean_mask]))

    assert irls_error < linear_error


def test_signal_fit_biexponential_falls_back_gracefully_when_curve_fit_fails(monkeypatch):
    time = np.linspace(0.0, 60.0, 1200)
    control = 2.0 + 0.01 * time
    active = control + 0.05 * np.sin(time)

    def raise_curve_fit(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(fitting, "curve_fit", raise_curve_fit)

    result = signal_fit("biexponential", control, active, time)

    assert list(result.columns) == ["Time", "DeltaF"]
    assert len(result) == len(time)
    assert np.isfinite(result["DeltaF"]).all()


def test_arpls_drift_fit_recovers_slow_baseline():
    time = np.linspace(0.0, 100.0, 2000)
    drift = 0.5 * np.sin(2.0 * np.pi * 0.01 * time)
    signal = drift + 0.1 * np.sin(2.0 * np.pi * 1.0 * time)

    estimated_drift = _arpls_drift_fit(signal, arpls_lambda=1e6)
    correlation = np.corrcoef(estimated_drift, drift)[0, 1]

    assert correlation > 0.9