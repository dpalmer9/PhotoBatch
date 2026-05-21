import numpy as np
import pandas as pd
import pytest

from photobatch.exceptions import SynchronizationError
from photobatch.Processing.IO.sync import abet_doric_synchronize


def test_ttl_synchronization_recovers_affine_time_mapping():
    abet_ttl_times = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    doric_ttl_times = abet_ttl_times * 1.002 + 5.0

    doric_time = np.arange(0.0, 100.0, 0.01)
    doric_pd = pd.DataFrame({"Time": doric_time, "Active": np.sin(doric_time)})

    ttl_series = np.zeros_like(doric_time)
    for pulse_time in doric_ttl_times:
        pulse_index = np.searchsorted(doric_time, pulse_time)
        ttl_series[pulse_index:pulse_index + 10] = 5.0
    ttl_pd = pd.DataFrame({"Time": doric_time, "TTL": ttl_series})

    abet_pd = pd.DataFrame(
        {
            "Evnt_Time": abet_ttl_times,
            "Item_Name": ["TTL #1"] * len(abet_ttl_times),
            "Evnt_Name": ["Output Event"] * len(abet_ttl_times),
        }
    )

    synced_doric = abet_doric_synchronize(doric_pd, ttl_pd, abet_pd)

    pulse_index = np.searchsorted(doric_time, doric_ttl_times[0])
    synced_time = synced_doric.iloc[pulse_index]["Time"]
    expected_time = (doric_ttl_times[0] - 5.0) / 1.002

    assert np.abs(synced_time - expected_time) < 0.05


def test_ttl_synchronization_raises_when_too_few_pulses_exist():
    doric_pd = pd.DataFrame({"Time": np.arange(0.0, 5.0, 0.1)})
    ttl_pd = pd.DataFrame({"Time": np.arange(0.0, 5.0, 0.1), "TTL": np.zeros(50)})
    ttl_pd.loc[10, "TTL"] = 5.0
    abet_pd = pd.DataFrame(
        {
            "Evnt_Time": [1.0],
            "Item_Name": ["TTL #1"],
            "Evnt_Name": ["Output Event"],
        }
    )

    with pytest.raises(SynchronizationError, match="Fewer than 2 TTL pulses"):
        abet_doric_synchronize(doric_pd, ttl_pd, abet_pd)