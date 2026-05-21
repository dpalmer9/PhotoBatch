import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_signal():
    rng = np.random.default_rng(12345)
    time = np.arange(0.0, 100.0, 0.01)
    baseline = 100.0 * np.exp(-0.005 * time)
    control = baseline + rng.normal(0.0, 0.05, len(time))
    active = (
        baseline
        + 2.0 * np.sin(2.0 * np.pi * time)
        + 0.5 * np.sin(2.0 * np.pi * 15.0 * time)
        + rng.normal(0.0, 0.05, len(time))
    )
    return pd.DataFrame({"Time": time, "Control": control, "Active": active})