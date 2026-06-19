"""Signal/fitting.py
Baseline fitting and delta-F/F computation.

Entry point: `signal_fit`  (previously `doric_fit` on the PhotometryData class).

Supported fitting strategies
-----------------------------
linear / lin
    Ordinary least-squares (polyfit) or robust HuberRegressor.
irls
    Robust linear model using statsmodels RLM with Huber's T loss.
expodecay / exp_decay / exp
    Single-component exponential decay fitted over time.
biexponential / biexp
    Two-component exponential decay fitted over time.
arpls / ar_pls / ar-pls
    Asymmetrically Reweighted Penalised Least Squares (Baek et al. 2015).
"""

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import HuberRegressor

logger = logging.getLogger(__name__)


def _ols_fit(filtered_f0, filtered_f):
    """Fit the active channel to the control channel with ordinary least squares.

    Parameters
    ----------
    filtered_f0 : numpy.ndarray
        Filtered isobestic/control signal.
    filtered_f : numpy.ndarray
        Filtered active signal.

    Returns
    -------
    numpy.ndarray
        Predicted active-channel baseline computed from a first-order
        polynomial fit of ``filtered_f0`` onto ``filtered_f``.
    """
    poly = np.polyfit(filtered_f0, filtered_f, 1)
    return np.multiply(poly[0], filtered_f0) + poly[1]


def _linear_fit(filtered_f0, filtered_f, robust_fit=True, huber_epsilon='auto'):
    """Fit the isobestic channel to the active channel using linear regression.

    Parameters
    ----------
    filtered_f0 : numpy.ndarray
        Filtered isobestic/control signal.
    filtered_f : numpy.ndarray
        Filtered active signal to be modeled.
    robust_fit : bool, optional
        If ``True``, fit with ``HuberRegressor``. If ``False``, use an
        ordinary least-squares line fit.
    huber_epsilon : str or float, optional
        Huber loss threshold. Values ``"auto"`` and ``"mad"`` derive the
        threshold from the residual median absolute deviation. Numeric values
        are used directly and clipped to be greater than ``1.0``.

    Returns
    -------
    numpy.ndarray
        Predicted active-channel baseline for each sample in
        ``filtered_f0``.
    """
    if robust_fit:
        f0_reshaped = filtered_f0.reshape(-1, 1)

        epsilon_str = str(huber_epsilon).strip().lower()
        if epsilon_str in ('auto', 'mad'):
            f_naive_pred = _ols_fit(filtered_f0, filtered_f)
            residuals = filtered_f - f_naive_pred
            mad_val = np.median(np.abs(residuals - np.median(residuals)))
            epsilon_val = max(1.01, 1.4826 * mad_val)
            logger.debug("Huber epsilon (auto/MAD): %.4f", epsilon_val)
        else:
            try:
                epsilon_val = max(1.01, float(huber_epsilon))
            except (ValueError, TypeError):
                epsilon_val = 1.35
            logger.debug("Huber epsilon (user-specified): %.4f", epsilon_val)

        huber = HuberRegressor(epsilon=epsilon_val)
        huber.fit(f0_reshaped, filtered_f)
        fitted = huber.predict(f0_reshaped)
    else:
        fitted = _ols_fit(filtered_f0, filtered_f)

    return fitted


def _irls_fit(filtered_f0, filtered_f):
    """Fit the control channel to the active channel with robust IRLS.

    Parameters
    ----------
    filtered_f0 : numpy.ndarray
        Filtered isobestic/control signal.
    filtered_f : numpy.ndarray
        Filtered active signal to be modeled.

    Returns
    -------
    numpy.ndarray
        Predicted active-channel baseline from a robust linear model fit with
        Huber's T loss.
    """
    design_matrix = sm.add_constant(np.asarray(filtered_f0, dtype=float), has_constant='add')
    response = np.asarray(filtered_f, dtype=float)
    rlm_result = sm.RLM(response, design_matrix, M=sm.robust.norms.HuberT()).fit()
    return rlm_result.predict(design_matrix)


def _exp_decay_fit(filtered_f0, filtered_f, time_data):
    """Fit the active channel using a single-exponential decay over time.

    Parameters
    ----------
    filtered_f0 : numpy.ndarray
        Filtered isobestic/control signal. This is only used for fallback
        linear fitting if the exponential optimization fails.
    filtered_f : numpy.ndarray
        Filtered active signal to be modeled.
    time_data : numpy.ndarray
        Time vector associated with the filtered signals.

    Returns
    -------
    numpy.ndarray
        Predicted baseline from a single-exponential decay model. Falls back
        to an OLS control-to-active fit if nonlinear optimization fails.
    """
    yf = np.asarray(filtered_f, dtype=float)
    t = np.asarray(time_data, dtype=float)
    try:
        p_95 = np.percentile(yf, 95)
        p_05 = np.percentile(yf, 5)
        a0 = p_95 - p_05
        k0 = 1.0 / max((t[-1] - t[0]), 1.0)
        c0 = np.min(yf)
        popt, _ = curve_fit(
            lambda tt, amp, decay, offset: amp * np.exp(-decay * tt) + offset,
            t,
            yf,
            p0=[a0, k0, c0],
            maxfev=10000,
        )
        fitted = popt[0] * np.exp(-popt[1] * t) + popt[2]
    except (RuntimeError, TypeError, ValueError):
        fitted = _ols_fit(filtered_f0, filtered_f)
    return fitted


def _biexp_decay_fit(filtered_f0, filtered_f, time_data):
    """Fit the active channel with a biexponential decay over time.

    Parameters
    ----------
    filtered_f0 : numpy.ndarray
        Filtered isobestic/control signal. Used for fallback linear fitting
        if the biexponential optimization fails.
    filtered_f : numpy.ndarray
        Filtered active signal to be modeled.
    time_data : numpy.ndarray
        Time vector associated with the filtered signals.

    Returns
    -------
    numpy.ndarray
        Predicted baseline from a two-component exponential decay model.
        Falls back to an OLS control-to-active fit if nonlinear optimization
        fails or the inputs are too short for stable fitting.
    """
    yf = np.asarray(filtered_f, dtype=float)
    control = np.asarray(filtered_f0, dtype=float)
    t = np.asarray(time_data, dtype=float)
    if yf.size < 5 or t.size != yf.size:
        return _ols_fit(control, yf)

    t_shifted = t - t[0]
    duration = max(float(t_shifted[-1]), 1.0)
    p10, p50, p90 = np.percentile(yf, [10, 50, 90])
    signal_span = max(float(p90 - p10), np.finfo(float).eps)
    baseline_floor = float(np.min(yf))

    amp_total = max(float(p90 - baseline_floor), np.finfo(float).eps)
    slow_amp = max(float(p90 - p50), 0.25 * amp_total)
    fast_amp = max(float(p50 - p10), 0.15 * amp_total)
    k_slow_0 = 0.5 / duration
    k_fast_0 = 5.0 / duration
    c0 = baseline_floor

    lower_bounds = [0.0, 0.0, 1e-8, 1e-8, baseline_floor - signal_span]
    upper_bounds = [
        4.0 * amp_total,
        4.0 * amp_total,
        10.0 / duration,
        100.0 / duration,
        float(np.max(yf)),
    ]

    def biexponential(tt, amp_slow, amp_fast, k_slow, k_fast, offset):
        return amp_slow * np.exp(-k_slow * tt) + amp_fast * np.exp(-k_fast * tt) + offset

    try:
        popt, _ = curve_fit(
            biexponential,
            t_shifted,
            yf,
            p0=[slow_amp, fast_amp, k_slow_0, k_fast_0, c0],
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000,
        )
        fitted = biexponential(t_shifted, *popt)
    except (RuntimeError, TypeError, ValueError):
        fitted = _ols_fit(control, yf)

    return fitted


def _arpls_drift_fit(
    dff_initial,
    arpls_lambda=1e5,
    arpls_max_iter=50,
    arpls_tol=1e-6,
    arpls_eps=1e-8,
    arpls_weight_scale=2.0,
):
    """Estimate baseline drift using Asymmetrically Reweighted Penalised Least Squares.

    Parameters
    ----------
    dff_initial : numpy.ndarray
        Initial delta-F/F trace from which slow baseline drift should be
        estimated.
    arpls_lambda : float, optional
        Smoothness penalty applied to the second-derivative term.
    arpls_max_iter : int, optional
        Maximum number of reweighting iterations.
    arpls_tol : float, optional
        Relative convergence tolerance for weight updates.
    arpls_eps : float, optional
        Lower bound applied to weights for numerical stability.
    arpls_weight_scale : float, optional
        Scaling factor controlling the sharpness of the logistic reweighting
        transition.

    Returns
    -------
    numpy.ndarray
        Estimated slow drift component with the same shape as
        ``dff_initial``.
    """
    y = dff_initial.astype(float)
    n = y.size
    lam = float(arpls_lambda)
    ratio = float(arpls_tol)
    max_iter = int(arpls_max_iter)
    eps = float(arpls_eps)
    weight_scale = float(arpls_weight_scale)

    e = np.ones(n)
    d_matrix = sparse.diags([e, -2 * e, e], [0, 1, 2], shape=(n - 2, n))
    penalty = lam * (d_matrix.transpose().dot(d_matrix))

    weights = np.ones(n)
    baseline = np.zeros(n)

    for _ in range(max_iter):
        weight_matrix = sparse.diags(weights, 0)
        system = (weight_matrix + penalty).tocsc()
        baseline = spsolve(system, weights * y)

        residuals = y - baseline
        negative_residuals = residuals[residuals < 0]
        if negative_residuals.size == 0:
            break

        mean_neg = negative_residuals.mean()
        std_neg = negative_residuals.std()
        if std_neg <= 0:
            break

        next_weights = 1.0 / (1.0 + np.exp(weight_scale * (residuals - (2.0 * std_neg - mean_neg)) / std_neg))
        next_weights = np.clip(next_weights, eps, 1.0)

        if np.linalg.norm(weights - next_weights) / np.linalg.norm(weights) < ratio:
            weights = next_weights
            break
        weights = next_weights

    return baseline


def signal_fit(
    fit_type,
    filtered_f0,
    filtered_f,
    time_data,
    robust_fit=True,
    baseline_detrend=None,
    arpls_lambda=1e5,
    arpls_max_iter=50,
    arpls_tol=1e-6,
    arpls_eps=1e-8,
    arpls_weight_scale=2.0,
    huber_epsilon='auto',
):
    """Fit a baseline to the photometry signals and compute delta-F/F.

    Parameters
    ----------
    fit_type : str
        Baseline fitting strategy. Supported values include ``"linear"``,
        ``"lin"``, ``"irls"``, ``"expodecay"``, ``"exp_decay"``,
        ``"exp"``, ``"biexponential"``, and ``"biexp"``.
    filtered_f0 : numpy.ndarray
        Filtered isobestic/control signal.
    filtered_f : numpy.ndarray
        Filtered active signal.
    time_data : numpy.ndarray
        Time vector corresponding to the filtered signals.
    robust_fit : bool, optional
        If ``True``, the linear fit strategy uses ``HuberRegressor``.
    baseline_detrend : str or None, optional
        Optional post-fit detrending method. Currently ``"arpls"`` applies
        arPLS drift removal; any other value leaves delta-F/F unchanged.
    arpls_lambda : float, optional
        Smoothness penalty used when ``baseline_detrend`` is ``"arpls"``.
    arpls_max_iter : int, optional
        Maximum number of arPLS iterations.
    arpls_tol : float, optional
        Convergence tolerance for arPLS weight updates.
    arpls_eps : float, optional
        Lower clipping bound for arPLS weights.
    arpls_weight_scale : float, optional
        Logistic weight transition scale for arPLS.
    huber_epsilon : str or float, optional
        Huber threshold control for the linear robust fit path.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``Time`` and ``DeltaF`` containing the input
        time vector and the computed delta-F/F trace.
    """
    fit_type_lower = str(fit_type).lower() if fit_type is not None else 'linear'

    if fit_type_lower in ('linear', 'lin'):
        fitted = _linear_fit(filtered_f0, filtered_f, robust_fit=robust_fit, huber_epsilon=huber_epsilon)
    elif fit_type_lower == 'irls':
        fitted = _irls_fit(filtered_f0, filtered_f)
    elif fit_type_lower in ('expodecay', 'exp_decay', 'exp'):
        fitted = _exp_decay_fit(filtered_f0, filtered_f, time_data)
    elif fit_type_lower in ('biexponential', 'biexp'):
        fitted = _biexp_decay_fit(filtered_f0, filtered_f, time_data)
    else:
        fitted = _ols_fit(filtered_f0, filtered_f)

    with np.errstate(divide='ignore', invalid='ignore'):
        delta_f = (filtered_f - fitted) / fitted
        delta_f = np.nan_to_num(delta_f)

    if baseline_detrend == 'arpls':
        drift_fit = _arpls_drift_fit(
            delta_f,
            arpls_lambda=arpls_lambda,
            arpls_max_iter=arpls_max_iter,
            arpls_tol=arpls_tol,
            arpls_eps=arpls_eps,
            arpls_weight_scale=arpls_weight_scale,
        )
        delta_f -= drift_fit

    result_pd = pd.DataFrame({'Time': time_data, 'DeltaF': delta_f})

    return result_pd
