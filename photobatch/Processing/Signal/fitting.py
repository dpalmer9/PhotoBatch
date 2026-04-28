"""Signal/fitting.py
Baseline fitting and delta-F/F computation.

Entry point: `signal_fit`  (previously `doric_fit` on the PhotometryData class).

Supported fitting strategies
-----------------------------
linear / lin
    Ordinary least-squares (polyfit) or robust HuberRegressor.
expodecay / exp_decay / exp
    Single-component exponential decay fitted over time.
arpls / ar_pls / ar-pls
    Asymmetrically Reweighted Penalised Least Squares (Baek et al. 2015).
"""

import gc
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.optimize import curve_fit
from scipy.sparse.linalg import cg, spilu, LinearOperator
from sklearn.linear_model import HuberRegressor

def _linear_fit(filtered_f0, filtered_f, robust_fit=True, huber_epsilon='auto'):
    if robust_fit:
        f0_reshaped = filtered_f0.reshape(-1, 1)

        epsilon_str = str(huber_epsilon).strip().lower()
        if epsilon_str in ('auto', 'mad'):
            poly_naive = np.polyfit(filtered_f0, filtered_f, 1)
            f_naive_pred = np.multiply(poly_naive[0], filtered_f0) + poly_naive[1]
            residuals  = filtered_f - f_naive_pred
            mad_val    = np.median(np.abs(residuals - np.median(residuals)))
            epsilon_val = max(1.01, 1.4826 * mad_val)
            print(f"Huber epsilon (auto/MAD): {epsilon_val:.4f}")
        else:
            try:
                epsilon_val = max(1.01, float(huber_epsilon))
            except (ValueError, TypeError):
                epsilon_val = 1.35
            print(f"Huber epsilon (user-specified): {epsilon_val:.4f}")

        huber = HuberRegressor(epsilon=epsilon_val)
        huber.fit(f0_reshaped, filtered_f)
        fitted = huber.predict(f0_reshaped)
    else:
        poly  = np.polyfit(filtered_f0, filtered_f, 1)
        fitted = np.multiply(poly[0], filtered_f0) + poly[1]

    return fitted

def _exp_decay_fit(filtered_f0, filtered_f, time_data):
    yf = filtered_f
    f0_aligned = _linear_fit(filtered_f0, filtered_f, robust_fit=True, huber_epsilon='auto')
    t = time_data.astype(float)
    try:
        p_95 = np.percentile(yf, 95)
        p_05 = np.percentile(yf, 5)
        A0 = p_95 - p_05
        k0 = 1.0 / max((t[-1] - t[0]), 1.0)
        C0 = np.min(yf)
        popt, _ = curve_fit(
            lambda tt, A, k, C: A * np.exp(-k * tt) + C,
            t, yf, p0=[A0, k0, C0], maxfev=10000
        )
        fitted = popt[0] * np.exp(-popt[1] * t) + popt[2]
    except (RuntimeError, TypeError, ValueError):
        poly   = np.polyfit(filtered_f0, filtered_f, 1)
        fitted = np.multiply(poly[0], filtered_f0) + poly[1]
    return fitted

def _arpls_drift_fit(dff_initial, arpls_lambda=1e5, arpls_max_iter=50, arpls_tol=1e-6, arpls_eps=1e-8, arpls_weight_scale=2.0):
    y  = dff_initial.astype(float)
    n  = y.size
    lam          = float(arpls_lambda)
    ratio        = float(arpls_tol)
    max_iter     = int(arpls_max_iter)
    eps          = float(arpls_eps)
    weight_scale = float(arpls_weight_scale)

    e = np.ones(n)
    D = sparse.diags([e, -2*e, e], [0, 1, 2], shape=(n-2, n))
    H = lam * (D.transpose().dot(D))

    w = np.ones(n)
    z = np.zeros(n)

    for i in range(max_iter):
        W = sparse.diags(w, 0)
        C = (W + H).tocsc()
        try:
            ilu = spilu(C, fill_factor=2)
            M   = LinearOperator(C.shape, ilu.solve)
        except RuntimeError:
            M = None
        z, info = cg(C, w * y, x0=z, M=M, atol=1e-8)
        if info != 0:
            print(f"arPLS CG solver did not converge at iteration {i} "
                  f"(info={info})")

        d     = y - z
        d_neg = d[d < 0]
        if d_neg.size == 0:
            break
        m = d_neg.mean()
        s = d_neg.std()
        if s <= 0:
            break

        wt = 1.0 / (1.0 + np.exp(weight_scale * (d - (2.0*s - m)) / s))
        wt = np.clip(wt, eps, 1.0)

        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            w = wt
            break
        w = wt

    return z

def signal_fit(fit_type, filtered_f0, filtered_f, time_data,
               robust_fit=True, baseline_detrend=None,
               arpls_lambda=1e5, arpls_max_iter=50, arpls_tol=1e-6,
               arpls_eps=1e-8, arpls_weight_scale=2.0,
               huber_epsilon='auto'):
    """Fit a baseline to the photometry signals and compute delta-F/F.

    Parameters
    ----------
    fit_type : str
        Fitting strategy: 'linear', 'expodecay', or 'arpls'.
    baseline_detrend : str or None
        Baseline detrending method: 'expodecay', 'arpls', or None.
    filtered_f0 : np.ndarray
        Filtered isobestic (Control) channel.
    filtered_f : np.ndarray
        Filtered active channel.
    time_data : np.ndarray
        Time axis corresponding to the filtered signals.
    robust_fit : bool
        Use HuberRegressor instead of polyfit for the linear strategy.
    arpls_lambda : float
        Smoothness penalty for arPLS (default 1e5).
    arpls_max_iter : int
        Maximum number of arPLS iterations (default 50).
    arpls_tol : float
        Convergence tolerance for arPLS weight update (default 1e-6).
    arpls_eps : float
        Weight clipping floor for arPLS (default 1e-8).
    arpls_weight_scale : float
        Sharpness of the logistic weight transition in arPLS (default 2.0).
    huber_epsilon : str or float
        'auto' / 'mad' – derive epsilon from the MAD of the noise floor.
        A numeric value – use directly (must be > 1.0).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Time', 'DeltaF'].
    """
    fit_type_lower = str(fit_type).lower() if fit_type is not None else 'linear'

    # ------------------------------------------------------------------
    # 1. Linear fit (OLS or robust Huber)
    # ------------------------------------------------------------------
    if fit_type_lower in ('linear', 'lin'):
        fitted = _linear_fit(filtered_f0, filtered_f, robust_fit=robust_fit, huber_epsilon=huber_epsilon)

    elif fit_type_lower in ('expodecay', 'exp_decay', 'exp'):
        fitted = _exp_decay_fit(filtered_f0, filtered_f, time_data)
    # ------------------------------------------------------------------
    # Fallback: unknown fit type → linear OLS
    # ------------------------------------------------------------------
    else:
        poly   = np.polyfit(filtered_f0, filtered_f, 1)
        fitted = np.multiply(poly[0], filtered_f0) + poly[1]

    # ------------------------------------------------------------------
    # Compute delta-F/F  ( (F - F_fit) / F_fit )
    # ------------------------------------------------------------------
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_f = (filtered_f - fitted) / fitted
        delta_f = np.nan_to_num(delta_f)

    # ------------------------------------------------------------------
    # 2. Conduct expodecay or arpls fitting for baseline drift if requested
    # ------------------------------------------------------------------
    if baseline_detrend == 'arpls':
        drift_fit = _arpls_drift_fit(delta_f, arpls_lambda=arpls_lambda, arpls_max_iter=arpls_max_iter, arpls_tol=arpls_tol, arpls_eps=arpls_eps, arpls_weight_scale=arpls_weight_scale)
        delta_f -= drift_fit

    result_pd = pd.DataFrame({'Time': time_data, 'DeltaF': delta_f})

    gc.collect()
    return result_pd
