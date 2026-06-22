"""Advanced statistical analyses for photometry sessions.

The implementations in this module are intentionally self-contained and use
only NumPy, SciPy, pandas, and Statsmodels. They are designed for the current
photometry pipeline rather than as generic, library-style HMM toolkits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.special import expit


logger = logging.getLogger(__name__)


def _coerce_trial_matrix(plot_data: pd.DataFrame | np.ndarray | list[list[float]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert peri-event trial data into a dense trials-by-time matrix."""
    if isinstance(plot_data, pd.DataFrame):
        numeric_plot = plot_data.apply(pd.to_numeric, errors='coerce')
        time_grid = pd.to_numeric(pd.Index(plot_data.index), errors='coerce').to_numpy(dtype=float, copy=False)
        if np.isnan(time_grid).any():
            time_grid = np.arange(len(plot_data), dtype=float)
        trial_names = [str(column) for column in numeric_plot.columns]
        matrix = numeric_plot.to_numpy(dtype=float, copy=True).T
    else:
        matrix = np.asarray(plot_data, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[np.newaxis, :]
        elif matrix.ndim != 2:
            raise ValueError("plot_data must be a 1-D or 2-D array-like object")
        time_grid = np.arange(matrix.shape[1], dtype=float)
        trial_names = [f"Trial {index + 1}" for index in range(matrix.shape[0])]

    if matrix.size == 0:
        return np.empty((0, 0), dtype=float), time_grid, trial_names

    filled = np.empty_like(matrix, dtype=float)
    for index, row in enumerate(matrix):
        series = pd.Series(row, dtype=float)
        if series.isna().all():
            filled[index] = np.zeros(series.shape[0], dtype=float)
        else:
            filled[index] = (
                series.interpolate(limit_direction='both')
                .ffill()
                .bfill()
                .to_numpy(dtype=float)
            )
    return filled, time_grid, trial_names


def _expand_trial_metadata(metadata: Any, n_trials: int, prefix: str) -> pd.DataFrame:
    """Broadcast scalar or trial-wise metadata into a trial-level DataFrame."""
    if metadata is None:
        return pd.DataFrame(index=np.arange(n_trials))

    if isinstance(metadata, pd.DataFrame):
        frame = metadata.copy()
    elif isinstance(metadata, pd.Series):
        frame = metadata.to_frame(name=metadata.name or prefix)
    elif isinstance(metadata, dict):
        columns: dict[str, list[Any]] = {}
        for key, value in metadata.items():
            if isinstance(value, (pd.Series, np.ndarray, list, tuple)) and len(value) == n_trials:
                columns[str(key)] = list(value)
            else:
                columns[str(key)] = [value] * n_trials
        frame = pd.DataFrame(columns)
    else:
        frame = pd.DataFrame({prefix: [metadata] * n_trials})

    if frame.empty:
        return pd.DataFrame(index=np.arange(n_trials))

    if len(frame) == 1 and n_trials > 1:
        frame = pd.concat([frame] * n_trials, ignore_index=True)
    elif len(frame) != n_trials:
        raise ValueError(f"{prefix} metadata must have exactly one row or one row per trial")

    return frame.reset_index(drop=True)


def _prepare_event_frame(events: Any) -> pd.DataFrame:
    """Normalize event annotations into a standard DataFrame."""
    if events is None:
        return pd.DataFrame(columns=['time', 'event_type'])

    if isinstance(events, pd.DataFrame):
        event_frame = events.copy()
    elif isinstance(events, dict):
        event_frame = pd.DataFrame(events)
    else:
        event_frame = pd.DataFrame(events)

    if event_frame.empty:
        return pd.DataFrame(columns=['time', 'event_type'])

    rename_map = {}
    for source, target in (
        ('Start_Time', 'time'),
        ('start_time', 'time'),
        ('Time', 'time'),
        ('timestamp', 'time'),
        ('event_alias', 'event_type'),
        ('behavior', 'event_type'),
        ('event_name', 'event_type'),
        ('label', 'event_type'),
    ):
        if source in event_frame.columns and target not in event_frame.columns:
            rename_map[source] = target
    event_frame = event_frame.rename(columns=rename_map)

    if 'time' not in event_frame.columns:
        raise ValueError("events must contain a time column")
    if 'event_type' not in event_frame.columns:
        event_frame['event_type'] = 'event'

    event_frame['time'] = pd.to_numeric(event_frame['time'], errors='coerce')
    event_frame['event_type'] = event_frame['event_type'].astype(str)
    return event_frame.dropna(subset=['time']).sort_values('time').reset_index(drop=True)


def _build_event_design_matrix(session_time: np.ndarray, events: Any) -> tuple[np.ndarray, list[str]]:
    """Construct convolved event covariates for session-level analyses."""
    event_frame = _prepare_event_frame(events)
    time = np.asarray(session_time, dtype=float)
    if time.size == 0 or event_frame.empty:
        return np.empty((len(time), 0), dtype=float), []

    positive_diffs = np.diff(time)
    dt = float(np.median(positive_diffs[positive_diffs > 0])) if np.any(positive_diffs > 0) else 1.0
    kernel_window = max(5, int(np.ceil(5.0 / max(dt, 1e-6))))
    kernel_time = np.arange(kernel_window, dtype=float) * dt
    tau = 1.0  # Fixed physical GCaMP sensor decay time constant in seconds
    kernel = np.exp(-kernel_time / tau)
    kernel /= max(kernel.sum(), np.finfo(float).eps)

    columns: list[np.ndarray] = []
    names: list[str] = []
    for event_name, group in event_frame.groupby('event_type', sort=True):
        impulse = np.zeros_like(time, dtype=float)
        event_indices = np.searchsorted(time, group['time'].to_numpy(dtype=float), side='left')
        event_indices = event_indices[(event_indices >= 0) & (event_indices < len(time))]
        if event_indices.size == 0:
            continue
        np.add.at(impulse, event_indices, 1.0)
        convolved = np.convolve(impulse, kernel, mode='full')[:len(time)]
        columns.append(convolved)
        names.append(str(event_name))

    if not columns:
        return np.empty((len(time), 0), dtype=float), []
    return np.column_stack(columns), names


def _smooth_vector(values: np.ndarray) -> np.ndarray:
    """Apply a light functional smoothing pass to a 1-D trajectory."""
    values = np.asarray(values, dtype=float)
    if values.size < 5:
        return values.copy()

    series = pd.Series(values, dtype=float).interpolate(limit_direction='both').ffill().bfill()
    filled = series.to_numpy(dtype=float)
    preferred_window = 9
    window = min(preferred_window, values.size if values.size % 2 else values.size - 1)
    if window < 5:
        window = 5 if values.size >= 5 else values.size
    if window % 2 == 0:
        window -= 1
    if window >= 5:
        smoothed = savgol_filter(filled, window_length=window, polyorder=min(3, window - 2), mode='interp')
    else:
        smoothed = filled
    sigma = max(window / 6.0, 1.0)
    return gaussian_filter1d(smoothed, sigma=sigma, mode='nearest')


def _smooth_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Smooth every numeric column in a DataFrame."""
    if frame.empty:
        return frame.copy()
    smoothed = frame.copy()
    for column in smoothed.columns:
        numeric = pd.to_numeric(smoothed[column], errors='coerce')
        if numeric.notna().any():
            smoothed[column] = _smooth_vector(numeric.to_numpy(dtype=float))
    return smoothed


def _encode_feature_frame(frame: pd.DataFrame, intercept: bool = True) -> pd.DataFrame:
    """Convert mixed metadata into a numeric feature matrix."""
    encoded = pd.DataFrame(index=frame.index)
    if intercept:
        encoded['intercept'] = 1.0
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors='coerce')
        if numeric.isna().all():
            categories = pd.Categorical(frame[column].astype(str))
            encoded[column] = categories.codes.astype(float)
        else:
            mean_value = float(numeric.mean()) if numeric.notna().any() else 0.0
            encoded[column] = numeric.fillna(mean_value).astype(float)
    return encoded


def _build_group_labels(random_frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Build grouping labels and numeric random-effect covariates."""
    labels = random_frame.astype(str).agg('|'.join, axis=1).to_numpy(dtype=object)
    encoded = _encode_feature_frame(random_frame, intercept=False) if random_frame.shape[1] > 1 else pd.DataFrame(index=random_frame.index)
    return labels, encoded


def _weighted_least_squares(design_matrix: np.ndarray, response: np.ndarray, weights: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Solve a weighted least-squares system with light ridge stabilization."""
    x = np.asarray(design_matrix, dtype=float)
    y = np.asarray(response, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-8, None)
    weighted_x = x * w[:, None]
    lhs = x.T @ weighted_x + ridge * np.eye(x.shape[1], dtype=float)
    rhs = x.T @ (w * y)
    return np.linalg.solve(lhs, rhs)


def _safe_normalize_rows(matrix: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Normalize a 2-D matrix row-wise with a stability floor."""
    normalized = np.asarray(matrix, dtype=float).copy()
    row_sums = normalized.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= floor, 1.0, row_sums)
    normalized /= row_sums
    return normalized


@dataclass
class _ScaledForwardBackwardResult:
    gamma: np.ndarray
    xi: np.ndarray
    log_likelihood: float


class _BaseHMM:
    """Shared scaled forward-backward and Viterbi utilities."""

    def __init__(self, n_states: int, max_iter: int = 50, tol: float = 1e-4, random_state: int = 0):
        self.n_states = int(max(1, n_states))
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = np.random.default_rng(random_state)
        self.initial_state_probs_ = np.full(self.n_states, 1.0 / self.n_states, dtype=float)
        stay_prob = 0.90 if self.n_states > 1 else 1.0
        off_diag = (1.0 - stay_prob) / max(self.n_states - 1, 1)
        self.transition_matrix_ = np.full((self.n_states, self.n_states), off_diag, dtype=float)
        np.fill_diagonal(self.transition_matrix_, stay_prob)
        self.log_likelihood_history_: list[float] = []
        self.n_iter_ = 0

    def _forward_backward(self, log_emissions: np.ndarray) -> _ScaledForwardBackwardResult:
        """Run scaled forward-backward on log emission densities."""
        log_emissions = np.asarray(log_emissions, dtype=float)
        n_samples, n_states = log_emissions.shape
        row_offsets = np.max(log_emissions, axis=1)
        emission_density = np.exp(log_emissions - row_offsets[:, None])
        emission_density = np.clip(emission_density, 1e-300, None)

        alpha = np.zeros((n_samples, n_states), dtype=float)
        beta = np.zeros((n_samples, n_states), dtype=float)
        scales = np.zeros(n_samples, dtype=float)

        alpha[0] = self.initial_state_probs_ * emission_density[0]
        scales[0] = max(alpha[0].sum(), 1e-300)
        alpha[0] /= scales[0]

        for t in range(1, n_samples):
            alpha[t] = (alpha[t - 1] @ self.transition_matrix_) * emission_density[t]
            scales[t] = max(alpha[t].sum(), 1e-300)
            alpha[t] /= scales[t]

        beta[-1] = 1.0
        for t in range(n_samples - 2, -1, -1):
            beta[t] = self.transition_matrix_ @ (emission_density[t + 1] * beta[t + 1])
            beta[t] /= max(scales[t + 1], 1e-300)

        gamma = _safe_normalize_rows(alpha * beta)

        xi = np.zeros((max(n_samples - 1, 0), n_states, n_states), dtype=float)
        for t in range(n_samples - 1):
            xi_t = alpha[t][:, None] * self.transition_matrix_ * (emission_density[t + 1] * beta[t + 1])[None, :]
            xi_sum = max(xi_t.sum(), 1e-300)
            xi[t] = xi_t / xi_sum

        log_likelihood = float(np.sum(np.log(scales) + row_offsets))
        return _ScaledForwardBackwardResult(gamma=gamma, xi=xi, log_likelihood=log_likelihood)

    def _viterbi(self, log_emissions: np.ndarray) -> np.ndarray:
        """Decode the most likely state sequence using Viterbi."""
        log_emissions = np.asarray(log_emissions, dtype=float)
        n_samples, n_states = log_emissions.shape
        log_transition = np.log(np.clip(self.transition_matrix_, 1e-300, None))
        log_initial = np.log(np.clip(self.initial_state_probs_, 1e-300, None))

        delta = np.zeros((n_samples, n_states), dtype=float)
        psi = np.zeros((n_samples, n_states), dtype=int)
        delta[0] = log_initial + log_emissions[0]

        for t in range(1, n_samples):
            scores = delta[t - 1][:, None] + log_transition
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_emissions[t]

        states = np.zeros(n_samples, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(n_samples - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states


class GaussianGLMHMM(_BaseHMM):
    """Gaussian GLM-HMM fitted with EM and weighted least squares."""

    def __init__(self, n_states: int = 3, max_iter: int = 50, tol: float = 1e-4, random_state: int = 0):
        super().__init__(n_states=n_states, max_iter=max_iter, tol=tol, random_state=random_state)
        self.weights_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None

    def _initialize_emissions(self, design_matrix: np.ndarray, response: np.ndarray) -> None:
        n_features = design_matrix.shape[1]
        global_weights, *_ = np.linalg.lstsq(design_matrix, response, rcond=None)
        quantile_edges = np.quantile(response, np.linspace(0.0, 1.0, self.n_states + 1))
        self.weights_ = np.tile(global_weights, (self.n_states, 1))
        self.variances_ = np.full(self.n_states, max(float(np.var(response)), 1e-3), dtype=float)

        for state in range(self.n_states):
            if state == self.n_states - 1:
                mask = response >= quantile_edges[state]
            else:
                mask = (response >= quantile_edges[state]) & (response < quantile_edges[state + 1])
            if np.count_nonzero(mask) >= design_matrix.shape[1]:
                self.weights_[state] = np.linalg.lstsq(design_matrix[mask], response[mask], rcond=None)[0]
                residuals = response[mask] - design_matrix[mask] @ self.weights_[state]
                self.variances_[state] = max(float(np.mean(residuals ** 2)), 1e-4)
            else:
                self.weights_[state] = global_weights + 0.05 * self.random_state.normal(size=n_features)

    def _compute_log_emissions(self, design_matrix: np.ndarray, response: np.ndarray) -> np.ndarray:
        weights = np.asarray(self.weights_, dtype=float)
        variances = np.clip(np.asarray(self.variances_, dtype=float), 1e-6, None)
        means = design_matrix @ weights.T
        residuals = response[:, None] - means
        return -0.5 * (np.log(2.0 * np.pi * variances)[None, :] + (residuals ** 2) / variances[None, :])

    def fit(self, design_matrix: np.ndarray, response: np.ndarray) -> 'GaussianGLMHMM':
        """Fit the model parameters with EM."""
        x = np.asarray(design_matrix, dtype=float)
        y = np.asarray(response, dtype=float)
        self._initialize_emissions(x, y)

        previous_log_likelihood = -np.inf
        for iteration in range(self.max_iter):
            posterior = self._forward_backward(self._compute_log_emissions(x, y))
            gamma = posterior.gamma
            xi = posterior.xi
            self.log_likelihood_history_.append(posterior.log_likelihood)

            self.initial_state_probs_ = gamma[0]
            if xi.size:
                numerator = xi.sum(axis=0)
                denominator = gamma[:-1].sum(axis=0, keepdims=True).T
                denominator = np.where(denominator <= 1e-12, 1.0, denominator)
                self.transition_matrix_ = _safe_normalize_rows(numerator / denominator)

            for state in range(self.n_states):
                weights = gamma[:, state]
                self.weights_[state] = _weighted_least_squares(x, y, weights)
                residuals = y - x @ self.weights_[state]
                variance = float(np.sum(weights * residuals ** 2) / max(np.sum(weights), 1e-8))
                self.variances_[state] = max(variance, 1e-5)

            self.n_iter_ = iteration + 1
            if abs(posterior.log_likelihood - previous_log_likelihood) < self.tol:
                break
            previous_log_likelihood = posterior.log_likelihood
        return self

    def predict_states(self, design_matrix: np.ndarray, response: np.ndarray) -> np.ndarray:
        """Return the most likely state sequence after fitting."""
        return self._viterbi(self._compute_log_emissions(np.asarray(design_matrix, dtype=float), np.asarray(response, dtype=float)))


class MoAHMM(_BaseHMM):
    """Mixture-of-agents HMM with continuous or binary emissions."""

    def __init__(self, n_states: int = 3, emission_mode: str = 'continuous', max_iter: int = 50, tol: float = 1e-4, random_state: int = 0):
        super().__init__(n_states=n_states, max_iter=max_iter, tol=tol, random_state=random_state)
        self.emission_mode = emission_mode
        self.agent_weights_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None

    def _initialize_emissions(self, agent_matrix: np.ndarray, response: np.ndarray) -> None:
        n_agents = agent_matrix.shape[1]
        self.agent_weights_ = 0.05 * self.random_state.normal(size=(self.n_states, n_agents))
        self.bias_ = np.linspace(-0.2, 0.2, self.n_states, dtype=float)
        self.variances_ = np.full(self.n_states, max(float(np.var(response)), 1e-3), dtype=float)

    def _linear_predictor(self, agent_matrix: np.ndarray) -> np.ndarray:
        return agent_matrix @ self.agent_weights_.T + self.bias_[None, :]

    def _compute_log_emissions(self, agent_matrix: np.ndarray, response: np.ndarray) -> np.ndarray:
        linear_predictor = self._linear_predictor(agent_matrix)
        if self.emission_mode == 'binary':
            probabilities = np.clip(expit(linear_predictor), 1e-8, 1.0 - 1e-8)
            return response[:, None] * np.log(probabilities) + (1.0 - response[:, None]) * np.log(1.0 - probabilities)

        variances = np.clip(np.asarray(self.variances_, dtype=float), 1e-6, None)
        residuals = response[:, None] - linear_predictor
        return -0.5 * (np.log(2.0 * np.pi * variances)[None, :] + (residuals ** 2) / variances[None, :])

    def _fit_weighted_logistic(self, agent_matrix: np.ndarray, response: np.ndarray, weights: np.ndarray, initial_params: np.ndarray) -> np.ndarray:
        """Fit one weighted logistic expert with L-BFGS-B."""
        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            linear = agent_matrix @ params[:-1] + params[-1]
            probabilities = np.clip(expit(linear), 1e-8, 1.0 - 1e-8)
            loss = -np.sum(weights * (response * np.log(probabilities) + (1.0 - response) * np.log(1.0 - probabilities)))
            residual = probabilities - response
            grad_w = agent_matrix.T @ (weights * residual)
            grad_b = np.sum(weights * residual)
            return float(loss), np.concatenate([grad_w, np.asarray([grad_b], dtype=float)])

        result = minimize(
            fun=lambda params: objective(params)[0],
            x0=initial_params,
            jac=lambda params: objective(params)[1],
            method='L-BFGS-B',
        )
        if not result.success:
            logger.debug("Weighted logistic MoA fit did not fully converge: %s", result.message)
        return np.asarray(result.x, dtype=float)

    def fit(self, agent_matrix: np.ndarray, response: np.ndarray) -> 'MoAHMM':
        """Fit the MoA-HMM with EM."""
        q = np.asarray(agent_matrix, dtype=float)
        y = np.asarray(response, dtype=float)
        self._initialize_emissions(q, y)

        previous_log_likelihood = -np.inf
        for iteration in range(self.max_iter):
            posterior = self._forward_backward(self._compute_log_emissions(q, y))
            gamma = posterior.gamma
            xi = posterior.xi
            self.log_likelihood_history_.append(posterior.log_likelihood)

            self.initial_state_probs_ = gamma[0]
            if xi.size:
                numerator = xi.sum(axis=0)
                denominator = gamma[:-1].sum(axis=0, keepdims=True).T
                denominator = np.where(denominator <= 1e-12, 1.0, denominator)
                self.transition_matrix_ = _safe_normalize_rows(numerator / denominator)

            for state in range(self.n_states):
                state_weights = np.clip(gamma[:, state], 1e-8, None)
                if self.emission_mode == 'binary':
                    initial = np.concatenate([self.agent_weights_[state], np.asarray([self.bias_[state]], dtype=float)])
                    fitted = self._fit_weighted_logistic(q, y, state_weights, initial)
                    self.agent_weights_[state] = fitted[:-1]
                    self.bias_[state] = fitted[-1]
                else:
                    augmented = np.column_stack([q, np.ones(len(q), dtype=float)])
                    fitted = _weighted_least_squares(augmented, y, state_weights)
                    self.agent_weights_[state] = fitted[:-1]
                    self.bias_[state] = fitted[-1]
                    residuals = y - (q @ self.agent_weights_[state] + self.bias_[state])
                    variance = float(np.sum(state_weights * residuals ** 2) / max(np.sum(state_weights), 1e-8))
                    self.variances_[state] = max(variance, 1e-5)

            self.n_iter_ = iteration + 1
            if abs(posterior.log_likelihood - previous_log_likelihood) < self.tol:
                break
            previous_log_likelihood = posterior.log_likelihood
        return self

    def predict_states(self, agent_matrix: np.ndarray, response: np.ndarray) -> np.ndarray:
        """Return the most likely latent-state sequence after fitting."""
        return self._viterbi(self._compute_log_emissions(np.asarray(agent_matrix, dtype=float), np.asarray(response, dtype=float)))


def run_flmm_analysis(plot_data, covariates=None, random_effects=None):
    """Run pointwise mixed-effects fits and smooth the resulting trajectories."""
    trial_matrix, time_grid, trial_names = _coerce_trial_matrix(plot_data)
    n_trials, n_timepoints = trial_matrix.shape if trial_matrix.size else (0, 0)
    if n_trials == 0 or n_timepoints == 0:
        return {
            'model_type': 'flmm',
            'status': 'empty',
            'grid_points': [],
            'trial_names': trial_names,
            'coefficient_curves': pd.DataFrame(),
            'covariance_parameters': pd.DataFrame(),
            'design_matrices': {
                'fixed_effects': pd.DataFrame(),
                'random_effects': pd.DataFrame(),
            },
            'fit_statistics': {},
        }

    fixed_frame = _expand_trial_metadata(covariates, n_trials, 'covariate')
    if 'time' in fixed_frame.columns:
        time_candidate = pd.to_numeric(pd.Series(fixed_frame.pop('time')), errors='coerce').to_numpy(dtype=float, copy=False)
        if time_candidate.size == n_timepoints and not np.isnan(time_candidate).any():
            time_grid = time_candidate
    fixed_effects = _encode_feature_frame(fixed_frame, intercept=True) if not fixed_frame.empty else pd.DataFrame({'intercept': np.ones(n_trials, dtype=float)})

    random_frame = _expand_trial_metadata(random_effects, n_trials, 'group')
    if random_frame.empty:
        random_frame = pd.DataFrame({'group': np.arange(n_trials, dtype=int)})
    group_labels, random_design = _build_group_labels(random_frame)

    coefficient_curves: dict[str, list[float]] = {column: [] for column in fixed_effects.columns}
    fitted_matrix = np.full_like(trial_matrix, np.nan, dtype=float)
    covariance_records: list[dict[str, float]] = []
    aic_values: list[float] = []
    bic_values: list[float] = []
    mixedlm_steps = 0

    for time_index, grid_value in enumerate(time_grid):
        response = trial_matrix[:, time_index]
        valid_mask = np.isfinite(response)
        design = fixed_effects.loc[valid_mask].copy()
        valid_groups = group_labels[valid_mask]
        valid_random_design = random_design.loc[valid_mask].copy() if not random_design.empty else pd.DataFrame(index=design.index)
        response_valid = response[valid_mask]

        if response_valid.size < max(3, design.shape[1]):
            for column in fixed_effects.columns:
                coefficient_curves[column].append(np.nan)
            covariance_records.append({'time': float(grid_value), 'residual_variance': np.nan, 'random_effect_variance': np.nan})
            continue

        fit_result = None
        random_variance = np.nan
        residual_variance = np.nan
        params = pd.Series(np.nan, index=fixed_effects.columns, dtype=float)

        if np.unique(valid_groups).size > 1:
            try:
                mixed_model = sm.MixedLM(response_valid, design, groups=valid_groups, exog_re=valid_random_design if not valid_random_design.empty else None)
                fit_result = mixed_model.fit(reml=False, disp=False)
                params = fit_result.fe_params.reindex(fixed_effects.columns, fill_value=np.nan)
                residual_variance = float(getattr(fit_result, 'scale', np.nan))
                cov_re = np.asarray(getattr(fit_result, 'cov_re', np.asarray([])), dtype=float)
                if cov_re.size:
                    random_variance = float(np.nanmean(np.diag(cov_re)))
                if np.isfinite(getattr(fit_result, 'aic', np.nan)):
                    aic_values.append(float(fit_result.aic))
                if np.isfinite(getattr(fit_result, 'bic', np.nan)):
                    bic_values.append(float(fit_result.bic))
                mixedlm_steps += 1
            except Exception as exc:
                logger.debug("FLMM MixedLM fit failed at time %.5f; using OLS fallback: %s", grid_value, exc)

        if fit_result is None:
            ols_result = sm.OLS(response_valid, design).fit()
            params = ols_result.params.reindex(fixed_effects.columns, fill_value=np.nan)
            residual_variance = float(getattr(ols_result, 'mse_resid', np.nan))
            if np.isfinite(getattr(ols_result, 'aic', np.nan)):
                aic_values.append(float(ols_result.aic))
            if np.isfinite(getattr(ols_result, 'bic', np.nan)):
                bic_values.append(float(ols_result.bic))

        fitted_matrix[:, time_index] = fixed_effects.to_numpy(dtype=float) @ params.to_numpy(dtype=float)
        for column in fixed_effects.columns:
            coefficient_curves[column].append(float(params.get(column, np.nan)))
        covariance_records.append({'time': float(grid_value), 'residual_variance': residual_variance, 'random_effect_variance': random_variance})

    raw_coefficient_df = pd.DataFrame(coefficient_curves, index=pd.Index(time_grid, name='time'))
    coefficient_df = _smooth_dataframe(raw_coefficient_df)
    raw_covariance_df = pd.DataFrame(covariance_records)
    covariance_df = _smooth_dataframe(raw_covariance_df[['residual_variance', 'random_effect_variance']]) if not raw_covariance_df.empty else pd.DataFrame()
    if not covariance_df.empty:
        covariance_df.insert(0, 'time', raw_covariance_df['time'].to_numpy(dtype=float))

    return {
        'model_type': 'flmm',
        'status': 'ok',
        'grid_points': time_grid.tolist(),
        'trial_names': trial_names,
        'coefficient_curves': coefficient_df,
        'raw_coefficient_curves': raw_coefficient_df,
        'covariance_parameters': covariance_df,
        'raw_covariance_parameters': raw_covariance_df,
        'design_matrices': {
            'fixed_effects': fixed_effects,
            'random_effects': random_frame,
        },
        'fitted_values': pd.DataFrame(fitted_matrix.T, index=pd.Index(time_grid, name='time'), columns=trial_names),
        'fit_statistics': {
            'mean_aic': float(np.nanmean(aic_values)) if aic_values else np.nan,
            'mean_bic': float(np.nanmean(bic_values)) if bic_values else np.nan,
            'successful_mixedlm_steps': int(mixedlm_steps),
            'total_timepoints': int(len(time_grid)),
        },
    }


def run_glm_hmm_analysis(session_time, session_signal, events, n_states=3):
    """Fit a native Gaussian GLM-HMM to a continuous session trace."""
    time = np.asarray(session_time, dtype=float)
    signal = np.asarray(session_signal, dtype=float)
    if time.ndim != 1 or signal.ndim != 1 or time.size != signal.size:
        raise ValueError("session_time and session_signal must be one-dimensional arrays of equal length")
    if time.size == 0:
        return {
            'model_type': 'glm_hmm',
            'status': 'empty',
            'transition_matrix': np.empty((0, 0), dtype=float),
            'glm_weights': pd.DataFrame(),
            'viterbi_path': np.empty((0,), dtype=int),
            'design_matrix': pd.DataFrame(),
            'emission_parameters': pd.DataFrame(),
        }

    signal_series = pd.Series(signal, dtype=float).interpolate(limit_direction='both').ffill().bfill()
    signal_values = signal_series.to_numpy(dtype=float)
    design_matrix, event_names = _build_event_design_matrix(time, events)
    feature_frame = pd.DataFrame(index=np.arange(len(time)))
    if design_matrix.size:
        for column_index, event_name in enumerate(event_names):
            feature_frame[str(event_name)] = design_matrix[:, column_index]
    if len(time) > 1:
        feature_frame['time_drift'] = (time - time[0]) / (time[-1] - time[0])
    else:
        feature_frame['time_drift'] = 0.0
    encoded_features = _encode_feature_frame(feature_frame, intercept=True)

    model = GaussianGLMHMM(n_states=int(n_states))
    model.fit(encoded_features.to_numpy(dtype=float), signal_values)
    viterbi_path = model.predict_states(encoded_features.to_numpy(dtype=float), signal_values)

    glm_weights = pd.DataFrame(model.weights_, columns=encoded_features.columns)
    glm_weights.insert(0, 'state', np.arange(model.n_states, dtype=int))
    emission_parameters = pd.DataFrame({
        'state': np.arange(model.n_states, dtype=int),
        'variance': np.asarray(model.variances_, dtype=float),
    })

    return {
        'model_type': 'glm_hmm',
        'status': 'ok',
        'transition_matrix': np.asarray(model.transition_matrix_, dtype=float),
        'glm_weights': glm_weights,
        'viterbi_path': viterbi_path,
        'event_names': event_names,
        'design_matrix': pd.DataFrame(encoded_features, index=pd.Index(time, name='time')),
        'emission_parameters': emission_parameters,
        'fit_statistics': {
            'log_likelihood': float(model.log_likelihood_history_[-1]) if model.log_likelihood_history_ else np.nan,
            'n_iter': int(model.n_iter_),
        },
    }


def _build_default_agent_predictions(session_time, session_signal, events) -> pd.DataFrame:
    """Build a tailored set of continuous expert signals for MoA-HMM."""
    time = np.asarray(session_time, dtype=float)
    design_matrix, event_names = _build_event_design_matrix(time, events)
    agent_frame = pd.DataFrame(index=np.arange(len(time)))
    if design_matrix.size:
        for column_index, event_name in enumerate(event_names):
            agent_frame[f'event_{event_name}'] = design_matrix[:, column_index]
        agent_frame['event_drive'] = design_matrix.sum(axis=1)
    else:
        agent_frame['event_drive'] = np.zeros(len(time), dtype=float)

    return agent_frame


def run_moa_hmm_analysis(session_time, session_signal, agent_predictions=None, events=None, n_states=3):
    """Fit a native mixture-of-agents HMM to continuous or binary outputs."""
    time = np.asarray(session_time, dtype=float)
    signal = np.asarray(session_signal, dtype=float)
    if time.ndim != 1 or signal.ndim != 1 or time.size != signal.size:
        raise ValueError("session_time and session_signal must be one-dimensional arrays of equal length")
    if time.size == 0:
        return {
            'model_type': 'moa_hmm',
            'status': 'empty',
            'transition_matrix': np.empty((0, 0), dtype=float),
            'agent_weights': pd.DataFrame(),
            'viterbi_path': np.empty((0,), dtype=int),
            'agent_predictions': pd.DataFrame(),
            'emission_parameters': pd.DataFrame(),
        }

    if agent_predictions is None:
        agent_frame = _build_default_agent_predictions(time, signal, events)
    elif isinstance(agent_predictions, pd.DataFrame):
        agent_frame = agent_predictions.copy()
    elif isinstance(agent_predictions, dict):
        agent_frame = pd.DataFrame(agent_predictions)
    else:
        agent_array = np.asarray(agent_predictions, dtype=float)
        if agent_array.ndim == 1:
            agent_array = agent_array[:, None]
        agent_frame = pd.DataFrame(agent_array, columns=[f'agent_{index + 1}' for index in range(agent_array.shape[1])])

    encoded_agents = _encode_feature_frame(agent_frame, intercept=False)
    is_binary = np.all(np.isfinite(signal)) and np.all(np.isin(np.unique(signal), [0.0, 1.0]))
    emission_mode = 'binary' if is_binary else 'continuous'

    model = MoAHMM(n_states=int(n_states), emission_mode=emission_mode)
    model.fit(encoded_agents.to_numpy(dtype=float), signal.astype(float))
    viterbi_path = model.predict_states(encoded_agents.to_numpy(dtype=float), signal.astype(float))

    weight_columns = list(encoded_agents.columns)
    agent_weights = pd.DataFrame(model.agent_weights_, columns=weight_columns)
    agent_weights.insert(0, 'bias', np.asarray(model.bias_, dtype=float))
    agent_weights.insert(0, 'state', np.arange(model.n_states, dtype=int))

    emission_parameters = pd.DataFrame({'state': np.arange(model.n_states, dtype=int), 'bias': np.asarray(model.bias_, dtype=float)})
    if emission_mode == 'continuous':
        emission_parameters['variance'] = np.asarray(model.variances_, dtype=float)

    return {
        'model_type': 'moa_hmm',
        'status': 'ok',
        'emission_mode': emission_mode,
        'transition_matrix': np.asarray(model.transition_matrix_, dtype=float),
        'agent_weights': agent_weights,
        'viterbi_path': viterbi_path,
        'agent_predictions': pd.DataFrame(encoded_agents, index=pd.Index(time, name='time')),
        'event_table': _prepare_event_frame(events),
        'emission_parameters': emission_parameters,
        'fit_statistics': {
            'log_likelihood': float(model.log_likelihood_history_[-1]) if model.log_likelihood_history_ else np.nan,
            'n_iter': int(model.n_iter_),
        },
    }
