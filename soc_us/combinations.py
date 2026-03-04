"""
combinations.py – Reglas de combinacion de pronosticos para soc_us.
"""
import numpy as np


def c_mean(forecasts: np.ndarray) -> float:
    valid = forecasts[~np.isnan(forecasts)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


def c_median(forecasts: np.ndarray) -> float:
    valid = forecasts[~np.isnan(forecasts)]
    return float(np.median(valid)) if len(valid) > 0 else np.nan


def c_trimmed_mean(forecasts: np.ndarray, trim: float = 0.10) -> float:
    valid = forecasts[~np.isnan(forecasts)]
    if len(valid) == 0:
        return np.nan
    n = len(valid)
    k = max(1, int(np.floor(trim * n)))
    if 2 * k >= n:
        return float(np.mean(valid))
    return float(np.mean(np.sort(valid)[k:-k]))


def c_dmspe(forecasts: np.ndarray, cumulative_sq_errors: np.ndarray, theta: float) -> float:
    valid = ~(np.isnan(forecasts) | np.isnan(cumulative_sq_errors))
    if valid.sum() == 0:
        return np.nan
    f = forecasts[valid]
    mspe = np.where(cumulative_sq_errors[valid] <= 0, 1e-12, cumulative_sq_errors[valid])
    w = 1.0 / (mspe ** theta)
    w /= w.sum()
    return float(w @ f)


def build_combinations(forecasts: np.ndarray, cumulative_sq_errors: np.ndarray) -> dict:
    return {
        "C_MEAN":     c_mean(forecasts),
        "C_MEDIAN":   c_median(forecasts),
        "C_TRIMEAN":  c_trimmed_mean(forecasts, trim=0.10),
        "C_DMSPE025": c_dmspe(forecasts, cumulative_sq_errors, theta=0.25),
        "C_DMSPE050": c_dmspe(forecasts, cumulative_sq_errors, theta=0.50),
        "C_DMSPE075": c_dmspe(forecasts, cumulative_sq_errors, theta=0.75),
        "C_DMSPE100": c_dmspe(forecasts, cumulative_sq_errors, theta=1.00),
    }
