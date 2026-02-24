"""
combinations.py – Reglas de combinación de pronósticos para el modelo SOC.

Basado en Verona (2026), sección 3.2.
"""
import numpy as np


def c_mean(forecasts: np.ndarray) -> float:
    """Media simple de pronósticos individuales."""
    valid = forecasts[~np.isnan(forecasts)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid))


def c_median(forecasts: np.ndarray) -> float:
    """Mediana de pronósticos individuales."""
    valid = forecasts[~np.isnan(forecasts)]
    if len(valid) == 0:
        return np.nan
    return float(np.median(valid))


def c_trimmed_mean(forecasts: np.ndarray, trim: float = 0.10) -> float:
    """Media recortada (10% por defecto)."""
    valid = forecasts[~np.isnan(forecasts)]
    if len(valid) == 0:
        return np.nan
    n = len(valid)
    k = max(1, int(np.floor(trim * n)))
    if 2 * k >= n:
        return float(np.mean(valid))
    sorted_f = np.sort(valid)
    trimmed = sorted_f[k:-k]
    return float(np.mean(trimmed))


def c_dmspe(forecasts: np.ndarray, cumulative_sq_errors: np.ndarray, theta: float) -> float:
    """
    Combinación DMSPE: pesos inversamente proporcionales a MSPE^theta.

    Parameters
    ----------
    forecasts : np.ndarray (M,)
        Pronósticos de cada modelo en el período actual.
    cumulative_sq_errors : np.ndarray (M,)
        Error cuadrático acumulado de cada modelo hasta t (excluyendo t actual).
    theta : float
        Parámetro de penalización (0.25, 0.50, 0.75, 1.0).

    Returns
    -------
    float
        Pronóstico combinado ponderado.
    """
    valid_mask = ~(np.isnan(forecasts) | np.isnan(cumulative_sq_errors))
    if valid_mask.sum() == 0:
        return np.nan

    f_valid = forecasts[valid_mask]
    err_valid = cumulative_sq_errors[valid_mask]

    # Evitar división por cero
    mspe = np.where(err_valid <= 0, 1e-12, err_valid)
    weights = 1.0 / (mspe ** theta)
    weights = weights / weights.sum()

    return float(weights @ f_valid)


def build_combinations(
    forecasts: np.ndarray,
    cumulative_sq_errors: np.ndarray,
) -> dict[str, float]:
    """
    Construye las 7 combinaciones de pronósticos.

    Parameters
    ----------
    forecasts : np.ndarray (M,)
        Pronósticos de M modelos individuales.
    cumulative_sq_errors : np.ndarray (M,)
        Errores cuadráticos acumulados de cada modelo.

    Returns
    -------
    dict con claves:
        C_MEAN, C_MEDIAN, C_TRIMEAN,
        C_DMSPE025, C_DMSPE050, C_DMSPE075, C_DMSPE100
    """
    return {
        "C_MEAN":    c_mean(forecasts),
        "C_MEDIAN":  c_median(forecasts),
        "C_TRIMEAN": c_trimmed_mean(forecasts, trim=0.10),
        "C_DMSPE025": c_dmspe(forecasts, cumulative_sq_errors, theta=0.25),
        "C_DMSPE050": c_dmspe(forecasts, cumulative_sq_errors, theta=0.50),
        "C_DMSPE075": c_dmspe(forecasts, cumulative_sq_errors, theta=0.75),
        "C_DMSPE100": c_dmspe(forecasts, cumulative_sq_errors, theta=1.00),
    }
