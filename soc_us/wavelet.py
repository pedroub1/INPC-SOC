"""
wavelet.py – MODWT (Haar) via pywt.swt para descomposicion de inflacion US.

Identico al modulo soc/wavelet.py, usando parametros de soc_us/config.py.

Componentes para datos mensuales (J=5):
  D1 -> ciclos de 2-4 meses
  D2 -> ciclos de 4-8 meses
  D3 -> ciclos de 8-16 meses
  D4 -> ciclos de 16-32 meses
  D5 -> ciclos de 32-64 meses
  S5 -> tendencia de largo plazo (>64 meses)
"""
import numpy as np
import pandas as pd
import pywt

from .config import WAVELET, J_LEVELS, COMPONENT_NAMES


def _next_power2_multiple(n: int, J: int) -> int:
    m = 2 ** J
    return int(np.ceil(n / m) * m)


def decompose_series(series: pd.Series) -> pd.DataFrame:
    """
    Descompone una serie en J_LEVELS+1 componentes MODWT (Haar).

    Returns
    -------
    pd.DataFrame con columnas D1..D5, S5 (mismo indice que series).
    La suma fila a fila reconstituye la serie original (error < 1e-6).
    """
    series = series.dropna()
    n = len(series)
    values = series.values.astype(float)

    n_pad = _next_power2_multiple(n, J_LEVELS)
    padded = np.pad(values, (0, n_pad - n), mode="reflect")

    # SWT con norm=True -> equivalente MODWT
    coeffs = pywt.swt(padded, WAVELET, level=J_LEVELS, norm=True)

    components = {}
    for j in range(1, J_LEVELS + 1):
        idx = J_LEVELS - j
        cD = coeffs[idx][1][:n]
        components[f"D{j}"] = cD

    cA = coeffs[0][0][:n]
    components[f"S{J_LEVELS}"] = cA

    df = pd.DataFrame(components, index=series.index)

    max_err = (df.sum(axis=1) - series).abs().max()
    if max_err > 1e-6:
        import warnings
        warnings.warn(
            f"Wavelet reconstruction error {max_err:.2e} > 1e-6.",
            RuntimeWarning,
        )

    return df[COMPONENT_NAMES]
