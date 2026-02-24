"""
wavelet.py – MODWT (Haar) via pywt.swt para descomposición de inflación.

Implementación basada en:
  Verona (2026) "Forecasting inflation: The sum of the cycles outperforms the whole"

Componentes:
  D1 → escala 2-4 meses
  D2 → escala 4-8 meses
  D3 → escala 8-16 meses
  D4 → escala 16-32 meses
  D5 → escala 32-64 meses
  S5 → tendencia (>64 meses)
"""
import numpy as np
import pandas as pd
import pywt

from .config import WAVELET, J_LEVELS, COMPONENT_NAMES


def _next_power2_multiple(n: int, J: int) -> int:
    """Retorna el múltiplo más pequeño de 2^J que sea >= n."""
    m = 2 ** J
    return int(np.ceil(n / m) * m)


def decompose_series(inflation: pd.Series) -> pd.DataFrame:
    """
    Descompone una serie de inflación en J_LEVELS+1 componentes MODWT (Haar).

    Parameters
    ----------
    inflation : pd.Series
        Serie de inflación anualizada, sin NaN (ya dropna).

    Returns
    -------
    pd.DataFrame
        Columnas: D1, D2, D3, D4, D5, S5 – con el mismo índice que `inflation`.
        Suma fila a fila ≈ inflation (error < 1e-6 por valor).
    """
    series = inflation.dropna()
    n = len(series)
    values = series.values.astype(float)

    # Pad to nearest multiple of 2^J_LEVELS
    n_pad = _next_power2_multiple(n, J_LEVELS)
    padded = np.pad(values, (0, n_pad - n), mode="reflect")

    # SWT con norm=True → equivalente MODWT
    coeffs = pywt.swt(padded, WAVELET, level=J_LEVELS, norm=True)
    # coeffs[0] → nivel J (escala más gruesa), coeffs[J-1] → nivel 1 (escala más fina)
    # Para nivel j (1-indexed): coeffs[J_LEVELS - j]
    # coeffs[i] = (cA, cD) donde cA es aproximación y cD es detalle

    components = {}
    for j in range(1, J_LEVELS + 1):
        idx = J_LEVELS - j       # índice en la lista de coeffs
        cD = coeffs[idx][1][:n]  # detalle en escala j
        components[f"D{j}"] = cD

    # Smooth: aproximación del nivel más grueso
    cA = coeffs[0][0][:n]
    components[f"S{J_LEVELS}"] = cA

    df_components = pd.DataFrame(components, index=series.index)

    # Verificación de reconstrucción
    reconstruction = df_components.sum(axis=1)
    max_err = (reconstruction - series).abs().max()
    if max_err > 1e-6:
        import warnings
        warnings.warn(
            f"Wavelet reconstruction error {max_err:.2e} > 1e-6. "
            "Verifique la implementación.",
            RuntimeWarning,
        )

    return df_components[COMPONENT_NAMES]


def verify_decomposition(inflation: pd.Series) -> dict:
    """
    Verifica la descomposición y retorna estadísticas de reconstrucción.

    Returns
    -------
    dict con claves: max_error, mean_error, ok
    """
    df = decompose_series(inflation)
    reconstruction = df.sum(axis=1)
    errors = (reconstruction - inflation.dropna()).abs()
    return {
        "max_error": float(errors.max()),
        "mean_error": float(errors.mean()),
        "ok": bool(errors.max() < 1e-6),
    }
