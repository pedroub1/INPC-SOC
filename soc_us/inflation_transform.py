"""
inflation_transform.py – Transformacion de niveles de precios en variacion %.

Formula: pi^h_t = (P_t / P_{t-h} - 1) * 100

Igual que el modulo SOC del INPC Mexico para consistencia entre dashboards:
  h=1  -> variacion % mensual          (ej. 0.27%)
  h=6  -> variacion % acumulada 6m     (ej. 1.63%)
  h=12 -> variacion % anual            (ej. 3.27%)
"""
import pandas as pd


def compute_inflation(price_level: pd.Series, h: int) -> pd.Series:
    """
    Calcula la variacion % simple del nivel de precios a h meses.

    Parameters
    ----------
    price_level : pd.Series
        Nivel de precios mensual (CPI o PCE).
    h : int
        Horizonte en meses (1, 6, 12).

    Returns
    -------
    pd.Series
        (P_t / P_{t-h} - 1) * 100. Los primeros h valores son NaN.
    """
    return (price_level / price_level.shift(h) - 1.0) * 100.0
