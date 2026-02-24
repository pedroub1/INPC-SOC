"""
inflation_transform.py – Transforma niveles de precios en inflación anualizada.

π^h_t = (1200 / h) * ln(P_t / P_{t-h})
"""
import numpy as np
import pandas as pd


def compute_inflation(price_level: pd.Series, h: int) -> pd.Series:
    """
    Calcula inflación anualizada a horizonte h meses.

    Parameters
    ----------
    price_level : pd.Series
        Nivel de precios mensual.
    h : int
        Horizonte en meses (1, 6, 12 ...).

    Returns
    -------
    pd.Series
        Inflación anualizada: π^h_t = (1200/h) * ln(P_t / P_{t-h}).
        Los primeros h valores serán NaN.
    """
    return (1200.0 / h) * np.log(price_level / price_level.shift(h))


def compute_all_inflation(df_inpc: pd.DataFrame, horizons: list[int]) -> dict:
    """
    Aplica compute_inflation a cada columna del DataFrame INPC y cada horizonte.

    Parameters
    ----------
    df_inpc : pd.DataFrame
        DataFrame con 16 series de nivel de precios.
    horizons : list[int]
        Lista de horizontes.

    Returns
    -------
    dict
        {(serie_name, h): pd.Series} con inflación anualizada.
    """
    result = {}
    for col in df_inpc.columns:
        series = df_inpc[col].dropna()
        for h in horizons:
            pi = compute_inflation(series, h)
            result[(col, h)] = pi
    return result
