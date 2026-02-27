"""
inflation_transform.py – Transforma niveles de precios en variacion % simple.

Formula: π^h_t = (P_t / P_{t-h} - 1) * 100

Esto replica la forma en que INEGI/Banxico reportan la inflacion en Mexico:
  h=1  → variacion % mensual        (ej. 0.35%)
  h=6  → variacion % acumulada 6m   (ej. 2.10%)
  h=12 → variacion % anual          (ej. 4.21%)
"""
import pandas as pd


def compute_inflation(price_level: pd.Series, h: int) -> pd.Series:
    """
    Calcula la variacion % simple del nivel de precios a h meses.

    Parameters
    ----------
    price_level : pd.Series
        Nivel de precios mensual.
    h : int
        Horizonte en meses (1, 6, 12 ...).

    Returns
    -------
    pd.Series
        Variacion porcentual: (P_t / P_{t-h} - 1) * 100.
        Los primeros h valores seran NaN.
    """
    return (price_level / price_level.shift(h) - 1.0) * 100.0


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
