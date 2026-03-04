"""
forecast_future.py – Pronostico genuino del SOC para los proximos N meses (US).
"""
import os
import warnings
import numpy as np
import pandas as pd

from .config import (
    US_DATA_PATH, US_MACRO_PATH, FORECASTS_DIR,
    SAFE_NAMES, COMPONENT_NAMES,
)
from .inflation_transform import compute_inflation
from .wavelet import decompose_series
from .models import build_model_registry


def _nearest_h(h: int) -> int:
    return min([1, 6, 12], key=lambda x: abs(x - h))


def _load_best_models(serie_name: str, h: int) -> dict[str, str]:
    safe = SAFE_NAMES.get(serie_name, serie_name)
    path = os.path.join(FORECASTS_DIR, f"{safe}_h{h}.parquet")
    if not os.path.exists(path):
        return {c: "AR_AIC" for c in COMPONENT_NAMES}
    df = pd.read_parquet(path)
    return {
        comp: str(df[f"BESTMODEL_{comp}"].iloc[0])
        if (f"BESTMODEL_{comp}" in df.columns and len(df) > 0) else "AR_AIC"
        for comp in COMPONENT_NAMES
    }


def _load_oos_rmse(serie_name: str, h: int) -> float | None:
    safe = SAFE_NAMES.get(serie_name, serie_name)
    path = os.path.join(FORECASTS_DIR, f"{safe}_h{h}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if "actual" not in df.columns or "SOC" not in df.columns:
        return None
    actual = df["actual"].values.astype(float)
    soc = df["SOC"].values.astype(float)
    mask = ~np.isnan(actual) & ~np.isnan(soc)
    if mask.sum() == 0:
        return None
    return float(np.sqrt(np.mean((actual[mask] - soc[mask]) ** 2)))


def forecast_future(
    serie_name: str,
    n_months: int = 12,
    df_prices: pd.DataFrame | None = None,
    df_macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Genera pronosticos SOC para los proximos n_months meses.

    Returns
    -------
    pd.DataFrame con columnas: h, SOC, SOC_opt, sigma, upper_68, lower_68,
                                upper_90, lower_90, comp_D1..comp_S5, best_D1..best_S5
    """
    if df_prices is None:
        df_prices = pd.read_csv(US_DATA_PATH, index_col="Date", parse_dates=True)
    if df_macro is None and os.path.exists(US_MACRO_PATH):
        df_macro = pd.read_csv(US_MACRO_PATH, index_col="Date", parse_dates=True)
        df_macro.index = pd.to_datetime(df_macro.index)

    if serie_name not in df_prices.columns:
        raise ValueError(f"Serie '{serie_name}' no encontrada.")

    price_level = df_prices[serie_name].dropna()
    last_date = price_level.index[-1]

    macro_cols = list(df_macro.columns) if df_macro is not None else []
    model_registry = build_model_registry(macro_cols)
    model_by_name = {m["name"]: m for m in model_registry}

    rmse_ref = {h_ref: _load_oos_rmse(serie_name, h_ref) for h_ref in [1, 6, 12]}

    rows = []
    for h in range(1, n_months + 1):
        ref_h = _nearest_h(h)
        best_models = _load_best_models(serie_name, ref_h)

        inflation = compute_inflation(price_level, h).dropna()
        if len(inflation) < 10:
            continue
        try:
            df_comp = decompose_series(inflation)
        except Exception:
            continue

        X_macro = df_macro.reindex(inflation.index) if df_macro is not None else None

        soc = 0.0
        soc_opt = 0.0
        comp_fcasts = {}
        comp_best = {}

        for comp_name in COMPONENT_NAMES:
            comp_series = df_comp[comp_name]
            y_train = comp_series.values.astype(float)
            best_name = best_models.get(comp_name, "AR_AIC")
            comp_best[comp_name] = best_name
            model_info = model_by_name.get(best_name)

            X_train = None
            if model_info and model_info.get("x_cols") and X_macro is not None:
                avail = [c for c in model_info["x_cols"] if c in X_macro.columns]
                if avail:
                    # PLS-2: para pronostico futuro se usa X original (no hay OOS aqui)
                    X_train = X_macro[avail].values.astype(float)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fcast = (
                        model_info["fn"](y_train, X_train, h)
                        if model_info else float(np.nanmean(y_train[-12:]))
                    )
                if not np.isfinite(fcast):
                    fcast = float(np.nanmean(y_train[-12:]))
            except Exception:
                fcast = float(np.nanmean(y_train[-12:]))

            comp_fcasts[comp_name] = fcast
            soc += fcast
            if comp_name != "D1":
                soc_opt += fcast

        rmse_base = rmse_ref.get(ref_h)
        sigma = rmse_base * np.sqrt(h / ref_h) if (rmse_base and ref_h > 0) else np.nan
        target_date = last_date + pd.DateOffset(months=h)

        row = {
            "fecha":    target_date,
            "h":        h,
            "SOC":      round(soc, 4),
            "SOC_opt":  round(soc_opt, 4),
            "sigma":    round(sigma, 4) if np.isfinite(sigma) else np.nan,
            "upper_68": round(soc + sigma, 4) if np.isfinite(sigma) else np.nan,
            "lower_68": round(soc - sigma, 4) if np.isfinite(sigma) else np.nan,
            "upper_90": round(soc + 1.65 * sigma, 4) if np.isfinite(sigma) else np.nan,
            "lower_90": round(soc - 1.65 * sigma, 4) if np.isfinite(sigma) else np.nan,
        }
        for comp_name in COMPONENT_NAMES:
            row[f"comp_{comp_name}"] = round(comp_fcasts.get(comp_name, np.nan), 4)
            row[f"best_{comp_name}"] = comp_best.get(comp_name, "—")
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("fecha")


def last_observed_inflation(
    serie_name: str,
    n_history: int = 48,
    df_prices: pd.DataFrame | None = None,
) -> pd.Series:
    if df_prices is None:
        df_prices = pd.read_csv(US_DATA_PATH, index_col="Date", parse_dates=True)
    price_level = df_prices[serie_name].dropna()
    return compute_inflation(price_level, 1).dropna().iloc[-n_history:]
