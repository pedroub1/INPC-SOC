"""
expanding_window.py – Motor de evaluacion OOS con ventana expansiva para soc_us.

Extension respecto al modulo Mexico:
  - Soporte para PLS-2: cuando model["use_filtered_x"] == True, se usa
    X_macro_filtered (predictores descompuestos en la misma frecuencia que el
    componente de inflacion que se pronostica) en lugar de X_macro original.
"""
import warnings
import numpy as np
import pandas as pd

from .combinations import build_combinations
from .config import OOS_START, AO_WINDOW


def _get_x_cols(X_macro: pd.DataFrame | None, x_cols: list[str] | None) -> np.ndarray | None:
    if X_macro is None or x_cols is None:
        return None
    available = [c for c in x_cols if c in X_macro.columns]
    if not available:
        return None
    return X_macro[available].values


def evaluate_component(
    component: pd.Series,
    X_macro: pd.DataFrame | None,
    model_registry: list[dict],
    h: int,
    oos_start: str = OOS_START,
    X_macro_filtered: pd.DataFrame | None = None,
) -> dict:
    """
    Evalua todos los modelos en ventana expansiva para un componente wavelet.

    Parameters
    ----------
    component : pd.Series
        Serie del componente wavelet (D1..D5 o S5).
    X_macro : pd.DataFrame | None
        Predictores macro originales (no filtrados).
    model_registry : list[dict]
        Lista de modelos de models.build_model_registry().
    h : int
        Horizonte de pronostico (meses).
    oos_start : str
        Primera fecha del periodo OOS.
    X_macro_filtered : pd.DataFrame | None
        Predictores macro descompuestos en la frecuencia de este componente.
        Se usa para PLS-2 (modelos con use_filtered_x=True).

    Returns
    -------
    dict con: dates, actuals, forecasts, combinations, best_model, best_forecast, rmse_by_model
    """
    comp = component.dropna()
    if len(comp) < 10:
        return {}

    # Alinear X_macro al indice del componente
    if X_macro is not None:
        X_macro = X_macro.reindex(comp.index)
    if X_macro_filtered is not None:
        X_macro_filtered = X_macro_filtered.reindex(comp.index)

    # AO benchmark
    ao_series = comp.rolling(AO_WINDOW, min_periods=1).mean().shift(h)

    oos_start_ts = pd.Timestamp(oos_start)
    n_models = len(model_registry)
    model_names = [m["name"] for m in model_registry]

    dates_out = []
    actuals_out = []
    forecasts_out = {name: [] for name in model_names}
    combo_keys = ["C_MEAN", "C_MEDIAN", "C_TRIMEAN",
                  "C_DMSPE025", "C_DMSPE050", "C_DMSPE075", "C_DMSPE100"]
    combos_out = {k: [] for k in combo_keys}
    cum_sq = np.zeros(n_models)

    comp_arr = comp.values
    comp_idx = comp.index

    # Columnas de X_macro_filtered disponibles para PLS-2
    filtered_cols = list(X_macro_filtered.columns) if X_macro_filtered is not None else []

    for t_idx, t_date in enumerate(comp_idx):
        if t_date < oos_start_ts:
            continue
        future_idx = t_idx + h
        if future_idx >= len(comp_arr):
            break
        actual = comp_arr[future_idx]
        if np.isnan(actual):
            continue

        y_train = comp_arr[: t_idx + 1]

        X_train_full = (
            X_macro.iloc[: t_idx + 1].values.astype(float)
            if X_macro is not None else None
        )
        X_filtered_full = (
            X_macro_filtered.iloc[: t_idx + 1].values.astype(float)
            if X_macro_filtered is not None else None
        )

        f_arr = np.full(n_models, np.nan)
        for m_idx, model in enumerate(model_registry):
            try:
                use_filtered = model.get("use_filtered_x", False)
                x_cols = model.get("x_cols")

                if use_filtered and X_filtered_full is not None:
                    # PLS-2: usa predictores filtrados por frecuencia
                    avail = [c for c in x_cols if c in filtered_cols] if x_cols else []
                    if avail:
                        col_idxs = [filtered_cols.index(c) for c in avail]
                        x_for_model = X_filtered_full[:, col_idxs]
                    else:
                        x_for_model = None
                else:
                    # Modelos estandar: usa X_macro original
                    if x_cols is None or X_train_full is None or X_macro is None:
                        x_for_model = None
                    else:
                        avail = [c for c in x_cols if c in X_macro.columns]
                        if not avail:
                            x_for_model = None
                        else:
                            col_idxs = [list(X_macro.columns).index(c) for c in avail]
                            x_for_model = X_train_full[:, col_idxs]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fcast = model["fn"](y_train, x_for_model, h)
                if np.isfinite(fcast):
                    f_arr[m_idx] = fcast
            except Exception:
                pass

        combos = build_combinations(f_arr, cum_sq)

        for m_idx in range(n_models):
            if not np.isnan(f_arr[m_idx]):
                cum_sq[m_idx] += (actual - f_arr[m_idx]) ** 2

        dates_out.append(comp_idx[future_idx])
        actuals_out.append(actual)
        for m_idx, name in enumerate(model_names):
            forecasts_out[name].append(f_arr[m_idx])
        for k in combo_keys:
            combos_out[k].append(combos.get(k, np.nan))

    if not dates_out:
        return {}

    actuals_arr = np.array(actuals_out)
    forecasts_arr = {k: np.array(v) for k, v in forecasts_out.items()}
    combos_arr = {k: np.array(v) for k, v in combos_out.items()}

    def _rmse(f):
        mask = ~np.isnan(f) & ~np.isnan(actuals_arr)
        if mask.sum() == 0:
            return np.inf
        return float(np.sqrt(np.mean((actuals_arr[mask] - f[mask]) ** 2)))

    rmse_models = {name: _rmse(forecasts_arr[name]) for name in model_names}
    rmse_combos = {name: _rmse(combos_arr[name]) for name in combo_keys}
    rmse_all = {**rmse_models, **rmse_combos}

    best_model = min(rmse_all, key=lambda k: rmse_all[k] if np.isfinite(rmse_all[k]) else np.inf)
    best_forecast = (
        forecasts_arr[best_model] if best_model in forecasts_arr else combos_arr[best_model]
    )

    return {
        "dates": dates_out,
        "actuals": actuals_arr,
        "forecasts": forecasts_arr,
        "combinations": combos_arr,
        "best_model": best_model,
        "best_forecast": best_forecast,
        "rmse_by_model": rmse_all,
    }
