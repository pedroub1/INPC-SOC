"""
expanding_window.py – Motor de evaluación OOS con ventana expansiva.

Para cada componente wavelet (D1..D5, S5), evalúa todos los modelos
en el período OOS usando expanding window (pseudo out-of-sample).
"""
import warnings
import numpy as np
import pandas as pd

from .combinations import build_combinations
from .config import OOS_START, AO_WINDOW


def _get_x_for_model(X_macro: pd.DataFrame | None, x_cols: list[str] | None) -> np.ndarray | None:
    """Extrae columnas relevantes de X_macro."""
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
) -> dict:
    """
    Evalúa todos los modelos en ventana expansiva para un componente wavelet.

    Parameters
    ----------
    component : pd.Series
        Serie del componente wavelet (D1..D5 o S5) con índice DatetimeIndex.
    X_macro : pd.DataFrame | None
        Predictores macro, mismo índice temporal. Puede ser None.
    model_registry : list[dict]
        Lista de modelos de models.build_model_registry().
    h : int
        Horizonte de pronóstico (meses).
    oos_start : str
        Primera fecha del período OOS.

    Returns
    -------
    dict con:
        dates        : list[pd.Timestamp] – fechas de pronóstico
        actuals      : np.ndarray         – valores realizados
        forecasts    : dict[str, np.ndarray] – pronóstico por modelo
        combinations : dict[str, np.ndarray] – 7 combinaciones
        cum_sq_errors: dict[str, np.ndarray] – errores acumulados
        best_model   : str                – modelo con menor RMSE OOS
        best_forecast: np.ndarray         – pronóstico del mejor modelo
        rmse_by_model: dict[str, float]   – RMSE por modelo
    """
    comp = component.dropna()
    if len(comp) < 10:
        return {}

    # Alinear X_macro al índice del componente
    if X_macro is not None:
        X_macro = X_macro.reindex(comp.index)

    # Calcular AO benchmark antes del loop
    # AO_t = media de π_{t-AO_WINDOW+1}..π_t (rolling mean del componente)
    ao_series = comp.rolling(AO_WINDOW, min_periods=1).mean().shift(h)

    oos_start_ts = pd.Timestamp(oos_start)
    # OOS abarca [oos_start, T-h] → pronóstico para [oos_start+h, T]
    oos_dates = comp.index[comp.index >= oos_start_ts]

    n_models = len(model_registry)
    model_names = [m["name"] for m in model_registry]

    dates_out = []
    actuals_out = []
    forecasts_out = {name: [] for name in model_names}
    combo_keys = ["C_MEAN", "C_MEDIAN", "C_TRIMEAN",
                  "C_DMSPE025", "C_DMSPE050", "C_DMSPE075", "C_DMSPE100"]
    combos_out = {k: [] for k in combo_keys}
    cum_sq = np.zeros(n_models)  # errores cuadráticos acumulados por modelo

    comp_arr = comp.values
    comp_idx = comp.index

    for t_idx, t_date in enumerate(comp_idx):
        if t_date < oos_start_ts:
            continue
        # Necesitamos el valor realizado en t+h
        future_idx = t_idx + h
        if future_idx >= len(comp_arr):
            break

        actual = comp_arr[future_idx]
        if np.isnan(actual):
            continue

        # Datos de entrenamiento hasta t (inclusive)
        y_train = comp_arr[: t_idx + 1]

        # X_macro hasta t
        if X_macro is not None:
            X_train_full = X_macro.iloc[: t_idx + 1].values.astype(float)
        else:
            X_train_full = None

        # Pronósticos de cada modelo
        f_arr = np.full(n_models, np.nan)
        for m_idx, model in enumerate(model_registry):
            try:
                x_cols = model.get("x_cols")
                if x_cols is None or X_train_full is None:
                    x_for_model = None
                else:
                    if X_macro is not None:
                        available = [c for c in x_cols if c in X_macro.columns]
                        if not available:
                            x_for_model = None
                        else:
                            col_idxs = [list(X_macro.columns).index(c) for c in available]
                            x_for_model = X_train_full[:, col_idxs]
                    else:
                        x_for_model = None

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fcast = model["fn"](y_train, x_for_model, h)

                if np.isfinite(fcast):
                    f_arr[m_idx] = fcast
            except Exception:
                pass

        # Combinar
        combos = build_combinations(f_arr, cum_sq)

        # Actualizar errores acumulados
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

    # RMSE por modelo (ignorar NaN)
    def _rmse(f: np.ndarray) -> float:
        mask = ~np.isnan(f) & ~np.isnan(actuals_arr)
        if mask.sum() == 0:
            return np.inf
        return float(np.sqrt(np.mean((actuals_arr[mask] - f[mask]) ** 2)))

    rmse_models = {name: _rmse(forecasts_arr[name]) for name in model_names}
    rmse_combos = {name: _rmse(combos_arr[name]) for name in combo_keys}
    rmse_all = {**rmse_models, **rmse_combos}

    best_model = min(rmse_all, key=lambda k: rmse_all[k] if np.isfinite(rmse_all[k]) else np.inf)

    if best_model in forecasts_arr:
        best_forecast = forecasts_arr[best_model]
    else:
        best_forecast = combos_arr[best_model]

    return {
        "dates": dates_out,
        "actuals": actuals_arr,
        "forecasts": forecasts_arr,
        "combinations": combos_arr,
        "best_model": best_model,
        "best_forecast": best_forecast,
        "rmse_by_model": rmse_all,
    }
