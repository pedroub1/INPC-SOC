"""
forecast_future.py – Pronostico genuino del SOC para los proximos N meses.

Metodologia:
  Para cada horizonte h = 1, 2, ..., N:
    1. Inflacion: pi^h_t = (P_t / P_{t-h} - 1) * 100  [toda la muestra]
       h=1  -> variacion % mensual   (ej. 0.35 %)
       h=6  -> variacion % acumulada 6 meses (ej. 2.10 %)
       h=12 -> variacion % anual     (ej. 4.21 %)
    2. Descomposicion MODWT Haar -> D1..D5, S5
    3. Por componente: aplica el mejor modelo identificado en el OOS
       (del horizonte evaluado mas cercano: 1, 6 o 12)
       entrenado sobre TODOS los datos disponibles
    4. SOC_h = sum de pronosticos de componentes
  Bandas de incertidumbre: RMSE_OOS * sqrt(h / h_ref), escalado por horizonte.
"""
import os
import warnings
import numpy as np
import pandas as pd

from .config import (
    INPC_PATH, MACRO_PATH, FORECASTS_DIR, SAFE_NAMES, COMPONENT_NAMES,
)
from .inflation_transform import compute_inflation
from .wavelet import decompose_series
from .models import build_model_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_h(h: int) -> int:
    """Horizonte evaluado mas cercano de {1, 6, 12}."""
    return min([1, 6, 12], key=lambda x: abs(x - h))


def _load_best_models(serie_name: str, h: int) -> dict[str, str]:
    """Lee el mejor modelo por componente del Parquet OOS mas cercano."""
    safe = SAFE_NAMES.get(serie_name, serie_name.replace(" ", "_"))
    path = os.path.join(FORECASTS_DIR, f"{safe}_h{h}.parquet")
    if not os.path.exists(path):
        return {c: "AR_AIC" for c in COMPONENT_NAMES}
    df = pd.read_parquet(path)
    result = {}
    for comp in COMPONENT_NAMES:
        col = f"BESTMODEL_{comp}"
        result[comp] = str(df[col].iloc[0]) if (col in df.columns and len(df) > 0) else "AR_AIC"
    return result


def _load_oos_rmse(serie_name: str, h: int) -> float | None:
    """Carga el RMSE OOS del SOC para (serie, h) como referencia de incertidumbre."""
    safe = SAFE_NAMES.get(serie_name, serie_name.replace(" ", "_"))
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


# ---------------------------------------------------------------------------
# Funcion principal
# ---------------------------------------------------------------------------

def forecast_future(
    serie_name: str,
    n_months: int = 12,
    df_inpc: pd.DataFrame | None = None,
    df_macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Genera pronosticos SOC para los proximos n_months meses.

    Parameters
    ----------
    serie_name : str
        Nombre de la serie INPC (sin espacios de indentacion).
    n_months : int
        Numero de meses a pronosticar (default 12).
    df_inpc, df_macro : pd.DataFrame | None
        Datos ya cargados; si None se cargan desde disco.

    Returns
    -------
    pd.DataFrame con columnas:
        fecha, h, SOC, SOC_opt, sigma_68, sigma_90,
        upper_68, lower_68, upper_90, lower_90,
        comp_D1 .. comp_S5, best_model_D1 .. best_model_S5
    """
    # --- Carga de datos ---
    if df_inpc is None:
        df_inpc = pd.read_csv(INPC_PATH, index_col="Fecha", parse_dates=True)
        df_inpc.columns = [c.strip() for c in df_inpc.columns]
    if df_macro is None and os.path.exists(MACRO_PATH):
        df_macro = pd.read_csv(MACRO_PATH, index_col="Fecha", parse_dates=True)
        df_macro.index = pd.to_datetime(df_macro.index)

    if serie_name not in df_inpc.columns:
        raise ValueError(f"Serie '{serie_name}' no encontrada en INPC.")

    price_level = df_inpc[serie_name].dropna()
    last_date = price_level.index[-1]

    macro_cols = list(df_macro.columns) if df_macro is not None else []
    model_registry = build_model_registry(macro_cols)
    model_by_name = {m["name"]: m for m in model_registry}

    # RMSE OOS por horizonte (para bandas de incertidumbre)
    rmse_ref = {h_ref: _load_oos_rmse(serie_name, h_ref) for h_ref in [1, 6, 12]}

    rows = []
    for h in range(1, n_months + 1):
        ref_h = _nearest_h(h)
        best_models = _load_best_models(serie_name, ref_h)

        # Inflacion a horizonte h (toda la muestra disponible)
        inflation = compute_inflation(price_level, h).dropna()
        if len(inflation) < 10:
            continue

        # Descomposicion wavelet
        try:
            df_comp = decompose_series(inflation)
        except Exception as e:
            print(f"  [warn] wavelet h={h}: {e}")
            continue

        # Macro alineada
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

            # Preparar X
            X_train = None
            if model_info and model_info.get("x_cols") and X_macro is not None:
                available = [c for c in model_info["x_cols"] if c in X_macro.columns]
                if available:
                    col_idxs = [list(X_macro.columns).index(c) for c in available]
                    X_train = X_macro.values.astype(float)[:, col_idxs]

            # Pronostico
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

        # Bandas de incertidumbre: RMSE_ref * sqrt(h / h_ref)
        rmse_base = rmse_ref.get(ref_h)
        if rmse_base is not None and ref_h > 0:
            sigma = rmse_base * np.sqrt(h / ref_h)
        else:
            sigma = np.nan

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
    df_inpc: pd.DataFrame | None = None,
) -> pd.Series:
    """Retorna los ultimos n_history meses de inflacion mensual (h=1) observada."""
    if df_inpc is None:
        df_inpc = pd.read_csv(INPC_PATH, index_col="Fecha", parse_dates=True)
        df_inpc.columns = [c.strip() for c in df_inpc.columns]
    price_level = df_inpc[serie_name].dropna()
    inflation = compute_inflation(price_level, 1).dropna()
    return inflation.iloc[-n_history:]
