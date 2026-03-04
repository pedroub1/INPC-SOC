"""
fit_soc_us.py – Script offline: ajusta modelos SOC para CPI y PCE de EE.UU.

Uso:
    python -m soc_us.fit_soc_us                       # CPI + PCE, todos los horizontes
    python -m soc_us.fit_soc_us --series CPI          # solo CPI
    python -m soc_us.fit_soc_us --horizon 1           # solo h=1
    python -m soc_us.fit_soc_us --workers 4           # paralelismo
    python -m soc_us.fit_soc_us --update-data         # re-fetch datos primero
    python -m soc_us.fit_soc_us --rebuild-metrics     # reconstruye all_metrics.parquet

Tiempo estimado: 30-60 min (2 series x 3 horizontes, con shortages).
"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from soc_us.config import (
    HORIZONS, US_DATA_PATH, US_MACRO_PATH, RESULTS_DIR,
    COMPONENTS_DIR, FORECASTS_DIR, METRICS_DIR,
    SAFE_NAMES, OOS_START, AO_WINDOW, COMPONENT_NAMES,
    US_SERIES_NAMES,
)
from soc_us.inflation_transform import compute_inflation
from soc_us.wavelet import decompose_series
from soc_us.expanding_window import evaluate_component
from soc_us.models import build_model_registry


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_prices() -> pd.DataFrame:
    df = pd.read_csv(US_DATA_PATH, index_col="Date", parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


def load_macro() -> pd.DataFrame | None:
    if not os.path.exists(US_MACRO_PATH):
        print(f"[fit_soc_us] AVISO: {US_MACRO_PATH} no existe. Sin predictores macro.")
        return None
    df = pd.read_csv(US_MACRO_PATH, index_col="Date", parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# Precomputo de predictores filtrados por frecuencia (para PLS-2)
# ---------------------------------------------------------------------------

def precompute_filtered_predictors(
    X_macro: pd.DataFrame,
    inflation_index: pd.Index,
) -> dict[str, pd.DataFrame]:
    """
    Descompone cada columna macro en sus componentes wavelet.
    Retorna dict: {comp_name: DataFrame con D_j de cada predictor}.

    Estos se usan para PLS-2: cuando se pronostica D_j de inflacion,
    se usan los D_j de cada predictor como X_macro_filtered.
    """
    X_aligned = X_macro.reindex(inflation_index)
    filtered = {c: {} for c in COMPONENT_NAMES}

    for col in X_aligned.columns:
        series = X_aligned[col].dropna()
        if len(series) < 32:
            continue
        try:
            df_comp = decompose_series(series)
            for comp_name in COMPONENT_NAMES:
                filtered[comp_name][col] = df_comp[comp_name].reindex(inflation_index)
        except Exception as e:
            print(f"  [warn] wavelet predictor {col}: {e}")

    return {
        comp: pd.DataFrame(cols, index=inflation_index).ffill()
        for comp, cols in filtered.items()
    }


# ---------------------------------------------------------------------------
# Tarea unitaria (serie x horizonte)
# ---------------------------------------------------------------------------

def _process_one(
    serie_name: str,
    h: int,
    df_prices: pd.DataFrame,
    df_macro: pd.DataFrame | None,
) -> dict:
    safe = SAFE_NAMES.get(serie_name, serie_name)
    print(f"  [start] {serie_name} h={h}")
    t0 = time.time()

    # Nivel de precios
    price_level = df_prices[serie_name].dropna()
    if len(price_level) < 50:
        return {"serie": serie_name, "h": h, "status": "skip"}

    # Inflacion
    inflation = compute_inflation(price_level, h).dropna()
    if len(inflation) < 30:
        return {"serie": serie_name, "h": h, "status": "skip"}

    # Descomposicion wavelet
    try:
        df_comp = decompose_series(inflation)
    except Exception as e:
        print(f"  [error] {serie_name} h={h}: wavelet fallo – {e}")
        return {"serie": serie_name, "h": h, "status": "error", "error": str(e)}

    # Guardar componentes
    comp_path = os.path.join(COMPONENTS_DIR, f"{safe}_h{h}.parquet")
    df_comp.to_parquet(comp_path)

    # Macro alineada
    X_macro = df_macro.reindex(inflation.index) if df_macro is not None else None
    macro_cols = list(X_macro.columns) if X_macro is not None else []
    model_registry = build_model_registry(macro_cols)

    # Predictores filtrados por frecuencia (para PLS-2)
    filtered_predictors: dict[str, pd.DataFrame] = {}
    has_pls2 = any(m.get("use_filtered_x") for m in model_registry)
    if has_pls2 and X_macro is not None:
        print(f"    Precomputando predictores filtrados para PLS-2...")
        filtered_predictors = precompute_filtered_predictors(X_macro, inflation.index)

    # Evaluar cada componente
    component_results = {}
    for comp_name in COMPONENT_NAMES:
        comp_series = df_comp[comp_name]
        x_filtered = filtered_predictors.get(comp_name)
        try:
            res = evaluate_component(
                comp_series, X_macro, model_registry, h, OOS_START,
                X_macro_filtered=x_filtered,
            )
            component_results[comp_name] = res
        except Exception as e:
            print(f"  [warn] {serie_name} h={h} {comp_name}: {e}")
            component_results[comp_name] = {}

    # Fechas comunes OOS
    all_dates = None
    for comp_name in COMPONENT_NAMES:
        res = component_results.get(comp_name, {})
        if res and "dates" in res:
            dates_set = set(res["dates"])
            all_dates = dates_set if all_dates is None else all_dates & dates_set

    if not all_dates:
        return {"serie": serie_name, "h": h, "status": "skip"}

    sorted_dates = sorted(all_dates)
    actual_inflation = inflation.reindex(sorted_dates)

    soc_arr = np.zeros(len(sorted_dates))
    soc_opt_arr = np.zeros(len(sorted_dates))
    rows_data = {"date": sorted_dates, "actual": actual_inflation.values}

    for comp_name in COMPONENT_NAMES:
        res = component_results.get(comp_name, {})
        if not res or "dates" not in res:
            continue
        date_to_idx = {d: i for i, d in enumerate(res["dates"])}
        best_f = res.get("best_forecast", np.full(len(res["dates"]), np.nan))

        comp_forecast = np.array([
            best_f[date_to_idx[d]] if d in date_to_idx else np.nan
            for d in sorted_dates
        ])
        soc_arr += np.where(np.isnan(comp_forecast), 0, comp_forecast)
        if comp_name != "D1":
            soc_opt_arr += np.where(np.isnan(comp_forecast), 0, comp_forecast)

        rows_data[f"BEST_{comp_name}"] = comp_forecast
        rows_data[f"BESTMODEL_{comp_name}"] = res.get("best_model", "NA")

        for m_name, f_arr in res.get("forecasts", {}).items():
            arr_aligned = np.array([
                f_arr[date_to_idx[d]] if d in date_to_idx and date_to_idx[d] < len(f_arr) else np.nan
                for d in sorted_dates
            ])
            rows_data[f"{comp_name}_{m_name}"] = arr_aligned

    # AO benchmark
    ao_values = []
    for d in sorted_dates:
        past = inflation[inflation.index < d]
        ao_values.append(float(past.iloc[-AO_WINDOW:].mean()) if len(past) > 0 else np.nan)

    rows_data["AO"] = ao_values
    rows_data["SOC"] = soc_arr
    rows_data["SOC_opt"] = soc_opt_arr

    df_out = pd.DataFrame(rows_data).set_index("date")
    fc_path = os.path.join(FORECASTS_DIR, f"{safe}_h{h}.parquet")
    df_out.to_parquet(fc_path)

    elapsed = time.time() - t0
    print(f"  [done]  {serie_name} h={h} en {elapsed:.1f}s -> {fc_path}")

    # Metricas
    actual = df_out["actual"].values.astype(float)
    ao = df_out["AO"].values.astype(float)
    soc = df_out["SOC"].values.astype(float)
    soc_opt = df_out["SOC_opt"].values.astype(float)

    def rmse(a, f):
        mask = ~np.isnan(a) & ~np.isnan(f)
        return float(np.sqrt(np.mean((a[mask] - f[mask]) ** 2))) if mask.sum() > 0 else np.nan

    rmse_ao = rmse(actual, ao)
    rmse_soc = rmse(actual, soc)
    rmse_soc_opt = rmse(actual, soc_opt)

    return {
        "serie": serie_name, "h": h, "status": "ok",
        "rmse_AO": rmse_ao, "rmse_SOC": rmse_soc, "rmse_SOC_opt": rmse_soc_opt,
        "ratio_SOC": rmse_soc / rmse_ao if rmse_ao and rmse_ao > 0 else np.nan,
        "ratio_SOC_opt": rmse_soc_opt / rmse_ao if rmse_ao and rmse_ao > 0 else np.nan,
        "n_oos": len(sorted_dates), "elapsed_s": elapsed,
    }


def _process_one_safe(args):
    serie_name, h, df_prices, df_macro = args
    try:
        return _process_one(serie_name, h, df_prices, df_macro)
    except Exception as e:
        print(f"  [ERROR] {serie_name} h={h}:\n{traceback.format_exc()}")
        return {"serie": serie_name, "h": h, "status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Metricas globales
# ---------------------------------------------------------------------------

def aggregate_metrics(all_results: list[dict]):
    rows = [r for r in all_results if r.get("status") == "ok"]
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    metrics_path = os.path.join(METRICS_DIR, "all_metrics.parquet")
    if os.path.exists(metrics_path):
        df_existing = pd.read_parquet(metrics_path)
        current_keys = set(zip(df_new["serie"].astype(str), df_new["h"].astype(int)))
        mask_keep = ~df_existing.apply(
            lambda r: (str(r["serie"]), int(r["h"])) in current_keys, axis=1
        )
        df_combined = pd.concat([df_existing[mask_keep], df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_parquet(metrics_path, index=False)
    print(f"[fit_soc_us] Metricas: {metrics_path}  ({len(df_combined)} filas)")
    print(df_new[["serie", "h", "ratio_SOC", "ratio_SOC_opt"]].to_string(index=False))


def rebuild_metrics_from_parquets() -> pd.DataFrame:
    rows = []
    for serie in US_SERIES_NAMES:
        safe = SAFE_NAMES.get(serie, serie)
        for h in HORIZONS:
            path = os.path.join(FORECASTS_DIR, f"{safe}_h{h}.parquet")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path)
                actual = df["actual"].values.astype(float)
                soc    = df["SOC"].values.astype(float)
                soc_opt = df["SOC_opt"].values.astype(float)
                ao     = df["AO"].values.astype(float)

                def _rmse(a, f):
                    mask = ~np.isnan(a) & ~np.isnan(f)
                    return float(np.sqrt(np.mean((a[mask] - f[mask]) ** 2))) if mask.sum() > 0 else np.nan

                r_ao  = _rmse(actual, ao)
                r_soc = _rmse(actual, soc)
                r_opt = _rmse(actual, soc_opt)
                rows.append({
                    "serie": serie, "h": h, "status": "ok",
                    "rmse_AO": r_ao, "rmse_SOC": r_soc, "rmse_SOC_opt": r_opt,
                    "ratio_SOC": r_soc / r_ao if r_ao and r_ao > 0 else np.nan,
                    "ratio_SOC_opt": r_opt / r_ao if r_ao and r_ao > 0 else np.nan,
                    "n_oos": len(df),
                })
                print(f"  [ok] {serie} h={h}  ratio={rows[-1]['ratio_SOC']:.3f}")
            except Exception as e:
                print(f"  [error] {serie} h={h}: {e}")
    if not rows:
        return pd.DataFrame()
    df_out = pd.DataFrame(rows)
    mp = os.path.join(METRICS_DIR, "all_metrics.parquet")
    df_out.to_parquet(mp, index=False)
    print(f"\n[rebuild] {len(df_out)} filas -> {mp}")
    return df_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ajusta modelos SOC para CPI/PCE EE.UU.")
    p.add_argument("--series",   type=str, default=None, choices=US_SERIES_NAMES + [None], nargs="?")
    p.add_argument("--horizon",  type=int, default=None, choices=[1, 6, 12])
    p.add_argument("--workers",  type=int, default=2)
    p.add_argument("--update-data",      action="store_true")
    p.add_argument("--rebuild-metrics",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("SOC - Forecasting US CPI/PCE")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if args.rebuild_metrics:
        rebuild_metrics_from_parquets()
        return

    if args.update_data:
        from soc_us.fetch_data import fetch_and_save_prices, fetch_and_save_macro
        fetch_and_save_prices()
        fetch_and_save_macro()

    df_prices = load_prices()
    df_macro  = load_macro()

    series_list  = [args.series] if args.series else US_SERIES_NAMES
    horizons_list = [args.horizon] if args.horizon else HORIZONS

    tasks = [(s, h, df_prices, df_macro) for s in series_list for h in horizons_list]
    print(f"[fit_soc_us] Tareas: {len(tasks)}  Workers: {args.workers}")

    all_results = []
    if args.workers <= 1 or len(tasks) == 1:
        for task in tasks:
            all_results.append(_process_one_safe(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_one_safe, task): task for task in tasks}
            for future in as_completed(futures):
                all_results.append(future.result())

    meta = {
        "run_date": datetime.now().isoformat(),
        "series": series_list, "horizons": horizons_list,
        "oos_start": OOS_START, "n_tasks": len(tasks),
        "n_ok": sum(1 for r in all_results if r.get("status") == "ok"),
        "n_error": sum(1 for r in all_results if r.get("status") == "error"),
    }
    with open(os.path.join(RESULTS_DIR, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    aggregate_metrics(all_results)

    print("=" * 60)
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"OK: {meta['n_ok']}  Errores: {meta['n_error']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
