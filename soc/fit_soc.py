"""
fit_soc.py – Script offline: corre todos los modelos SOC y guarda resultados en Parquet.

Uso:
    python soc/fit_soc.py                              # todos: 16 series × 3 horizontes
    python soc/fit_soc.py --series "Indice General"   # una sola serie
    python soc/fit_soc.py --horizon 1                  # un solo horizonte
    python soc/fit_soc.py --workers 4                  # paralelismo
    python soc/fit_soc.py --update-data                # re-fetch macro primero

Tiempo estimado: 3-6 horas en laptop moderno.
El dashboard solo lee Parquet, no recalcula.
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

# Asegurar que el directorio padre esté en path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from soc.config import (
    HORIZONS, INPC_PATH, MACRO_PATH, RESULTS_DIR, COMPONENTS_DIR,
    FORECASTS_DIR, METRICS_DIR, SAFE_NAMES, OOS_START, AO_WINDOW, COMPONENT_NAMES,
)
from soc.inflation_transform import compute_inflation
from soc.wavelet import decompose_series
from soc.expanding_window import evaluate_component
from soc.models import build_model_registry


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_inpc() -> pd.DataFrame:
    df = pd.read_csv(INPC_PATH, index_col="Fecha", parse_dates=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_macro() -> pd.DataFrame | None:
    if not os.path.exists(MACRO_PATH):
        print(f"[fit_soc] AVISO: {MACRO_PATH} no existe. Corriendo sin predictores macro.")
        return None
    df = pd.read_csv(MACRO_PATH, index_col="Fecha", parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# Tarea unitaria (serie × horizonte)
# ---------------------------------------------------------------------------

def _process_one(serie_name: str, h: int, df_inpc: pd.DataFrame, df_macro: pd.DataFrame | None) -> dict:
    """Procesa una combinación (serie, horizonte). Retorna dict de métricas."""
    safe = SAFE_NAMES.get(serie_name, serie_name.replace(" ", "_"))
    print(f"  [start] {serie_name} h={h}")
    t0 = time.time()

    # Nivel de precios
    price_level = df_inpc[serie_name].dropna()
    if len(price_level) < 50:
        print(f"  [skip]  {serie_name} h={h}: datos insuficientes ({len(price_level)})")
        return {"serie": serie_name, "h": h, "status": "skip"}

    # Inflación anualizada
    inflation = compute_inflation(price_level, h).dropna()
    if len(inflation) < 30:
        print(f"  [skip]  {serie_name} h={h}: inflación insuficiente ({len(inflation)})")
        return {"serie": serie_name, "h": h, "status": "skip"}

    # Descomposición wavelet
    try:
        df_comp = decompose_series(inflation)
    except Exception as e:
        print(f"  [error] {serie_name} h={h}: wavelet falló – {e}")
        return {"serie": serie_name, "h": h, "status": "error", "error": str(e)}

    # Guardar componentes
    comp_path = os.path.join(COMPONENTS_DIR, f"{safe}_h{h}.parquet")
    df_comp.to_parquet(comp_path, index=True)

    # X_macro alineado al índice de inflación
    if df_macro is not None:
        X_macro = df_macro.reindex(inflation.index)
    else:
        X_macro = None

    macro_cols = list(X_macro.columns) if X_macro is not None else []
    model_registry = build_model_registry(macro_cols)

    # Evaluar cada componente
    component_results = {}
    for comp_name in COMPONENT_NAMES:
        comp_series = df_comp[comp_name]
        try:
            res = evaluate_component(comp_series, X_macro, model_registry, h, OOS_START)
            component_results[comp_name] = res
        except Exception as e:
            print(f"  [warn]  {serie_name} h={h} {comp_name}: {e}")
            component_results[comp_name] = {}

    # Construir DataFrame de pronósticos SOC
    # Necesitamos fechas comunes para todos los componentes
    all_dates = None
    for comp_name in COMPONENT_NAMES:
        res = component_results.get(comp_name, {})
        if res and "dates" in res:
            dates_set = set(res["dates"])
            if all_dates is None:
                all_dates = dates_set
            else:
                all_dates = all_dates & dates_set

    if not all_dates:
        print(f"  [skip]  {serie_name} h={h}: sin fechas OOS comunes")
        return {"serie": serie_name, "h": h, "status": "skip"}

    sorted_dates = sorted(all_dates)

    # Reconstruir inflación realizada
    actual_inflation = inflation.reindex(sorted_dates)

    # SOC = suma de best_forecast de cada componente
    soc_arr = np.zeros(len(sorted_dates))
    soc_opt_arr = np.zeros(len(sorted_dates))  # excluye D1 (ruido)

    rows_data = {"date": sorted_dates, "actual": actual_inflation.values}

    for comp_name in COMPONENT_NAMES:
        res = component_results.get(comp_name, {})
        if not res or "dates" not in res:
            continue
        res_dates = res["dates"]
        date_to_idx = {d: i for i, d in enumerate(res_dates)}
        best_f = res.get("best_forecast", np.full(len(res_dates), np.nan))

        comp_forecast = np.array([
            best_f[date_to_idx[d]] if d in date_to_idx else np.nan
            for d in sorted_dates
        ])

        soc_arr += np.where(np.isnan(comp_forecast), 0, comp_forecast)
        if comp_name != "D1":
            soc_opt_arr += np.where(np.isnan(comp_forecast), 0, comp_forecast)

        rows_data[f"BEST_{comp_name}"] = comp_forecast
        rows_data[f"BESTMODEL_{comp_name}"] = res.get("best_model", "NA")

        # También guardar pronósticos por modelo para Tab 3
        for m_name, f_arr in res.get("forecasts", {}).items():
            arr_aligned = np.array([
                f_arr[date_to_idx[d]] if d in date_to_idx and date_to_idx[d] < len(f_arr) else np.nan
                for d in sorted_dates
            ])
            rows_data[f"{comp_name}_{m_name}"] = arr_aligned

    # AO benchmark
    inflation_ts = inflation.copy()
    ao_values = []
    for d in sorted_dates:
        # AO_t = media de los últimos AO_WINDOW observaciones antes de d
        past = inflation_ts[inflation_ts.index < d]
        if len(past) == 0:
            ao_values.append(np.nan)
        else:
            ao_values.append(float(past.iloc[-AO_WINDOW:].mean()))
    rows_data["AO"] = ao_values
    rows_data["SOC"] = soc_arr
    rows_data["SOC_opt"] = soc_opt_arr

    df_out = pd.DataFrame(rows_data)
    df_out = df_out.set_index("date")

    # Guardar pronósticos
    fc_path = os.path.join(FORECASTS_DIR, f"{safe}_h{h}.parquet")
    df_out.to_parquet(fc_path, index=True)

    elapsed = time.time() - t0
    print(f"  [done]  {serie_name} h={h} en {elapsed:.1f}s -> {fc_path}")

    # Métricas
    actual = df_out["actual"].values
    ao = df_out["AO"].values
    soc = df_out["SOC"].values
    soc_opt = df_out["SOC_opt"].values

    def rmse(a, f):
        mask = ~np.isnan(a) & ~np.isnan(f)
        if mask.sum() == 0:
            return np.nan
        return float(np.sqrt(np.mean((a[mask] - f[mask]) ** 2)))

    rmse_ao = rmse(actual, ao)
    rmse_soc = rmse(actual, soc)
    rmse_soc_opt = rmse(actual, soc_opt)

    return {
        "serie": serie_name,
        "h": h,
        "status": "ok",
        "rmse_AO": rmse_ao,
        "rmse_SOC": rmse_soc,
        "rmse_SOC_opt": rmse_soc_opt,
        "ratio_SOC": rmse_soc / rmse_ao if rmse_ao > 0 else np.nan,
        "ratio_SOC_opt": rmse_soc_opt / rmse_ao if rmse_ao > 0 else np.nan,
        "n_oos": len(sorted_dates),
        "elapsed_s": elapsed,
    }


def _process_one_safe(args):
    """Wrapper seguro para ProcessPoolExecutor."""
    serie_name, h, df_inpc, df_macro = args
    try:
        return _process_one(serie_name, h, df_inpc, df_macro)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  [ERROR] {serie_name} h={h}:\n{tb}")
        return {"serie": serie_name, "h": h, "status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Métricas globales
# ---------------------------------------------------------------------------

def aggregate_metrics(all_results: list[dict]):
    """
    Agrega métricas del run actual y las combina con resultados previos en
    all_metrics.parquet. Las filas de las mismas (serie, h) se reemplazan;
    el resto de series se conserva.
    """
    rows = [r for r in all_results if r.get("status") == "ok"]
    if not rows:
        print("[fit_soc] Sin métricas para agregar.")
        return
    df_new = pd.DataFrame(rows)
    metrics_path = os.path.join(METRICS_DIR, "all_metrics.parquet")

    # Combinar con métricas existentes sin sobreescribir otras series
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
    print(f"[fit_soc] Métricas guardadas: {metrics_path}  ({len(df_combined)} filas total)")
    print(df_new[["serie", "h", "ratio_SOC", "ratio_SOC_opt"]].to_string(index=False))
    return df_combined


def rebuild_metrics_from_parquets(verbose: bool = True) -> pd.DataFrame:
    """
    Reconstruye all_metrics.parquet leyendo todos los archivos de pronóstico
    existentes en FORECASTS_DIR. Util cuando el metrics file quedó incompleto
    por corridas parciales.

    Uso:  python -m soc.fit_soc --rebuild-metrics
    """
    from soc.config import INPC_SERIES_NAMES, SAFE_NAMES

    rows = []
    for serie_name in INPC_SERIES_NAMES:
        safe = SAFE_NAMES.get(serie_name, serie_name.replace(" ", "_"))
        for h in HORIZONS:
            path = os.path.join(FORECASTS_DIR, f"{safe}_h{h}.parquet")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path)
                actual  = df["actual"].values.astype(float)
                soc     = df["SOC"].values.astype(float)
                soc_opt = df["SOC_opt"].values.astype(float)
                ao      = df["AO"].values.astype(float)

                def _rmse(a, f):
                    mask = ~np.isnan(a) & ~np.isnan(f)
                    return float(np.sqrt(np.mean((a[mask] - f[mask]) ** 2))) if mask.sum() > 0 else np.nan

                r_ao  = _rmse(actual, ao)
                r_soc = _rmse(actual, soc)
                r_opt = _rmse(actual, soc_opt)

                rows.append({
                    "serie":         serie_name,
                    "h":             h,
                    "status":        "ok",
                    "rmse_AO":       r_ao,
                    "rmse_SOC":      r_soc,
                    "rmse_SOC_opt":  r_opt,
                    "ratio_SOC":     r_soc / r_ao if (r_ao and r_ao > 0) else np.nan,
                    "ratio_SOC_opt": r_opt / r_ao if (r_ao and r_ao > 0) else np.nan,
                    "n_oos":         len(df),
                })
                if verbose:
                    print(f"  [ok] {serie_name} h={h}  ratio={rows[-1]['ratio_SOC']:.3f}")
            except Exception as e:
                if verbose:
                    print(f"  [error] {serie_name} h={h}: {e}")

    if not rows:
        print("[rebuild] No se encontraron parquets de pronóstico.")
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    metrics_path = os.path.join(METRICS_DIR, "all_metrics.parquet")
    df_out.to_parquet(metrics_path, index=False)
    print(f"\n[rebuild] Listo: {len(df_out)} filas -> {metrics_path}")
    print(df_out[["serie", "h", "ratio_SOC"]].to_string(index=False))
    return df_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ajusta modelos SOC para INPC México")
    p.add_argument("--series", type=str, default=None, help="Nombre de serie específica")
    p.add_argument("--horizon", type=int, default=None, choices=[1, 6, 12],
                   help="Horizonte específico")
    p.add_argument("--workers", type=int, default=2, help="Procesos paralelos")
    p.add_argument("--update-data", action="store_true",
                   help="Re-descarga macro_data.csv antes de correr")
    p.add_argument("--rebuild-metrics", action="store_true",
                   help="Reconstruye all_metrics.parquet desde los parquets existentes (sin re-estimar modelos)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SOC - Forecasting INPC Mexico")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Atajo: solo reconstruir métricas sin re-estimar
    if args.rebuild_metrics:
        print("[fit_soc] Reconstruyendo all_metrics.parquet desde parquets existentes...")
        rebuild_metrics_from_parquets()
        print("[fit_soc] Listo. Reinicia el dashboard para ver los cambios.")
        return

    # Opcional: actualizar macro
    if args.update_data:
        print("[fit_soc] Actualizando macro_data.csv...")
        from soc.fetch_macro import fetch_and_save_macro
        fetch_and_save_macro()

    # Cargar datos
    print("[fit_soc] Cargando datos INPC...")
    df_inpc = load_inpc()
    df_macro = load_macro()

    # Selección de series y horizontes
    all_series = list(df_inpc.columns)
    if args.series:
        if args.series not in all_series:
            # Buscar match parcial
            matches = [s for s in all_series if args.series.lower() in s.lower()]
            if matches:
                series_list = matches
                print(f"[fit_soc] Series encontradas: {series_list}")
            else:
                print(f"[fit_soc] ERROR: serie '{args.series}' no encontrada.")
                print(f"  Series disponibles: {all_series}")
                sys.exit(1)
        else:
            series_list = [args.series]
    else:
        series_list = all_series

    horizons_list = [args.horizon] if args.horizon else HORIZONS

    # Construir tareas
    tasks = [(s, h, df_inpc, df_macro) for s in series_list for h in horizons_list]
    print(f"[fit_soc] Tareas: {len(tasks)} ({len(series_list)} series x {len(horizons_list)} horizontes)")
    print(f"[fit_soc] Workers: {args.workers}")

    all_results = []

    if args.workers <= 1 or len(tasks) == 1:
        # Ejecución secuencial
        for task in tasks:
            result = _process_one_safe(task)
            all_results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_one_safe, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)

    # Guardar metadatos
    meta = {
        "run_date": datetime.now().isoformat(),
        "series": series_list,
        "horizons": horizons_list,
        "oos_start": OOS_START,
        "n_tasks": len(tasks),
        "n_ok": sum(1 for r in all_results if r.get("status") == "ok"),
        "n_error": sum(1 for r in all_results if r.get("status") == "error"),
    }
    meta_path = os.path.join(RESULTS_DIR, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[fit_soc] Metadatos: {meta_path}")

    # Métricas globales
    aggregate_metrics(all_results)

    print("\n" + "=" * 60)
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ok = meta["n_ok"]
    err_count = meta["n_error"]
    print(f"Completado: {ok} OK, {err_count} errores")
    print("=" * 60)


if __name__ == "__main__":
    main()
