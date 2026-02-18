"""
Pre-calcula los órdenes ARIMA y SARIMA para cada serie del INPC
usando los últimos N períodos. Guarda resultados en data/model_orders.json.

Ejecutar manualmente cuando se quiera recalibrar:
    python fit_orders.py
    python fit_orders.py --quincenal
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from pmdarima import auto_arima

from data_loader import load_data, load_data_q

BASE_DIR = os.path.dirname(__file__)


def fit_series(name, values, index_str, freq, m):
    """Ajusta auto_arima (ARIMA y SARIMA) sobre una serie y devuelve los órdenes."""
    idx = pd.DatetimeIndex(index_str)
    if freq is not None:
        idx.freq = freq
    s = pd.Series(values, index=idx)

    common_kw = dict(suppress_warnings=True, error_action="ignore",
                     stepwise=True, max_p=3, max_q=3)

    arima = auto_arima(s, seasonal=False, **common_kw)
    sarima = auto_arima(s, seasonal=True, m=m,
                        max_P=2, max_Q=2, max_D=1, **common_kw)

    return {
        "arima_order": list(arima.order),
        "arima_intercept": bool(arima.with_intercept),
        "sarima_order": list(sarima.order),
        "sarima_seasonal_order": list(sarima.seasonal_order),
        "sarima_intercept": bool(sarima.with_intercept),
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-calcula órdenes ARIMA/SARIMA")
    parser.add_argument("--quincenal", action="store_true",
                        help="Usar series quincenales en lugar de mensuales")
    args = parser.parse_args()

    if args.quincenal:
        df = load_data_q()
        n_periods = 96
        m = 24
        freq = None  # quincenal (1o y 16) no tiene freq estandar en pandas
        orders_path = os.path.join(BASE_DIR, "data", "model_orders_q.json")
        label = "quincenas"
    else:
        df = load_data()
        n_periods = 48
        m = 12
        freq = "MS"
        orders_path = os.path.join(BASE_DIR, "data", "model_orders.json")
        label = "meses"

    if df is None:
        print("No hay datos. Ejecuta primero la descarga desde el dashboard.")
        return

    results = {}
    series_names = df.columns.tolist()
    print(f"Ajustando {len(series_names)} series (últimos {n_periods} {label}, m={m})...\n")

    start = time.time()

    def _fit(name):
        serie = df[name].dropna().iloc[-n_periods:]
        if freq is not None:
            serie.index.freq = freq
        idx_str = serie.index.astype(str).tolist()
        return name, fit_series(name, serie.values, idx_str, freq, m)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fit, name): name for name in series_names}
        for fut in as_completed(futures):
            name, orders = fut.result()
            results[name] = orders
            a = tuple(orders["arima_order"])
            ai = orders["arima_intercept"]
            s = tuple(orders["sarima_order"])
            ss = tuple(orders["sarima_seasonal_order"])
            si = orders["sarima_intercept"]
            print(f"  {name:45s}  ARIMA{a} int={ai}  SARIMA{s}x{ss} int={si}")

    elapsed = time.time() - start
    print(f"\nListo en {elapsed:.1f}s")

    with open(orders_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Guardado en {orders_path}")


if __name__ == "__main__":
    main()
