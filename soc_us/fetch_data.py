"""
fetch_data.py – Descarga CPI, PCE y predictores macro para el modulo soc_us.

Fuentes:
  - FRED (via fredapi si disponible, sino urllib)
  - Caldara, Iacoviello y Yu (2025) IFDP 1407 para shortage indices

Uso:
    python -m soc_us.fetch_data --prices      # CPI + PCE
    python -m soc_us.fetch_data --macro       # predictores macro
    python -m soc_us.fetch_data --shortage    # indices de escasez
    python -m soc_us.fetch_data --all         # todo
"""
import json
import os
import urllib.request
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from .config import (
    DATA_DIR, US_DATA_PATH, US_MACRO_PATH,
    PRICE_FRED, MACRO_FRED_SERIES, JOLTS_FRED_SERIES,
    MACRO_FINAL_COLS, SHORTAGE_URL, SHORTAGE_COL_MAP,
)

# ---------------------------------------------------------------------------
# FRED helper
# ---------------------------------------------------------------------------
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
if not FRED_API_KEY:
    _env = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(_env):
        with open(_env) as _f:
            for _line in _f:
                if _line.strip().startswith("FRED_API_KEY="):
                    FRED_API_KEY = _line.strip().split("=", 1)[1].strip()
                    break


def _fred_get(series_id: str, start: str = "1970-01-01") -> pd.Series:
    """Descarga una serie de FRED. Usa fredapi si hay clave, sino API publica."""
    if FRED_API_KEY:
        try:
            from fredapi import Fred
            fred = Fred(api_key=FRED_API_KEY)
            s = fred.get_series(series_id, observation_start=start)
            s.index = pd.to_datetime(s.index)
            s.name = series_id
            return s
        except Exception as e:
            print(f"  [warn] fredapi fallo para {series_id}: {e}. Intentando API publica.")

    # API publica de FRED (sin clave, limite de uso)
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&observation_start={start}"
        f"&file_type=json"
        + (f"&api_key={FRED_API_KEY}" if FRED_API_KEY else "")
    )
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        obs = data.get("observations", [])
        if not obs:
            return pd.Series(name=series_id, dtype=float)
        dates = pd.to_datetime([o["date"] for o in obs])
        vals = pd.to_numeric([o["value"] for o in obs], errors="coerce")
        s = pd.Series(vals.values, index=dates, name=series_id)
        return s
    except Exception as e:
        print(f"  [error] No se pudo descargar {series_id}: {e}")
        return pd.Series(name=series_id, dtype=float)


def _to_monthly(s: pd.Series) -> pd.Series:
    """Resamplea a inicio de mes, tomando el ultimo valor del mes."""
    if s.empty:
        return s
    s.index = pd.to_datetime(s.index)
    return s.resample("MS").last()


# ---------------------------------------------------------------------------
# Niveles de precios: CPI y PCE
# ---------------------------------------------------------------------------

def fetch_and_save_prices(verbose: bool = True) -> pd.DataFrame:
    """Descarga CPIAUCSL y PCEPI de FRED y guarda us_cpi_pce.csv."""
    if verbose:
        print("[fetch_data] Descargando CPI y PCE de FRED...")
    frames = {}
    for fred_id, col_name in PRICE_FRED.items():
        if verbose:
            print(f"  {fred_id} -> {col_name}")
        s = _fred_get(fred_id, start="1947-01-01")
        if not s.empty:
            frames[col_name] = _to_monthly(s)

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df.to_csv(US_DATA_PATH)
    if verbose:
        print(f"  Guardado: {US_DATA_PATH}  ({df.index.min().strftime('%Y-%m')} – {df.index.max().strftime('%Y-%m')})")
    return df


# ---------------------------------------------------------------------------
# Shortage indices – Caldara, Iacoviello y Yu (2025)
# ---------------------------------------------------------------------------

def fetch_shortage_indices(verbose: bool = True) -> pd.DataFrame | None:
    """
    Descarga el CSV de shortage indices de Caldara, Iacoviello y Yu (2025)
    desde matteoiacoviello.com y retorna DataFrame mensual.

    Columnas fuente: index_all_shortage, industry_shortage, energy_shortage,
                     food_shortage, labor_shortage, index_usa
    Formato de fecha: '1900m1', '1900m12', etc.
    """
    if verbose:
        print(f"[fetch_data] Descargando shortage indices de {SHORTAGE_URL}")
    local_path = os.path.join(DATA_DIR, "shortage_index_web.csv")

    try:
        req = urllib.request.Request(SHORTAGE_URL)
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = resp.read()
        with open(local_path, "wb") as f:
            f.write(content)
        if verbose:
            print(f"  Descargado: {local_path} ({len(content)//1024} KB)")
    except Exception as e:
        print(f"  [error] No se pudo descargar shortage data: {e}")
        if not os.path.exists(local_path):
            return None
        if verbose:
            print(f"  Usando copia local: {local_path}")

    try:
        df_raw = pd.read_csv(local_path)

        # Parsear fecha formato "1900m1" -> Timestamp
        def _parse_ym(s: str) -> pd.Timestamp:
            parts = str(s).split("m")
            year, month = int(parts[0]), int(parts[1])
            return pd.Timestamp(year=year, month=month, day=1)

        df_raw["date"] = df_raw["date"].apply(_parse_ym)
        df_raw = df_raw.set_index("date").sort_index()

        # Seleccionar y renombrar columnas
        rename = {orig: dest for orig, dest in SHORTAGE_COL_MAP.items()
                  if orig in df_raw.columns}
        df_sh = df_raw[list(rename.keys())].rename(columns=rename)
        df_sh = df_sh.apply(pd.to_numeric, errors="coerce")
        df_sh.index = pd.to_datetime(df_sh.index)

        if verbose:
            print(f"  Shortage cols: {list(df_sh.columns)}")
            print(f"  Rango: {df_sh.index.min().strftime('%Y-%m')} – "
                  f"{df_sh.index.max().strftime('%Y-%m')}")
        return df_sh

    except Exception as e:
        print(f"  [error] Parseo del CSV fallo: {e}")
        return None


# ---------------------------------------------------------------------------
# Predictores macro
# ---------------------------------------------------------------------------

def fetch_and_save_macro(verbose: bool = True) -> pd.DataFrame:
    """
    Descarga predictores macro de FRED, calcula derivados, agrega shortage
    indices y guarda us_macro_data.csv.
    """
    if verbose:
        print("[fetch_data] Descargando macro predictores de FRED...")

    raw = {}
    all_series = {**MACRO_FRED_SERIES, **JOLTS_FRED_SERIES}
    for fred_id, col_name in all_series.items():
        if verbose:
            print(f"  {fred_id} -> {col_name}")
        s = _fred_get(fred_id, start="1970-01-01")
        if not s.empty:
            raw[col_name] = _to_monthly(s)

    df = pd.DataFrame(raw)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # --- Derivados ---
    # M2_REAL = M2 / CPI_idx * 100
    if "M2" in df.columns and "CPI_IDX" in df.columns:
        cpi_aligned = df["CPI_IDX"].reindex(df.index)
        df["M2_REAL"] = df["M2"] / cpi_aligned * 100

    # TMS = GS10 - TBL
    if "GS10" in df.columns and "TBL" in df.columns:
        df["TMS"] = df["GS10"] - df["TBL"]

    # ENERGY = cambio YoY del indice de energia CPI (%)
    if "ENERGY_IDX" in df.columns:
        df["ENERGY"] = (df["ENERGY_IDX"] / df["ENERGY_IDX"].shift(12) - 1.0) * 100.0

    # OIL = retorno logaritmico mensual del WTI * 100
    if "WTI_RAW" in df.columns:
        df["OIL"] = np.log(df["WTI_RAW"] / df["WTI_RAW"].shift(1)) * 100.0

    # JWG = (Employed + JobOpenings - LaborForce) en miles
    if all(c in df.columns for c in ["EMPLOYED", "JOBOPENINGS", "LABORFORCE"]):
        df["JWG"] = df["EMPLOYED"] + df["JOBOPENINGS"] - df["LABORFORCE"]

    # --- Shortage indices ---
    df_sh = fetch_shortage_indices(verbose=verbose)
    if df_sh is not None:
        for col in df_sh.columns:
            df[col] = df_sh[col].reindex(df.index)

    # --- Seleccionar columnas finales ---
    available = [c for c in MACRO_FINAL_COLS if c in df.columns]
    df_out = df[available].copy()
    df_out.index.name = "Date"
    df_out.to_csv(US_MACRO_PATH)

    if verbose:
        print(f"[fetch_data] Macro guardada: {US_MACRO_PATH}")
        print(f"  Columnas: {list(df_out.columns)}")
        print(f"  Rango: {df_out.index.min().strftime('%Y-%m')} – {df_out.index.max().strftime('%Y-%m')}")
    return df_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fetch de datos US para soc_us")
    p.add_argument("--prices",   action="store_true", help="Descargar CPI y PCE")
    p.add_argument("--macro",    action="store_true", help="Descargar predictores macro")
    p.add_argument("--shortage", action="store_true", help="Solo shortage indices")
    p.add_argument("--all",      action="store_true", help="Todo")
    args = p.parse_args()

    if args.all or args.prices:
        fetch_and_save_prices()
    if args.all or args.macro:
        fetch_and_save_macro()
    if args.shortage and not args.macro and not args.all:
        df_sh = fetch_shortage_indices()
        if df_sh is not None:
            out = os.path.join(DATA_DIR, "us_shortage_data.csv")
            df_sh.to_csv(out)
            print(f"Shortage data guardada: {out}")
    if not any([args.all, args.prices, args.macro, args.shortage]):
        fetch_and_save_prices()
        fetch_and_save_macro()
