"""
fetch_macro.py – Descarga predictores macro de Banxico + FRED para el modelo SOC.

Reutiliza exactamente el mismo patrón de data_loader.py:
  - mismo token BANXICO_TOKEN
  - misma estructura urllib.request
  - mismo parseo JSON data["bmx"]["series"]

Columnas finales de macro_data.csv:
  M2_REAL, CETES28, TIIE28, TMS, MBONO10Y, USDMXN, IPC_BMV_RET, ENERGY, WTI
"""
import json
import os
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from .config import MACRO_BANXICO, MACRO_FRED, MACRO_PATH, INPC_PATH

# ---------------------------------------------------------------------------
# Token Banxico (igual que data_loader.py)
# ---------------------------------------------------------------------------
BANXICO_TOKEN = os.environ.get("BANXICO_TOKEN", "")
if not BANXICO_TOKEN:
    _env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                if _line.strip().startswith("BANXICO_TOKEN="):
                    BANXICO_TOKEN = _line.strip().split("=", 1)[1].strip()
                    break


def _banxico_request(series_ids: list[str]) -> pd.DataFrame:
    """Descarga series de Banxico y retorna DataFrame mensual."""
    all_ids = ",".join(series_ids)
    url = (
        f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{all_ids}"
        f"/datos/1990-01-01/{datetime.now().strftime('%Y-%m-%d')}?token={BANXICO_TOKEN}"
    )
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())

    frames = {}
    for serie_data in data["bmx"]["series"]:
        serie_id = serie_data["idSerie"]
        name = MACRO_BANXICO.get(serie_id, serie_id)
        obs = serie_data["datos"]
        df_s = pd.DataFrame(obs)
        df_s["fecha"] = pd.to_datetime(df_s["fecha"], format="%d/%m/%Y")
        df_s["dato"] = pd.to_numeric(df_s["dato"].str.replace(",", ""), errors="coerce")
        df_s = df_s.set_index("fecha")
        frames[name] = df_s["dato"]

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    return df


def fetch_banxico_macro() -> pd.DataFrame:
    """Descarga M2, CETES28, TIIE28, MBONO10Y, USDMXN, IPC_BMV de Banxico."""
    return _banxico_request(list(MACRO_BANXICO.keys()))


def fetch_fred_macro() -> pd.Series | None:
    """
    Descarga WTI de FRED.
    Requiere FRED_API_KEY en entorno o .env.
    Retorna Series mensual o None si no hay clave / falla.
    """
    fred_key = os.environ.get("FRED_API_KEY", "")
    if not fred_key:
        _env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(_env_path):
            with open(_env_path) as _f:
                for _line in _f:
                    if _line.strip().startswith("FRED_API_KEY="):
                        fred_key = _line.strip().split("=", 1)[1].strip()
                        break

    if not fred_key:
        print("[fetch_macro] Sin FRED_API_KEY – WTI se omitirá o usará CSV manual.")
        return None

    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_key)
        wti = fred.get_series("DCOILWTICO", observation_start="1990-01-01")
        wti.index = pd.to_datetime(wti.index)
        wti_monthly = wti.resample("MS").mean()
        wti_monthly.name = "WTI"
        return wti_monthly
    except Exception as e:
        print(f"[fetch_macro] Error descargando WTI de FRED: {e}")
        return None


def compute_derived_series(df_banxico: pd.DataFrame, df_inpc: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula series derivadas a partir de los datos crudos de Banxico.

    Columnas generadas:
      M2_REAL     = M2 / INPC_General * 100
      TMS         = MBONO10Y - CETES28
      IPC_BMV_RET = log(IPC_t / IPC_{t-1}) * 1200
      ENERGY      = columna Energeticos de inpc_data.csv
    """
    df = df_banxico.copy()

    # INPC General mensual (primer columna del CSV)
    inpc_general = df_inpc["Indice General"].dropna()
    inpc_general = inpc_general.resample("MS").last()

    # M2 real
    if "M2" in df.columns:
        m2_aligned = df["M2"].reindex(inpc_general.index)
        inpc_aligned = inpc_general.reindex(m2_aligned.index)
        df["M2_REAL"] = m2_aligned / inpc_aligned * 100

    # Term spread
    if "MBONO10Y" in df.columns and "CETES28" in df.columns:
        df["TMS"] = df["MBONO10Y"] - df["CETES28"]

    # IPC BMV log return annualized
    if "IPC_BMV" in df.columns:
        df["IPC_BMV_RET"] = np.log(df["IPC_BMV"] / df["IPC_BMV"].shift(1)) * 1200

    # ENERGY: nivel de Energeticos del INPC
    energy_col = None
    for c in df_inpc.columns:
        if "Energeticos" in c and "Tarifas" not in c:
            energy_col = c
            break
    if energy_col is not None:
        energy = df_inpc[energy_col].dropna()
        energy_monthly = energy.resample("MS").last()
        df["ENERGY"] = energy_monthly.reindex(df.index)

    return df


def fetch_and_save_macro(verbose: bool = True) -> pd.DataFrame:
    """
    Orquesta la descarga completa de macro predictores y guarda macro_data.csv.

    Columnas finales: M2_REAL, CETES28, TIIE28, TMS, MBONO10Y, USDMXN, IPC_BMV_RET, ENERGY, WTI
    """
    if verbose:
        print("[fetch_macro] Descargando datos de Banxico...")
    df_banxico = fetch_banxico_macro()

    # Cargar INPC para derivadas
    df_inpc = pd.read_csv(INPC_PATH, index_col="Fecha", parse_dates=True)
    # Strip whitespace from columns
    df_inpc.columns = [c.strip() for c in df_inpc.columns]

    if verbose:
        print("[fetch_macro] Calculando series derivadas...")
    df_banxico.index = pd.to_datetime(df_banxico.index)
    df_banxico = df_banxico.resample("MS").last()
    df_macro = compute_derived_series(df_banxico, df_inpc)

    # WTI
    if verbose:
        print("[fetch_macro] Descargando WTI de FRED...")
    wti = fetch_fred_macro()
    if wti is not None:
        wti.index = pd.to_datetime(wti.index)
        df_macro["WTI"] = wti.reindex(df_macro.index)

    # Seleccionar columnas finales
    final_cols = [
        "M2_REAL", "CETES28", "TIIE28", "TMS", "MBONO10Y",
        "USDMXN", "IPC_BMV_RET", "ENERGY", "WTI",
    ]
    available = [c for c in final_cols if c in df_macro.columns]
    df_out = df_macro[available].copy()
    df_out.index.name = "Fecha"

    df_out.to_csv(MACRO_PATH)
    if verbose:
        print(f"[fetch_macro] Guardado en {MACRO_PATH}")
        print(f"  Columnas: {list(df_out.columns)}")
        print(f"  Rango: {df_out.index.min()} – {df_out.index.max()}")
        print(f"  Filas: {len(df_out)}")

    return df_out


# ---------------------------------------------------------------------------
# Mapeo Banxico ID -> nombre con indentacion jerárquica (igual que data_loader.py)
# Se usa para guardar inpc_data.csv compatible con ambas apps.
# ---------------------------------------------------------------------------
_INPC_ID_TO_INDENTED = {
    "SP1":     "Indice General",
    "SP74625": "Subyacente",
    "SP74626": "  Mercancias",
    "SP66540": "    Alimentos Bebidas y Tabaco",
    "SP74627": "    Mercancias no Alimenticias",
    "SP74628": "  Servicios",
    "SP66542": "    Vivienda",
    "SP56339": "    Educacion (colegiaturas)",
    "SP74629": "    Otros Servicios",
    "SP74630": "No Subyacente",
    "SP56337": "  Agropecuarios",
    "SP56385": "    Frutas y Verduras",
    "SP56386": "    Pecuarios",
    "SP74631": "  Energeticos y Tarifas Aut. Gob.",
    "SP56373": "    Energeticos",
    "SP74640": "    Tarifas Aut. por el Gobierno",
}
_INPC_COLUMN_ORDER = list(_INPC_ID_TO_INDENTED.values())


def fetch_and_save_inpc(verbose: bool = True) -> pd.DataFrame:
    """
    Descarga las 16 series del INPC mensual desde Banxico y guarda inpc_data.csv.

    El archivo se guarda con la misma indentacion jerarquica que usa el Streamlit
    principal (data_loader.py), por lo que ambas apps comparten el mismo CSV.

    Requiere BANXICO_TOKEN en entorno o .env
    """
    if not BANXICO_TOKEN:
        raise RuntimeError(
            "BANXICO_TOKEN no configurado. "
            "Agrega BANXICO_TOKEN=<token> a dashboard/.env o como variable de entorno."
        )

    all_ids = ",".join(_INPC_ID_TO_INDENTED.keys())
    url = (
        f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{all_ids}"
        f"/datos/1969-01-01/{datetime.now().strftime('%Y-%m-%d')}?token={BANXICO_TOKEN}"
    )
    if verbose:
        print("[fetch_inpc] Descargando INPC desde Banxico...")

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())

    frames = {}
    for serie_data in data["bmx"]["series"]:
        serie_id = serie_data["idSerie"]
        indented_name = _INPC_ID_TO_INDENTED.get(serie_id, serie_id)
        obs = serie_data.get("datos", [])
        if not obs:
            if verbose:
                print(f"  [warn] Sin datos para {serie_id}")
            continue
        df_s = pd.DataFrame(obs)
        df_s["fecha"] = pd.to_datetime(df_s["fecha"], format="%d/%m/%Y")
        df_s["dato"] = pd.to_numeric(df_s["dato"].str.replace(",", ""), errors="coerce")
        df_s = df_s.set_index("fecha")
        frames[indented_name] = df_s["dato"]

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    # Reordenar columnas en orden jerarquico y resamplear a inicio de mes
    available = [c for c in _INPC_COLUMN_ORDER if c in df.columns]
    df = df[available].resample("MS").last()
    df.index.name = "Fecha"

    df.to_csv(INPC_PATH)
    if verbose:
        print(f"[fetch_inpc] Guardado en {INPC_PATH}")
        print(f"  Series:  {len(df.columns)}")
        print(f"  Rango:   {df.index.min().strftime('%Y-%m')} – {df.index.max().strftime('%Y-%m')}")
        print(f"  Filas:   {len(df)}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch de datos para el modulo SOC")
    parser.add_argument("--inpc",  action="store_true", help="Descargar inpc_data.csv desde Banxico")
    parser.add_argument("--macro", action="store_true", help="Descargar macro_data.csv (Banxico + FRED)")
    parser.add_argument("--all",   action="store_true", help="Descargar INPC y macro (flujo completo)")
    args = parser.parse_args()

    if args.all or args.inpc:
        fetch_and_save_inpc()
    if args.all or args.macro:
        fetch_and_save_macro()
    if not (args.all or args.inpc or args.macro):
        # Comportamiento original: solo macro
        fetch_and_save_macro()
