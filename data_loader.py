import pandas as pd
import json
import urllib.request
import os
from datetime import datetime

BANXICO_TOKEN = os.environ.get("BANXICO_TOKEN", "")
if not BANXICO_TOKEN:
    try:
        import streamlit as st
        BANXICO_TOKEN = st.secrets.get("BANXICO_TOKEN", "")
    except Exception:
        pass
if not BANXICO_TOKEN:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("BANXICO_TOKEN="):
                    BANXICO_TOKEN = line.strip().split("=", 1)[1]

SERIES_MAP = {
    "SP1": "Indice General",
    "SP74625": "Subyacente",
    "SP74626": "Mercancias",
    "SP66540": "Alimentos Bebidas y Tabaco",
    "SP74627": "Mercancias no Alimenticias",
    "SP74628": "Servicios",
    "SP66542": "Vivienda",
    "SP56339": "Educacion (colegiaturas)",
    "SP74629": "Otros Servicios",
    "SP74630": "No Subyacente",
    "SP56337": "Agropecuarios",
    "SP56385": "Frutas y Verduras",
    "SP56386": "Pecuarios",
    "SP74631": "Energeticos y Tarifas Aut. Gob.",
    "SP56373": "Energeticos",
    "SP74640": "Tarifas Aut. por el Gobierno",
}

COLUMN_ORDER = [
    "Indice General",
    "Subyacente",
    "  Mercancias",
    "    Alimentos Bebidas y Tabaco",
    "    Mercancias no Alimenticias",
    "  Servicios",
    "    Vivienda",
    "    Educacion (colegiaturas)",
    "    Otros Servicios",
    "No Subyacente",
    "  Agropecuarios",
    "    Frutas y Verduras",
    "    Pecuarios",
    "  Energeticos y Tarifas Aut. Gob.",
    "    Energeticos",
    "    Tarifas Aut. por el Gobierno",
]

# Ponderadores INPC 2024 (vigentes desde 2a quincena julio 2024)
# Fuente: INEGI, Ponderadores_INPC 2024_Nacional_Final.xlsx
PONDERADORES = {
    "Indice General": 100.0,
    "Subyacente": 76.7415,
    "  Mercancias": 37.5338,
    "    Alimentos Bebidas y Tabaco": 17.2148,
    "    Mercancias no Alimenticias": 20.3190,
    "  Servicios": 39.2077,
    "    Vivienda": 18.0550,
    "    Educacion (colegiaturas)": 2.5207,
    "    Otros Servicios": 18.6321,
    "No Subyacente": 23.2585,
    "  Agropecuarios": 10.6577,
    "    Frutas y Verduras": 4.7789,
    "    Pecuarios": 5.8788,
    "  Energeticos y Tarifas Aut. Gob.": 12.6008,
    "    Energeticos": 8.0458,
    "    Tarifas Aut. por el Gobierno": 4.5550,
}

# Jerarquia: padre -> lista de hijos inmediatos
JERARQUIA = {
    "Indice General": ["Subyacente", "No Subyacente"],
    "Subyacente": ["  Mercancias", "  Servicios"],
    "  Mercancias": ["    Alimentos Bebidas y Tabaco", "    Mercancias no Alimenticias"],
    "  Servicios": ["    Vivienda", "    Educacion (colegiaturas)", "    Otros Servicios"],
    "No Subyacente": ["  Agropecuarios", "  Energeticos y Tarifas Aut. Gob."],
    "  Agropecuarios": ["    Frutas y Verduras", "    Pecuarios"],
    "  Energeticos y Tarifas Aut. Gob.": ["    Energeticos", "    Tarifas Aut. por el Gobierno"],
}

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "inpc_data.csv")

# --- Series quincenales ---
SERIES_MAP_Q = {
    "SP8664": "Indice General",
    "SP74632": "Subyacente",
    "SP74633": "Mercancias",
    "SP66536": "Alimentos Bebidas y Tabaco",
    "SP74634": "Mercancias no Alimenticias",
    "SP74635": "Servicios",
    "SP66538": "Vivienda",
    "SP56384": "Educacion (colegiaturas)",
    "SP74636": "Otros Servicios",
    "SP74637": "No Subyacente",
    "SP56378": "Agropecuarios",
    "SP56379": "Frutas y Verduras",
    "SP56380": "Pecuarios",
    "SP74638": "Energeticos y Tarifas Aut. Gob.",
    "SP56382": "Energeticos",
    "SP74639": "Tarifas Aut. por el Gobierno",
}

DATA_PATH_Q = os.path.join(os.path.dirname(__file__), "data", "inpc_data_q.csv")


def fetch_from_banxico():
    """Descarga todas las series del API de Banxico y guarda CSV."""
    all_ids = ",".join(SERIES_MAP.keys())
    url = (f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{all_ids}"
           f"/datos/1969-01-01/{datetime.now().strftime('%Y-%m-%d')}?token={BANXICO_TOKEN}")
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())

    series_list = data["bmx"]["series"]
    frames = {}
    for serie_data in series_list:
        serie_id = serie_data["idSerie"]
        name = SERIES_MAP.get(serie_id, serie_id)
        obs = serie_data["datos"]
        df_s = pd.DataFrame(obs)
        df_s["fecha"] = pd.to_datetime(df_s["fecha"], format="%d/%m/%Y")
        df_s["dato"] = pd.to_numeric(df_s["dato"].str.replace(",", ""), errors="coerce")
        df_s = df_s.set_index("fecha")
        frames[name] = df_s["dato"]

    df = pd.DataFrame(frames)
    # Rename to hierarchical names and reorder
    rename = {plain: indented for plain, indented in
              zip([v for v in SERIES_MAP.values()],
                  [c.strip() for c in COLUMN_ORDER])}
    # Reverse: map plain name -> indented name
    rename = {}
    for col_indented in COLUMN_ORDER:
        plain = col_indented.strip()
        if plain in df.columns:
            rename[plain] = col_indented
    df = df.rename(columns=rename)
    df = df[COLUMN_ORDER]
    df.index.name = "Fecha"
    df.to_csv(DATA_PATH)
    return df


def load_data():
    """Carga datos del CSV local."""
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH, index_col="Fecha", parse_dates=True)
    # Ensure column order
    ordered = [c for c in COLUMN_ORDER if c in df.columns]
    df = df[ordered]
    return df


def fetch_from_banxico_q():
    """Descarga todas las series quincenales del API de Banxico y guarda CSV."""
    all_ids = ",".join(SERIES_MAP_Q.keys())
    url = (f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{all_ids}"
           f"/datos/1988-01-01/{datetime.now().strftime('%Y-%m-%d')}?token={BANXICO_TOKEN}")
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())

    series_list = data["bmx"]["series"]
    frames = {}
    for serie_data in series_list:
        serie_id = serie_data["idSerie"]
        name = SERIES_MAP_Q.get(serie_id, serie_id)
        obs = serie_data["datos"]
        df_s = pd.DataFrame(obs)
        df_s["fecha"] = pd.to_datetime(df_s["fecha"], format="%d/%m/%Y")
        df_s["dato"] = pd.to_numeric(df_s["dato"].str.replace(",", ""), errors="coerce")
        df_s = df_s.set_index("fecha")
        frames[name] = df_s["dato"]

    df = pd.DataFrame(frames)
    # Rename to hierarchical names and reorder
    rename = {}
    for col_indented in COLUMN_ORDER:
        plain = col_indented.strip()
        if plain in df.columns:
            rename[plain] = col_indented
    df = df.rename(columns=rename)
    df = df[COLUMN_ORDER]
    df.index.name = "Fecha"
    df.to_csv(DATA_PATH_Q)
    return df


def load_data_q():
    """Carga datos quincenales del CSV local."""
    if not os.path.exists(DATA_PATH_Q):
        return None
    df = pd.read_csv(DATA_PATH_Q, index_col="Fecha", parse_dates=True)
    # Ensure column order
    ordered = [c for c in COLUMN_ORDER if c in df.columns]
    df = df[ordered]
    return df
