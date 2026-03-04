"""
config.py – Constantes y rutas para el modulo soc_us (CPI/PCE de EE.UU.).
"""
import os
import re

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dashboard/
DATA_DIR = os.path.join(_DIR, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "soc_us_results")
COMPONENTS_DIR = os.path.join(RESULTS_DIR, "components")
FORECASTS_DIR = os.path.join(RESULTS_DIR, "forecasts")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
US_DATA_PATH = os.path.join(DATA_DIR, "us_cpi_pce.csv")
US_MACRO_PATH = os.path.join(DATA_DIR, "us_macro_data.csv")

for _d in [RESULTS_DIR, COMPONENTS_DIR, FORECASTS_DIR, METRICS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Series a pronosticar
# ---------------------------------------------------------------------------
US_SERIES_NAMES = ["CPI", "PCE", "CORE_CPI", "CORE_PCE"]


def _safe_name(name: str) -> str:
    nfkd = name
    safe = re.sub(r"[^A-Za-z0-9]+", "_", nfkd.strip())
    return safe.strip("_")


SAFE_NAMES: dict[str, str] = {name: _safe_name(name) for name in US_SERIES_NAMES}

# ---------------------------------------------------------------------------
# FRED – niveles de precios
# ---------------------------------------------------------------------------
PRICE_FRED = {
    "CPIAUCSL": "CPI",      # Consumer Price Index, All Urban, Seasonally Adjusted
    "PCEPI":    "PCE",      # PCE Price Index
    "CPILFESL": "CORE_CPI", # CPI Less Food and Energy (Core CPI), SA
    "PCEPILFE": "CORE_PCE", # PCE Excluding Food and Energy (Core PCE)
}

# ---------------------------------------------------------------------------
# FRED – predictores macro
# ---------------------------------------------------------------------------
MACRO_FRED_SERIES = {
    "M2SL":         "M2",         # M2 money stock (SA, $B)
    "TB3MS":        "TBL",        # 3-month T-bill secondary market rate
    "GS10":         "GS10",       # 10-year Treasury constant maturity
    "FEDFUNDS":     "SHR",        # Federal Funds Rate (proxy Wu-Xia shadow rate)
    "MICH":         "MSC",        # U Michigan inflation expectations (1-yr ahead)
    "UNRATE":       "U",          # Civilian unemployment rate (SA)
    "CFNAI":        "CFNAI",      # Chicago Fed National Activity Index
    "SAHMREALTIME": "SAHM",       # Sahm Rule real-time
    "CPIENGSL":     "ENERGY_IDX", # CPI Energy subindex (SA) – se transforma
    "CPIAUCSL":     "CPI_IDX",    # CPI total – para M2_REAL
    "DCOILWTICO":   "WTI_RAW",    # WTI crude oil price – se transforma
}

# FRED – JOLTS + CPS para JWG
JOLTS_FRED_SERIES = {
    "JTSJOL":  "JOBOPENINGS",  # Job Openings: Total Nonfarm (miles)
    "CE16OV":  "EMPLOYED",     # Civilian Employment Level (miles)
    "CLF16OV": "LABORFORCE",   # Civilian Labor Force (miles)
}

# ---------------------------------------------------------------------------
# Columnas macro finales en us_macro_data.csv
# ---------------------------------------------------------------------------
MACRO_FINAL_COLS = [
    "M2_REAL",  # M2 real (M2/CPI * 100)
    "TBL",      # 3-month T-bill
    "TMS",      # Term spread (GS10 - TBL)
    "SHR",      # Shadow rate (FEDFUNDS proxy)
    "MSC",      # Michigan Survey expectations
    "U",        # Unemployment
    "JWG",      # Jobs-workers gap (CE16OV + JTSJOL - CLF16OV, miles)
    "CFNAI",    # Chicago Fed National Activity Index
    "SAHM",     # Sahm Rule
    "ENERGY",   # CPI energy YoY %
    "OIL",      # WTI log return mensual * 100
    # Shortage indices (opcionales – se agregan si la descarga tiene exito)
    "SH_ALL",
    "SH_IND",
    "SH_EN",
    "SH_FOOD",
    "SH_LAB",
    "SH_USA",
]

# ---------------------------------------------------------------------------
# Phillips Curve – columnas por variante
# ---------------------------------------------------------------------------
PC_COLS = {
    "PC1": ["MSC", "U",    "ENERGY"],
    "PC2": ["MSC", "JWG",  "ENERGY"],
    "PC3": ["MSC", "SAHM", "ENERGY"],
}

# ---------------------------------------------------------------------------
# Shortage indices – fuente Caldara, Iacoviello y Yu (2025) IFDP 1407
# ---------------------------------------------------------------------------
SHORTAGE_URL = (
    "https://www.matteoiacoviello.com/shortages_files/shortage_index_web.csv"
)
# Mapeo de columnas del CSV a nombres internos (Caldara, Iacoviello y Yu 2025)
SHORTAGE_COL_MAP = {
    "index_all_shortage": "SH_ALL",
    "industry_shortage":  "SH_IND",
    "energy_shortage":    "SH_EN",
    "food_shortage":      "SH_FOOD",
    "labor_shortage":     "SH_LAB",
    "index_usa":          "SH_USA",
}

# ---------------------------------------------------------------------------
# Wavelet
# ---------------------------------------------------------------------------
WAVELET = "haar"
J_LEVELS = 5
COMPONENT_NAMES = [f"D{j}" for j in range(1, J_LEVELS + 1)] + [f"S{J_LEVELS}"]

# ---------------------------------------------------------------------------
# Horizontes y ventana OOS
# ---------------------------------------------------------------------------
HORIZONS = [1, 6, 12]        # meses
OOS_START = "2000-01-01"
AO_WINDOW = 12               # meses (analogo de 4 trimestres del paper)

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
AR_MAX_LAGS = 12
PCA_COMPONENTS = 3
PLS_COMPONENTS = 2
