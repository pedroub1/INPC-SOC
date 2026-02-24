"""
config.py – Constantes y rutas para el módulo SOC.
"""
import os
import unicodedata
import re

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dashboard/
DATA_DIR = os.path.join(_DIR, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "soc_results")
COMPONENTS_DIR = os.path.join(RESULTS_DIR, "components")
FORECASTS_DIR = os.path.join(RESULTS_DIR, "forecasts")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
INPC_PATH = os.path.join(DATA_DIR, "inpc_data.csv")
MACRO_PATH = os.path.join(DATA_DIR, "macro_data.csv")

for _d in [RESULTS_DIR, COMPONENTS_DIR, FORECASTS_DIR, METRICS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Series INPC (iguales a data_loader.SERIES_MAP – 16 series mensuales)
# ---------------------------------------------------------------------------
INPC_SERIES_MAP = {
    "SP1":     "Indice General",
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

# Column names as they appear in inpc_data.csv (with indentation stripped)
INPC_SERIES_NAMES = list(INPC_SERIES_MAP.values())


def _safe_name(name: str) -> str:
    """Convierte nombre de serie a string apto para nombres de archivo."""
    # Remove accents
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Replace spaces, dots, parentheses with underscores
    safe = re.sub(r"[^A-Za-z0-9]+", "_", ascii_str.strip())
    return safe.strip("_")


SAFE_NAMES: dict[str, str] = {name: _safe_name(name) for name in INPC_SERIES_NAMES}

# ---------------------------------------------------------------------------
# Predictores macro – Banxico
# ---------------------------------------------------------------------------
MACRO_BANXICO = {
    "SF46407": "M2",
    "SF43936": "CETES28",
    "SF43878": "TIIE28",
    "SF3367":  "MBONO10Y",
    "SF60653": "USDMXN",
    "SF63528": "IPC_BMV",
}

# ---------------------------------------------------------------------------
# Predictores macro – FRED
# ---------------------------------------------------------------------------
MACRO_FRED = {"DCOILWTICO": "WTI"}

# ---------------------------------------------------------------------------
# Wavelet
# ---------------------------------------------------------------------------
WAVELET = "haar"
J_LEVELS = 5           # Produce D1..D5 y S5 (6 componentes)
COMPONENT_NAMES = [f"D{j}" for j in range(1, J_LEVELS + 1)] + [f"S{J_LEVELS}"]

# ---------------------------------------------------------------------------
# Horizontes y ventana OOS
# ---------------------------------------------------------------------------
HORIZONS = [1, 6, 12]          # meses
OOS_START = "2005-01-01"
AO_WINDOW = 12                  # meses para benchmark Article-Origin

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
AR_MAX_LAGS = 12
PCA_COMPONENTS = 3
PLS_COMPONENTS = 2
