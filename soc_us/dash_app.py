"""
dash_app.py – App Plotly Dash SOC para pronostico de CPI/PCE EE.UU.

Puerto: 8052
Inicio: python -m soc_us.dash_app

Tab 1 – Pronostico SOC: serie temporal + bandas, scatter, residuos, KPIs, descarga
Tab 2 – Componentes Wavelet: D1..D5, S5
Tab 3 – Comparacion de Modelos: ranking RMSE por componente
Tab 4 – Metricas Globales: heatmap ratio SOC/AO (2 series x 3 horizontes)
Tab 5 – Pronostico Futuro: 12 meses hacia adelante con bandas de incertidumbre
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from soc_us.config import (
    HORIZONS, US_SERIES_NAMES, SAFE_NAMES, COMPONENT_NAMES,
    FORECASTS_DIR, COMPONENTS_DIR, METRICS_DIR, RESULTS_DIR,
    OOS_START, AO_WINDOW, J_LEVELS,
)
from soc_us.forecast_future import forecast_future, last_observed_inflation

# ---------------------------------------------------------------------------
# Helpers de carga
# ---------------------------------------------------------------------------

def _safe(name: str) -> str:
    return SAFE_NAMES.get(name, name)


def load_forecasts(serie: str, h: int) -> pd.DataFrame | None:
    path = os.path.join(FORECASTS_DIR, f"{_safe(serie)}_h{h}.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


def load_components(serie: str, h: int) -> pd.DataFrame | None:
    path = os.path.join(COMPONENTS_DIR, f"{_safe(serie)}_h{h}.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


def load_metrics() -> pd.DataFrame | None:
    path = os.path.join(METRICS_DIR, "all_metrics.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


def load_metadata() -> dict:
    path = os.path.join(RESULTS_DIR, "run_metadata.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def _rmse(a, f):
    mask = ~np.isnan(a) & ~np.isnan(f)
    return float(np.sqrt(np.mean((a[mask] - f[mask]) ** 2))) if mask.sum() > 0 else np.nan


def _mae(a, f):
    mask = ~np.isnan(a) & ~np.isnan(f)
    return float(np.mean(np.abs(a[mask] - f[mask]))) if mask.sum() > 0 else np.nan


def _r2(a, f):
    mask = ~np.isnan(a) & ~np.isnan(f)
    if mask.sum() < 2:
        return np.nan
    ss_res = np.sum((a[mask] - f[mask]) ** 2)
    ss_tot = np.sum((a[mask] - a[mask].mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan


def _rolling_error_std(a, f, window=24):
    errors = a - f
    return pd.Series(errors).rolling(window, min_periods=max(6, window // 4)).std().values


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="SOC - US CPI/PCE Forecast",
    suppress_callback_exceptions=True,
    requests_pathname_prefix="/us/",   # el browser/JS usa /us/_dash-*
    routes_pathname_prefix="/",         # Flask ve la ruta ya sin /us/ (lo quitó DispatcherMiddleware)
)
server = app.server

COLORS = {
    "actual":  "#2C3E50",
    "soc":     "#27AE60",
    "soc_opt": "#2980B9",
    "ao":      "#E74C3C",
    "bg":      "#F8F9FA",
    "error":   "#8E44AD",
}

HORIZON_LABELS = {1: "1 mes", 6: "6 meses", 12: "12 meses"}
ROLLING_STD_WINDOW = 24
CI_MULTIPLIER = 1.65

# ---------------------------------------------------------------------------
# Navbar con metadatos
# ---------------------------------------------------------------------------

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("SOC - US CPI/PCE Forecast", className="fw-bold"),
        html.Span(id="us-navbar-meta", className="text-white ms-3 small"),
    ], fluid=True),
    color="dark", dark=True, className="mb-3",
)

# ---------------------------------------------------------------------------
# Acordeon de especificacion del modelo
# ---------------------------------------------------------------------------

MODEL_SPEC_ACCORDION = dbc.Accordion([
    dbc.AccordionItem(
        title="Especificacion del modelo SOC (Verona, 2026)",
        children=[
            dbc.Row([
                dbc.Col([
                    html.H6("1. Transformacion de inflacion", className="fw-bold"),
                    html.P([
                        html.Code("pi^h_t = (P_t / P_{t-h} - 1) * 100"),
                        html.Br(),
                        "Variacion % simple del nivel de precios a h meses. "
                        "CPI: CPIAUCSL (SA). PCE: PCEPI. Fuente: FRED.",
                    ], className="small"),

                    html.H6("2. Descomposicion MODWT Haar", className="fw-bold mt-2"),
                    html.P([
                        f"5 niveles (J={J_LEVELS}) via pywt.swt (norm=True). "
                        "6 componentes en frecuencia mensual:"
                    ], className="small mb-1"),
                    html.Ul([
                        html.Li("D1: ciclos 2-4 meses (alta frecuencia)", className="small"),
                        html.Li("D2: ciclos 4-8 meses", className="small"),
                        html.Li("D3: ciclos 8-16 meses", className="small"),
                        html.Li("D4: ciclos 16-32 meses", className="small"),
                        html.Li("D5: ciclos 32-64 meses", className="small"),
                        html.Li("S5: tendencia >64 meses", className="small"),
                    ]),
                ], md=6),
                dbc.Col([
                    html.H6("3. Modelos de pronostico (38 por componente)", className="fw-bold"),
                    html.Ul([
                        html.Li([html.Strong("AR: "), "AR-AIC, AR-SIC (hasta 12 rezagos)"], className="small"),
                        html.Li([html.Strong("Phillips Curve: "),
                                 "PC1 (MSC+U+ENERGY), PC2 (MSC+JWG+ENERGY), PC3 (MSC+SAHM+ENERGY)"],
                                className="small"),
                        html.Li([html.Strong("Bivariados: "), "AR(1) + cada predictor macro"], className="small"),
                        html.Li([html.Strong("Dim. reduction: "), "PCA (3 comp.), PLS-1, PLS-2 (filtrado)"], className="small"),
                        html.Li([html.Strong("Penalizados: "), "LASSO-CV, ElasticNet-CV, Ridge-CV"], className="small"),
                        html.Li([html.Strong("Combinaciones: "), "Media, Mediana, Tr.Media, DMSPE(0.25/0.50/0.75/1.00)"], className="small"),
                    ]),

                    html.H6("4. Construccion del SOC", className="fw-bold mt-2"),
                    html.Ul([
                        html.Li([html.Strong("Ventana expansiva OOS: "), f"desde {OOS_START}"], className="small"),
                        html.Li([html.Strong("SOC: "), "suma del mejor pronostico de los 6 componentes"], className="small"),
                        html.Li([html.Strong("SOC_opt: "), "igual pero excluye D1 (ruido)"], className="small"),
                        html.Li([html.Strong("Benchmark AO: "), f"media ultimos {AO_WINDOW} meses"], className="small"),
                    ]),
                    html.P([
                        html.Strong("Predictores: "),
                        "M2_REAL, TBL, TMS, SHR, MSC, U, JWG, CFNAI, SAHM, ENERGY, OIL, "
                        "SH_ALL, SH_IND, SH_EN, SH_FOOD, SH_LAB, SH_USA. Fuente: FRED + Caldara et al. (2025).",
                    ], className="small text-muted mt-1"),
                ], md=6),
            ]),
        ],
    ),
], start_collapsed=True, className="mb-3")


# ---------------------------------------------------------------------------
# KPI card helper
# ---------------------------------------------------------------------------

def _kpi_card(label, value, color="primary"):
    return dbc.Card(
        dbc.CardBody([
            html.P(label, className="card-subtitle text-muted small mb-1"),
            html.H5(value, className=f"text-{color} mb-0"),
        ]),
        className="text-center shadow-sm h-100",
    )


# ---------------------------------------------------------------------------
# Tab 1 – Pronostico SOC
# ---------------------------------------------------------------------------

tab1_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Serie", className="fw-bold"),
            dcc.Dropdown(
                id="t1-serie",
                options=[{"label": s, "value": s} for s in US_SERIES_NAMES],
                value=US_SERIES_NAMES[0],
                clearable=False,
            ),
        ], md=6),
        dbc.Col([
            html.Label("Horizonte", className="fw-bold"),
            dbc.RadioItems(
                id="t1-horizon",
                options=[{"label": HORIZON_LABELS[h], "value": h} for h in HORIZONS],
                value=1, inline=True, className="mt-2",
            ),
        ], md=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(id="kpi-rmse-soc",     md=3),
        dbc.Col(id="kpi-rmse-soc-opt", md=3),
        dbc.Col(id="kpi-rmse-ao",      md=3),
        dbc.Col(id="kpi-ratio",        md=3),
    ], className="mb-3 g-2"),

    html.H6("Serie temporal con bandas de confianza (90%, rolling 24 meses)", className="text-muted mb-1"),
    dcc.Graph(id="t1-chart", style={"height": "430px"}),

    dbc.Row([
        dbc.Col([
            html.H6("Observado vs. Pronosticado", className="text-muted mb-1"),
            dcc.Graph(id="t1-scatter", style={"height": "380px"}),
        ], md=6),
        dbc.Col([
            html.H6("Errores de pronostico (actual - SOC)", className="text-muted mb-1"),
            dcc.Graph(id="t1-errors", style={"height": "380px"}),
        ], md=6),
    ], className="mt-3"),

    html.Div(MODEL_SPEC_ACCORDION, className="mt-4"),

    html.H6("Ultimos 24 periodos OOS", className="mt-2 mb-2 fw-bold"),
    html.Div(id="t1-table"),
    dbc.Row([
        dbc.Col([
            dbc.Button("Descargar OOS (Excel)", id="t1-download-btn",
                       color="outline-success", size="sm", className="mt-2"),
            dcc.Download(id="t1-download"),
        ]),
    ], className="mt-1 mb-3"),
], fluid=True)


# ---------------------------------------------------------------------------
# Tab 2 – Componentes Wavelet
# ---------------------------------------------------------------------------

tab2_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Serie", className="fw-bold"),
            dcc.Dropdown(id="t2-serie",
                         options=[{"label": s, "value": s} for s in US_SERIES_NAMES],
                         value=US_SERIES_NAMES[0], clearable=False),
        ], md=6),
        dbc.Col([
            html.Label("Horizonte", className="fw-bold"),
            dbc.RadioItems(id="t2-horizon",
                           options=[{"label": HORIZON_LABELS[h], "value": h} for h in HORIZONS],
                           value=1, inline=True, className="mt-2"),
        ], md=6),
    ], className="mb-3"),
    dcc.Graph(id="t2-subplots", style={"height": "700px"}),
    html.H6("Mejor modelo por componente (menor RMSE OOS)", className="mt-3 mb-2 fw-bold"),
    html.Div(id="t2-best-models"),
], fluid=True)


# ---------------------------------------------------------------------------
# Tab 3 – Comparacion de Modelos
# ---------------------------------------------------------------------------

tab3_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Serie", className="fw-bold"),
            dcc.Dropdown(id="t3-serie",
                         options=[{"label": s, "value": s} for s in US_SERIES_NAMES],
                         value=US_SERIES_NAMES[0], clearable=False),
        ], md=4),
        dbc.Col([
            html.Label("Horizonte", className="fw-bold"),
            dbc.RadioItems(id="t3-horizon",
                           options=[{"label": HORIZON_LABELS[h], "value": h} for h in HORIZONS],
                           value=1, inline=True, className="mt-2"),
        ], md=4),
        dbc.Col([
            html.Label("Componente wavelet", className="fw-bold"),
            dcc.Dropdown(id="t3-component",
                         options=[{"label": c, "value": c} for c in COMPONENT_NAMES],
                         value=COMPONENT_NAMES[0], clearable=False),
        ], md=4),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Switch(id="t3-show-combos", label="Incluir combinaciones de modelos", value=True),
        ]),
    ], className="mb-2"),
    dcc.Graph(id="t3-rmse-chart", style={"height": "500px"}),
    html.Div(id="t3-table", className="mt-3"),
], fluid=True)


# ---------------------------------------------------------------------------
# Tab 4 – Metricas Globales
# ---------------------------------------------------------------------------

tab4_layout = dbc.Container([
    html.H5("Ratio RMSE SOC/AO por serie y horizonte", className="mb-3"),
    html.P("Valores < 1 indican que SOC supera al benchmark Atkeson-Ohanian.",
           className="text-muted small"),
    dcc.Graph(id="t4-heatmap", style={"height": "400px"}),
    html.H6("Tabla de metricas detalladas", className="mt-4 mb-2 fw-bold"),
    html.Div(id="t4-table"),
], fluid=True)


# ---------------------------------------------------------------------------
# Tab 5 – Pronostico Futuro
# ---------------------------------------------------------------------------

tab5_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Serie", className="fw-bold"),
            dcc.Dropdown(id="t5-serie",
                         options=[{"label": s, "value": s} for s in US_SERIES_NAMES],
                         value=US_SERIES_NAMES[0], clearable=False),
        ], md=4),
        dbc.Col([
            html.Label("Meses de historia", className="fw-bold"),
            dcc.Slider(id="t5-history", min=12, max=120, step=12, value=48,
                       marks={v: str(v) for v in [12, 24, 36, 48, 60, 84, 120]}),
        ], md=5),
        dbc.Col([
            dbc.Button("Calcular pronostico", id="t5-btn",
                       color="primary", className="mt-4 w-100"),
        ], md=3),
    ], className="mb-3"),
    dcc.Graph(id="t5-chart", style={"height": "500px"}),
    html.H6("Pronosticos por horizonte", className="mt-3 mb-2 fw-bold"),
    html.Div(id="t5-table"),
], fluid=True)


# ---------------------------------------------------------------------------
# Tab 6 – Informacion Metodologica
# ---------------------------------------------------------------------------

_US_SERIES_TABLE = [
    {"Serie": "CPI",      "FRED ID": "CPIAUCSL", "Fuente": "BLS/FRED",
     "Descripcion": "Consumer Price Index, All Urban Consumers, SA",
     "Muestra": "1947-01 – presente"},
    {"Serie": "Core CPI", "FRED ID": "CPILFESL", "Fuente": "BLS/FRED",
     "Descripcion": "CPI Less Food and Energy, SA (inflacion subyacente)",
     "Muestra": "1957-01 – presente"},
    {"Serie": "PCE",      "FRED ID": "PCEPI",    "Fuente": "BEA/FRED",
     "Descripcion": "PCE Price Index (deflactor preferido por la Fed)",
     "Muestra": "1959-01 – presente"},
    {"Serie": "Core PCE", "FRED ID": "PCEPILFE", "Fuente": "BEA/FRED",
     "Descripcion": "PCE Excluding Food and Energy (objetivo operativo Fed: 2%)",
     "Muestra": "1959-01 – presente"},
]

_US_MACRO_TABLE = [
    {"Variable": "M2_REAL",   "FRED ID": "M2SL / CPIAUCSL",  "Fuente": "FRED",
     "Descripcion": "M2 real (presion monetaria)",            "Transformacion": "M2 nominal / CPI × 100"},
    {"Variable": "TBL",       "FRED ID": "TB3MS",             "Fuente": "FRED",
     "Descripcion": "T-bill 3 meses (tasa corto plazo)",      "Transformacion": "Nivel (%)"},
    {"Variable": "TMS",       "FRED ID": "GS10 - TB3MS",      "Fuente": "FRED",
     "Descripcion": "Pendiente de la curva de rendimientos",  "Transformacion": "GS10 − TBL (pp)"},
    {"Variable": "SHR",       "FRED ID": "FEDFUNDS",          "Fuente": "FRED",
     "Descripcion": "Tasa Fed Funds (proxy Wu-Xia shadow rate)","Transformacion": "Nivel (%)"},
    {"Variable": "MSC",       "FRED ID": "MICH",              "Fuente": "U.Mich/FRED",
     "Descripcion": "Expectativas de inflacion a 1 ano",      "Transformacion": "Nivel (%)"},
    {"Variable": "U",         "FRED ID": "UNRATE",            "Fuente": "BLS/FRED",
     "Descripcion": "Tasa de desempleo civil, SA",            "Transformacion": "Nivel (%)"},
    {"Variable": "JWG",       "FRED ID": "CE16OV+JTSJOL-CLF16OV","Fuente": "BLS/FRED",
     "Descripcion": "Jobs-workers gap (presion mercado laboral)","Transformacion": "Empleados + Vacantes − Fuerza laboral (miles)"},
    {"Variable": "CFNAI",     "FRED ID": "CFNAI",             "Fuente": "Chicago Fed/FRED",
     "Descripcion": "Indice Nacional de Actividad Economica", "Transformacion": "Nivel (std.)"},
    {"Variable": "SAHM",      "FRED ID": "SAHMREALTIME",      "Fuente": "FRED",
     "Descripcion": "Regla de Sahm en tiempo real (recesion)","Transformacion": "Nivel (pp)"},
    {"Variable": "ENERGY",    "FRED ID": "CPIENGSL",          "Fuente": "BLS/FRED",
     "Descripcion": "Sub-indice CPI Energia",                 "Transformacion": "Variacion YoY (%)"},
    {"Variable": "OIL",       "FRED ID": "DCOILWTICO",        "Fuente": "EIA/FRED",
     "Descripcion": "Petroleo crudo WTI (spot)",              "Transformacion": "log(WTI_t / WTI_{t-1}) × 100"},
    {"Variable": "SH_ALL",    "FRED ID": "index_all_shortage","Fuente": "Caldara et al. (2025)",
     "Descripcion": "Indice global de escasez",               "Transformacion": "Indice textual"},
    {"Variable": "SH_IND",    "FRED ID": "industry_shortage", "Fuente": "Caldara et al. (2025)",
     "Descripcion": "Escasez en sector industrial",           "Transformacion": "Indice textual"},
    {"Variable": "SH_EN",     "FRED ID": "energy_shortage",   "Fuente": "Caldara et al. (2025)",
     "Descripcion": "Escasez energetica",                     "Transformacion": "Indice textual"},
    {"Variable": "SH_FOOD",   "FRED ID": "food_shortage",     "Fuente": "Caldara et al. (2025)",
     "Descripcion": "Escasez alimentaria",                    "Transformacion": "Indice textual"},
    {"Variable": "SH_LAB",    "FRED ID": "labor_shortage",    "Fuente": "Caldara et al. (2025)",
     "Descripcion": "Escasez laboral",                        "Transformacion": "Indice textual"},
    {"Variable": "SH_USA",    "FRED ID": "index_usa",         "Fuente": "Caldara et al. (2025)",
     "Descripcion": "Indice de escasez especifico EE.UU.",    "Transformacion": "Indice textual"},
]

_US_MODELS_TABLE = [
    {"Nombre": "AR_AIC",      "Tipo": "Autorregresivo", "Variables": "y (rezagos)",
     "Descripcion": "AR directo a h pasos; rezagos optimos por AIC (max. 12)"},
    {"Nombre": "AR_SIC",      "Tipo": "Autorregresivo", "Variables": "y (rezagos)",
     "Descripcion": "AR directo a h pasos; rezagos optimos por BIC (max. 12)"},
    {"Nombre": "PC1",         "Tipo": "Phillips Curve", "Variables": "y, MSC, U, ENERGY",
     "Descripcion": "Curva de Phillips: expectativas (MSC) + desempleo (U) + energia"},
    {"Nombre": "PC2",         "Tipo": "Phillips Curve", "Variables": "y, MSC, JWG, ENERGY",
     "Descripcion": "Curva de Phillips: expectativas + jobs-workers gap + energia"},
    {"Nombre": "PC3",         "Tipo": "Phillips Curve", "Variables": "y, MSC, SAHM, ENERGY",
     "Descripcion": "Curva de Phillips: expectativas + regla Sahm + energia"},
    {"Nombre": "BIV_{x}",     "Tipo": "Bivariado",      "Variables": "y + un predictor macro",
     "Descripcion": "AR(1) directo + cada predictor macro por separado (17 modelos)"},
    {"Nombre": "PCA",         "Tipo": "Reduccion dim.", "Variables": "y + 3 comp. principales",
     "Descripcion": "AR(1) + 3 componentes principales de todos los predictores macro"},
    {"Nombre": "PLS1",        "Tipo": "Reduccion dim.", "Variables": "y + 2 comp. PLS",
     "Descripcion": "AR(1) + 2 comp. PLS sobre predictores originales (maximiza cov con y)"},
    {"Nombre": "PLS2",        "Tipo": "Reduccion dim.", "Variables": "y + 2 comp. PLS filtradas",
     "Descripcion": "PLS sobre versiones wavelet-filtradas de los predictores (mismo nivel de frecuencia)"},
    {"Nombre": "LASSO",       "Tipo": "Penalizado",     "Variables": "todos macro + lags de y",
     "Descripcion": "LASSO con lambda por CV 5-fold; seleccion automatica de predictores"},
    {"Nombre": "ELASTICNET",  "Tipo": "Penalizado",     "Variables": "todos macro + lags de y",
     "Descripcion": "ElasticNet con lambda y l1_ratio por CV; combina LASSO y Ridge"},
    {"Nombre": "RIDGE",       "Tipo": "Penalizado",     "Variables": "todos macro + lags de y",
     "Descripcion": "Ridge con lambda por CV; sin seleccion de variables"},
    {"Nombre": "C_MEAN",      "Tipo": "Combinacion",    "Variables": "todos los modelos",
     "Descripcion": "Media simple de todos los pronosticos individuales"},
    {"Nombre": "C_MEDIAN",    "Tipo": "Combinacion",    "Variables": "todos los modelos",
     "Descripcion": "Mediana; robusta a modelos atipicos"},
    {"Nombre": "C_TRIMEAN",   "Tipo": "Combinacion",    "Variables": "todos los modelos",
     "Descripcion": "Media recortada al 10%; excluye extremos"},
    {"Nombre": "C_DMSPE025",  "Tipo": "Comb. DMSPE",   "Variables": "todos los modelos",
     "Descripcion": "Pesos inv. proporcionales a MSPE^0.25 (pesos casi uniformes)"},
    {"Nombre": "C_DMSPE050",  "Tipo": "Comb. DMSPE",   "Variables": "todos los modelos",
     "Descripcion": "Pesos inv. proporcionales a MSPE^0.50"},
    {"Nombre": "C_DMSPE075",  "Tipo": "Comb. DMSPE",   "Variables": "todos los modelos",
     "Descripcion": "Pesos inv. proporcionales a MSPE^0.75"},
    {"Nombre": "C_DMSPE100",  "Tipo": "Comb. DMSPE",   "Variables": "todos los modelos",
     "Descripcion": "Pesos inv. proporcionales a MSPE^1.00 (penaliza mas al peor modelo)"},
]

_TYPE_COLORS = {
    "Autorregresivo": "#2980B9",
    "Phillips Curve": "#8E44AD",
    "Bivariado":      "#27AE60",
    "Reduccion dim.": "#E67E22",
    "Penalizado":     "#E74C3C",
    "Combinacion":    "#95A5A6",
    "Comb. DMSPE":    "#7F8C8D",
}

tab6_layout = dbc.Container([
    html.H5("Informacion Metodologica", className="fw-bold mt-2 mb-1"),
    html.P(
        "Documentacion tecnica de la metodologia, los datos y los modelos utilizados "
        "en la construccion del pronostico SOC (Sum of the Cycles) para EE.UU.",
        className="text-muted small mb-3",
    ),

    # ---- Seccion 1: Datos ----
    html.H6("1. Series de precios utilizadas", className="fw-bold text-primary"),
    html.P(
        "Cuatro indices de precios en frecuencia mensual desde FRED (Federal Reserve Bank of St. Louis). "
        "La transformacion a inflacion h-mensual es: pi_t^h = (P_t / P_{t-h} - 1) × 100.",
        className="small text-muted mb-1",
    ),
    dash_table.DataTable(
        data=_US_SERIES_TABLE,
        columns=[{"name": c, "id": c} for c in ["Serie", "FRED ID", "Fuente", "Descripcion", "Muestra"]],
        style_cell={"fontSize": "12px", "textAlign": "left", "padding": "4px 8px",
                    "whiteSpace": "normal", "height": "auto"},
        style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
        style_data_conditional=[
            {"if": {"filter_query": '{Serie} contains "Core"'},
             "color": "#555", "fontStyle": "italic"},
        ],
        style_table={"overflowX": "auto", "marginBottom": "16px"},
        page_size=4,
    ),

    html.H6("2. Predictores macroeconomicos (FRED + Caldara et al. 2025)", className="fw-bold text-primary"),
    html.P(
        "17 predictores macro alineados al indice mensual de inflacion antes de la estimacion. "
        "Los shortage indices de Caldara, Iacoviello & Yu (2025) se descargan de matteoiacoviello.com.",
        className="small text-muted mb-1",
    ),
    dash_table.DataTable(
        data=_US_MACRO_TABLE,
        columns=[{"name": c, "id": c} for c in ["Variable", "FRED ID", "Fuente", "Descripcion", "Transformacion"]],
        style_cell={"fontSize": "11px", "textAlign": "left", "padding": "4px 8px",
                    "whiteSpace": "normal", "height": "auto"},
        style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
        style_data_conditional=[
            {"if": {"filter_query": '{Fuente} contains "Caldara"'},
             "fontStyle": "italic", "color": "#666"},
        ],
        style_table={"overflowX": "auto", "marginBottom": "20px"},
        page_size=17,
    ),

    # ---- Seccion 2: Metodologia ----
    html.H6("3. Metodologia SOC (Verona, 2026)", className="fw-bold text-primary"),
    dbc.Accordion([
        dbc.AccordionItem(
            title="Descomposicion wavelet y transformacion de inflacion",
            children=dbc.Row([
                dbc.Col([
                    html.H6("Transformacion de inflacion", className="fw-bold small"),
                    html.P([
                        html.Code("pi_t^h = (P_t / P_{t-h} - 1) × 100"), html.Br(),
                        "Variacion porcentual acumulada del nivel de precios a h meses adelante. "
                        "Se aplica a CPI, Core CPI, PCE y Core PCE para h = {1, 6, 12}.",
                    ], className="small"),
                ], md=5),
                dbc.Col([
                    html.H6("Descomposicion MODWT Haar (J=5)", className="fw-bold small"),
                    dash_table.DataTable(
                        data=[
                            {"Comp.": "D1", "Escala": "2–4 meses",   "Interpretacion": "Ruido de alta frecuencia (excluido en SOC_opt)"},
                            {"Comp.": "D2", "Escala": "4–8 meses",   "Interpretacion": "Ciclos de corto plazo"},
                            {"Comp.": "D3", "Escala": "8–16 meses",  "Interpretacion": "Ciclos trimestrales-anuales"},
                            {"Comp.": "D4", "Escala": "16–32 meses", "Interpretacion": "Ciclos de mediano plazo"},
                            {"Comp.": "D5", "Escala": "32–64 meses", "Interpretacion": "Ciclos de largo plazo"},
                            {"Comp.": "S5", "Escala": ">64 meses",   "Interpretacion": "Tendencia secular"},
                        ],
                        columns=[{"name": c, "id": c} for c in ["Comp.", "Escala", "Interpretacion"]],
                        style_cell={"fontSize": "11px", "padding": "3px 8px"},
                        style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
                        style_data_conditional=[
                            {"if": {"filter_query": '{Comp.} = "D1"'}, "color": "#aaa", "fontStyle": "italic"},
                            {"if": {"filter_query": '{Comp.} = "S5"'}, "fontWeight": "bold"},
                        ],
                    ),
                    html.P(
                        "La suma de los 6 componentes reconstruye exactamente la serie original (error < 1e-6). "
                        "Implementacion: pywt.swt(wavelet='haar', norm=True, level=5).",
                        className="small text-muted mt-1",
                    ),
                ], md=7),
            ]),
        ),
        dbc.AccordionItem(
            title="Evaluacion OOS, benchmark AO y construccion del SOC",
            children=dbc.Row([
                dbc.Col([
                    html.H6("Ventana expansiva OOS", className="fw-bold small"),
                    html.Ul([
                        html.Li(f"Inicio OOS: {OOS_START}", className="small"),
                        html.Li("En cada t: entrena con [inicio, t], pronostica t+h (pronostico directo).", className="small"),
                        html.Li("Mejor modelo por componente = minimo RMSE acumulado OOS.", className="small"),
                        html.Li("SOC = suma de los mejores 6 pronosticos por componente.", className="small"),
                        html.Li("SOC_opt = igual pero excluye D1 (ruido de alta frecuencia).", className="small"),
                    ]),
                ], md=6),
                dbc.Col([
                    html.H6(f"Benchmark AO (Atkeson-Ohanian)", className="fw-bold small"),
                    html.P([
                        f"Media de los ultimos {AO_WINDOW} meses de inflacion realizada. ",
                        "Si ratio RMSE SOC/AO < 1, el SOC supera al naive benchmark.",
                    ], className="small"),
                    html.H6("Horizontes evaluados", className="fw-bold small mt-2"),
                    html.P(
                        "h = 1 (inflacion mensual), h = 6 (acumulada 6 meses), h = 12 (variacion anual). "
                        "Para el pronostico futuro se usa el horizonte OOS mas cercano.",
                        className="small",
                    ),
                ], md=6),
            ]),
        ),
    ], start_collapsed=True, className="mb-4"),

    # ---- Seccion 3: Catalogo de modelos ----
    html.H6("4. Catalogo de modelos (38 por componente)", className="fw-bold text-primary"),
    html.P(
        "Se estiman hasta 38 modelos individuales + 7 combinaciones por componente wavelet y horizonte. "
        "PLS-2 es exclusivo de este modulo: usa versiones wavelet-filtradas de los predictores macro, "
        "garantizando coherencia de frecuencia entre la variable dependiente y los predictores.",
        className="small text-muted mb-1",
    ),
    dbc.Accordion([
        dbc.AccordionItem(
            title="Ver catalogo completo de modelos",
            children=dash_table.DataTable(
                data=_US_MODELS_TABLE,
                columns=[{"name": c, "id": c} for c in ["Nombre", "Tipo", "Variables", "Descripcion"]],
                style_cell={"fontSize": "11px", "textAlign": "left", "padding": "4px 8px",
                            "whiteSpace": "normal", "height": "auto"},
                style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
                style_data_conditional=[
                    {"if": {"filter_query": f'{{Tipo}} = "{t}"', "column_id": "Tipo"},
                     "color": c, "fontWeight": "bold"}
                    for t, c in _TYPE_COLORS.items()
                ],
                style_table={"overflowX": "auto"},
                page_size=19,
                sort_action="native",
            ),
        ),
    ], start_collapsed=True, className="mb-4"),

    # ---- Seccion 4: Petroleo y Cadenas de Suministro ----
    html.H6("5. Petroleo y cadenas de suministro", className="fw-bold text-primary"),
    dbc.Accordion([
        dbc.AccordionItem(
            title="Precio del petroleo: mecanismo de transmision a la inflacion",
            children=[
                dbc.Row([
                    dbc.Col([
                        html.H6("Canales de transmision directa", className="fw-bold small"),
                        html.P(
                            "El petróleo crudo WTI (DCOILWTICO) se transforma en retorno logarítmico mensual × 100 "
                            "para capturar su volatilidad asimétrica. Sus canales de transmision al CPI son:",
                            className="small",
                        ),
                        html.Ul([
                            html.Li([html.Strong("Energía directa: "),
                                     "Gasolina, gas natural y combustibles de calefacción representan ~7% del CPI. "
                                     "Un alza del 10% en WTI eleva el sub-índice de energía ~5-7% en 1-2 meses."],
                                    className="small"),
                            html.Li([html.Strong("Transporte y flete: "),
                                     "Costos de logística y distribución se trasladan a precios al consumidor "
                                     "con rezago de 2-4 meses."],
                                    className="small"),
                            html.Li([html.Strong("Insumos industriales: "),
                                     "Petroquímicos, fertilizantes y plásticos elevan costos de producción. "
                                     "Pass-through a Core CPI estimado en 0.1–0.2 pp por cada 10% de alza en WTI, "
                                     "materializado a 3–6 meses."],
                                    className="small"),
                        ]),
                    ], md=6),
                    dbc.Col([
                        html.H6("Core vs. Headline: asimetria del pass-through", className="fw-bold small"),
                        html.P(
                            "El petróleo impacta desproporcionadamente al CPI headline (vía energía y alimentos) "
                            "vs. al Core CPI/PCE. Esta asimetría es relevante para la política monetaria:",
                            className="small",
                        ),
                        html.Ul([
                            html.Li([html.Strong("CPI/PCE headline: "),
                                     "Respuesta inmediata y magnitud alta (elasticidad ~0.1). "
                                     "Componente D1-D2 del wavelet captura estos choques transitorios."],
                                    className="small"),
                            html.Li([html.Strong("Core CPI/PCE: "),
                                     "Respuesta más suave y rezagada (~6 meses). Afecta principalmente "
                                     "componentes D3-D4 del wavelet (ciclos 8-32 meses)."],
                                    className="small"),
                            html.Li([html.Strong("Episodios históricos: "),
                                     "1973-74 embargo OPEC (+400% WTI → CPI +12%), "
                                     "1979 crisis iraní (+130% → estagflación), "
                                     "2022 invasión Ucrania (+76% → CPI pico 9.1% jun-2022)."],
                                    className="small"),
                        ]),
                        html.P(
                            "El predictor ENERGY (variación YoY del sub-índice CPI Energía) captura "
                            "el efecto acumulado de oil y gas en el nivel general.",
                            className="small text-muted mt-1",
                        ),
                    ], md=6),
                ]),
            ],
        ),
        dbc.AccordionItem(
            title="Indices de escasez (Caldara, Iacoviello & Yu, 2025): construccion e interpretacion",
            children=[
                dbc.Row([
                    dbc.Col([
                        html.H6("Construccion de los indices", className="fw-bold small"),
                        html.P([
                            "Caldara, Iacoviello & Yu (2025) — ",
                            html.Em("IFDP 1407, Federal Reserve Board"),
                            " — construyen 6 indices de escasez a partir del análisis textual de:",
                        ], className="small"),
                        html.Ul([
                            html.Li("Transcripciones de earnings calls de empresas cotizadas (2000-presente)", className="small"),
                            html.Li("Libros Beige del Fed regional (1970-presente)", className="small"),
                            html.Li("Noticias financieras", className="small"),
                        ]),
                        html.P(
                            "Cuentan la frecuencia de términos de escasez ('shortage', 'backlog', "
                            "'supply constraint', etc.) normalizados por el total de palabras en el corpus. "
                            "El resultado es un índice de alta frecuencia, sin revision, disponible en tiempo real.",
                            className="small",
                        ),
                        html.H6("Los 6 indices disponibles", className="fw-bold small mt-2"),
                        dash_table.DataTable(
                            data=[
                                {"Indice": "SH_ALL",  "Nombre": "Global shortage",     "Cobertura": "Todas las categorías"},
                                {"Indice": "SH_IND",  "Nombre": "Industry shortage",   "Cobertura": "Manufacturas e insumos industriales"},
                                {"Indice": "SH_EN",   "Nombre": "Energy shortage",     "Cobertura": "Petróleo, gas, electricidad"},
                                {"Indice": "SH_FOOD", "Nombre": "Food shortage",       "Cobertura": "Alimentos y agro"},
                                {"Indice": "SH_LAB",  "Nombre": "Labor shortage",      "Cobertura": "Mercado laboral, salarios"},
                                {"Indice": "SH_USA",  "Nombre": "US-specific shortage","Cobertura": "Escasez específica de EE.UU."},
                            ],
                            columns=[{"name": c, "id": c} for c in ["Indice", "Nombre", "Cobertura"]],
                            style_cell={"fontSize": "11px", "padding": "3px 8px"},
                            style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
                        ),
                    ], md=6),
                    dbc.Col([
                        html.H6("Mecanismo de transmision a la inflacion", className="fw-bold small"),
                        html.P(
                            "Los índices de escasez capturan presiones de oferta que los indicadores macro "
                            "convencionales no detectan a tiempo. Sus canales de transmision son:",
                            className="small",
                        ),
                        html.Ul([
                            html.Li([html.Strong("Costos de producción (SH_IND, SH_EN): "),
                                     "Restricciones de insumos elevan costos marginales → empresas transfieren "
                                     "al precio de venta con rezago de 2-4 meses. Afecta principalmente "
                                     "bienes duraderos y semi-duraderos."],
                                    className="small"),
                            html.Li([html.Strong("Presion salarial (SH_LAB): "),
                                     "Escasez laboral eleva salarios nominales → presion sobre servicios "
                                     "y Core PCE con rezagos de 3-9 meses (componentes D3-D4)."],
                                    className="small"),
                            html.Li([html.Strong("Expectativas de inflacion: "),
                                     "Alta escasez puede desenclavar expectativas si persiste >6 meses, "
                                     "afectando componentes de largo plazo (D5, S5)."],
                                    className="small"),
                        ]),
                        html.H6("Episodio COVID-19 (2021-2022)", className="fw-bold small mt-2"),
                        html.P(
                            "Los índices de escasez alcanzaron máximos históricos en 2021-2022, "
                            "anticipando el repunte inflacionario antes que los modelos macro convencionales. "
                            "SH_ALL y SH_IND superaron 2 desviaciones estándar sobre su media histórica "
                            "desde Q2-2021. La inclusión de estos índices mejora significativamente el RMSE "
                            "del SOC para h=6 y h=12 en ese periodo.",
                            className="small",
                        ),
                        html.P([
                            html.Strong("Fuente: "),
                            "Caldara, D., Iacoviello, M., & Yu, A. (2025). ",
                            html.Em("Measuring Supply and Demand Conditions in the Economy. "),
                            "IFDP 1407, Federal Reserve Board.",
                        ], className="small text-muted mt-2"),
                    ], md=6),
                ]),
            ],
        ),
    ], start_collapsed=False, className="mb-4"),

    # ---- Seccion 5: Resultados dinamicos ----
    html.H6("6. Resultados de estimacion por serie y horizonte", className="fw-bold text-primary"),
    html.P("Selecciona una serie y horizonte para ver los modelos estimados y su desempeno OOS.",
           className="small text-muted mb-2"),
    dbc.Row([
        dbc.Col([
            html.Label("Serie", className="fw-bold small"),
            dcc.Dropdown(
                id="t6-serie",
                options=[{"label": s, "value": s} for s in US_SERIES_NAMES],
                value=US_SERIES_NAMES[0], clearable=False,
            ),
        ], md=6),
        dbc.Col([
            html.Label("Horizonte", className="fw-bold small"),
            dbc.RadioItems(
                id="t6-horizon",
                options=[{"label": HORIZON_LABELS[h], "value": h} for h in HORIZONS],
                value=1, inline=True, className="mt-2",
            ),
        ], md=6),
    ], className="mb-3"),
    html.Div(id="t6-results"),
], fluid=True)


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------

app.layout = html.Div([
    navbar,
    dcc.Interval(id="meta-interval", interval=120_000, n_intervals=0),
    dbc.Container([
        dbc.Tabs([
            dbc.Tab(tab1_layout, label="Pronostico SOC",         tab_id="tab1"),
            dbc.Tab(tab2_layout, label="Componentes Wavelet",    tab_id="tab2"),
            dbc.Tab(tab3_layout, label="Comparacion de Modelos", tab_id="tab3"),
            dbc.Tab(tab4_layout, label="Metricas Globales",      tab_id="tab4"),
            dbc.Tab(tab5_layout, label="Pronostico Futuro",      tab_id="tab5"),
            dbc.Tab(tab6_layout, label="Info Metodologia",       tab_id="tab6"),
        ], id="tabs", active_tab="tab1"),
    ], fluid=True),
])


# ---------------------------------------------------------------------------
# Callbacks – Navbar
# ---------------------------------------------------------------------------

@app.callback(
    Output("us-navbar-meta", "children"),
    Input("meta-interval", "n_intervals"),
)
def update_navbar(_):
    meta = load_metadata()
    run_date = meta.get("run_date", "")
    if run_date:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(run_date)
            run_date = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
    n_ok = meta.get("n_ok", "?")
    n_tasks = meta.get("n_tasks", 6)
    return f"Ultima corrida: {run_date}  |  {n_ok}/{n_tasks} series OK"


# ---------------------------------------------------------------------------
# Callbacks – Tab 1
# ---------------------------------------------------------------------------

@app.callback(
    [Output("kpi-rmse-soc", "children"),
     Output("kpi-rmse-soc-opt", "children"),
     Output("kpi-rmse-ao", "children"),
     Output("kpi-ratio", "children"),
     Output("t1-chart", "figure"),
     Output("t1-scatter", "figure"),
     Output("t1-errors", "figure"),
     Output("t1-table", "children")],
    [Input("t1-serie", "value"), Input("t1-horizon", "value")],
)
def update_tab1(serie, h):
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_white",
                            annotations=[{"text": "Sin datos disponibles. Ejecuta fit_soc_us.py primero.",
                                          "showarrow": False, "font": {"size": 14}}])
    no_data = (_kpi_card("RMSE SOC", "N/D"), _kpi_card("RMSE SOC_opt", "N/D"),
               _kpi_card("RMSE AO", "N/D"), _kpi_card("Ratio SOC/AO", "N/D"),
               empty_fig, empty_fig, empty_fig, html.P("Sin datos."))

    df = load_forecasts(serie, h)
    if df is None:
        return no_data

    actual  = df["actual"].values.astype(float)
    soc     = df["SOC"].values.astype(float)
    soc_opt = df["SOC_opt"].values.astype(float)
    ao      = df["AO"].values.astype(float)
    dates   = df.index

    r_soc     = _rmse(actual, soc)
    r_soc_opt = _rmse(actual, soc_opt)
    r_ao      = _rmse(actual, ao)
    ratio     = r_soc / r_ao if r_ao and r_ao > 0 else np.nan

    kpi1 = _kpi_card("RMSE SOC",     f"{r_soc:.4f}", "success")
    kpi2 = _kpi_card("RMSE SOC_opt", f"{r_soc_opt:.4f}", "primary")
    kpi3 = _kpi_card("RMSE AO",      f"{r_ao:.4f}", "danger")
    ratio_color = "success" if (not np.isnan(ratio) and ratio < 1) else "warning"
    kpi4 = _kpi_card("Ratio SOC/AO", f"{ratio:.3f}" if not np.isnan(ratio) else "N/D", ratio_color)

    # --- Grafica principal ---
    rolling_std = _rolling_error_std(actual, soc, ROLLING_STD_WINDOW)
    upper = soc + CI_MULTIPLIER * rolling_std
    lower = soc - CI_MULTIPLIER * rolling_std

    fig1 = go.Figure()
    fig1.add_traces([
        go.Scatter(x=dates, y=upper, mode="lines", line=dict(width=0),
                   showlegend=False, hoverinfo="skip"),
        go.Scatter(x=dates, y=lower, mode="lines", line=dict(width=0),
                   fill="tonexty", fillcolor="rgba(39,174,96,0.15)",
                   name="Banda 90%", hoverinfo="skip"),
        go.Scatter(x=dates, y=actual,  mode="lines", line=dict(color=COLORS["actual"], width=2),
                   name="Realizado"),
        go.Scatter(x=dates, y=soc,     mode="lines", line=dict(color=COLORS["soc"], width=2, dash="dash"),
                   name="SOC"),
        go.Scatter(x=dates, y=soc_opt, mode="lines", line=dict(color=COLORS["soc_opt"], width=1.5, dash="dot"),
                   name="SOC_opt"),
        go.Scatter(x=dates, y=ao,      mode="lines", line=dict(color=COLORS["ao"], width=1.5, dash="dashdot"),
                   name="AO (benchmark)"),
    ])
    fig1.update_layout(template="plotly_white", legend=dict(orientation="h", y=-0.15),
                       xaxis_title="Fecha", yaxis_title=f"Inflacion h={h} (%)",
                       hovermode="x unified")

    # --- Scatter ---
    mask = ~np.isnan(actual) & ~np.isnan(soc)
    r2 = _r2(actual, soc)
    mae = _mae(actual, soc)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=actual[mask], y=soc[mask], mode="markers",
                               marker=dict(color=COLORS["soc"], opacity=0.6),
                               name="SOC vs Realizado"))
    mn = min(np.nanmin(actual), np.nanmin(soc))
    mx = max(np.nanmax(actual), np.nanmax(soc))
    fig2.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                               line=dict(color="gray", dash="dash"), name="45°"))
    fig2.update_layout(template="plotly_white",
                       xaxis_title="Realizado (%)", yaxis_title="Pronosticado SOC (%)",
                       title=f"R²={r2:.3f}  MAE={mae:.4f}" if not np.isnan(r2) else "")

    # --- Errores ---
    errors = actual - soc
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=dates, y=errors, marker_color=np.where(errors >= 0, COLORS["soc"], COLORS["ao"]),
                          name="Error (actual - SOC)"))
    fig3.add_hline(y=0, line_color="black", line_width=1)
    fig3.update_layout(template="plotly_white", xaxis_title="Fecha", yaxis_title="Error (%)",
                       showlegend=False)

    # --- Tabla ---
    df_show = df[["actual", "SOC", "SOC_opt", "AO"]].tail(24).copy()
    df_show.columns = ["Realizado", "SOC", "SOC_opt", "AO"]
    df_show = df_show.round(4)
    df_show.index = df_show.index.strftime("%Y-%m")
    df_show.insert(0, "Fecha", df_show.index)
    table = dash_table.DataTable(
        data=df_show.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_show.columns],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "fontSize": "12px", "padding": "4px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
        page_size=24,
    )

    return kpi1, kpi2, kpi3, kpi4, fig1, fig2, fig3, table


@app.callback(
    Output("t1-download", "data"),
    Input("t1-download-btn", "n_clicks"),
    State("t1-serie", "value"),
    State("t1-horizon", "value"),
    prevent_initial_call=True,
)
def download_oos(_, serie, h):
    df = load_forecasts(serie, h)
    if df is None:
        return None
    cols = ["actual", "SOC", "SOC_opt", "AO"] + [c for c in df.columns if c.startswith("BEST_")]
    df_dl = df[[c for c in cols if c in df.columns]].copy()
    df_dl.index = df_dl.index.strftime("%Y-%m")
    return dcc.send_data_frame(df_dl.to_excel, f"SOC_US_{serie}_h{h}.xlsx", index=True)


# ---------------------------------------------------------------------------
# Callbacks – Tab 2
# ---------------------------------------------------------------------------

@app.callback(
    [Output("t2-subplots", "figure"),
     Output("t2-best-models", "children")],
    [Input("t2-serie", "value"), Input("t2-horizon", "value")],
)
def update_tab2(serie, h):
    df_comp = load_components(serie, h)
    df_fc   = load_forecasts(serie, h)
    empty = go.Figure()
    if df_comp is None:
        empty.update_layout(annotations=[{"text": "Sin datos.", "showarrow": False}])
        return empty, html.P("Sin datos.")

    n_comp = len(COMPONENT_NAMES)
    fig = make_subplots(rows=n_comp, cols=1, shared_xaxes=True,
                        subplot_titles=COMPONENT_NAMES,
                        vertical_spacing=0.04)
    for i, comp in enumerate(COMPONENT_NAMES, 1):
        if comp not in df_comp.columns:
            continue
        fig.add_trace(go.Scatter(x=df_comp.index, y=df_comp[comp],
                                 mode="lines", line=dict(color=COLORS["actual"], width=1.5),
                                 name=comp, showlegend=False), row=i, col=1)
        if df_fc is not None:
            best_col = f"BEST_{comp}"
            if best_col in df_fc.columns:
                fig.add_trace(go.Scatter(x=df_fc.index, y=df_fc[best_col],
                                         mode="lines", line=dict(color=COLORS["soc"], width=1.5, dash="dash"),
                                         name=f"{comp} SOC", showlegend=False), row=i, col=1)
    fig.update_layout(height=700, template="plotly_white",
                      title=f"Componentes MODWT Haar – {serie} h={h}")

    # Tabla mejores modelos
    if df_fc is not None:
        rows = []
        for comp in COMPONENT_NAMES:
            col = f"BESTMODEL_{comp}"
            if col in df_fc.columns:
                rows.append({"Componente": comp, "Mejor modelo": str(df_fc[col].iloc[0])})
        if rows:
            tbl = dash_table.DataTable(
                data=rows,
                columns=[{"name": c, "id": c} for c in ["Componente", "Mejor modelo"]],
                style_cell={"textAlign": "center", "fontSize": "13px"},
                style_header={"fontWeight": "bold"},
            )
            return fig, tbl

    return fig, html.P("Sin datos de modelos.")


# ---------------------------------------------------------------------------
# Callbacks – Tab 3
# ---------------------------------------------------------------------------

@app.callback(
    [Output("t3-rmse-chart", "figure"),
     Output("t3-table", "children")],
    [Input("t3-serie", "value"),
     Input("t3-horizon", "value"),
     Input("t3-component", "value"),
     Input("t3-show-combos", "value")],
)
def update_tab3(serie, h, component, show_combos):
    empty = go.Figure()
    df = load_forecasts(serie, h)
    if df is None:
        empty.update_layout(annotations=[{"text": "Sin datos.", "showarrow": False}])
        return empty, html.P("Sin datos.")

    actual = df["actual"].values.astype(float) if "actual" in df.columns else None
    if actual is None:
        return empty, html.P("Sin datos de realizados.")

    combo_keys = {"C_MEAN", "C_MEDIAN", "C_TRIMEAN",
                  "C_DMSPE025", "C_DMSPE050", "C_DMSPE075", "C_DMSPE100"}
    prefix = f"{component}_"
    model_cols = [c for c in df.columns if c.startswith(prefix) and not c.startswith(f"{component}_BEST")]
    model_names = [c[len(prefix):] for c in model_cols]

    if not show_combos:
        filtered = [(c, n) for c, n in zip(model_cols, model_names) if n not in combo_keys]
        model_cols, model_names = zip(*filtered) if filtered else ([], [])

    rmse_vals = []
    for col, name in zip(model_cols, model_names):
        f = df[col].values.astype(float) if col in df.columns else np.full(len(actual), np.nan)
        rmse_vals.append({"Modelo": name, "RMSE": _rmse(actual, f)})

    df_rmse = pd.DataFrame(rmse_vals).dropna().sort_values("RMSE")

    is_combo = df_rmse["Modelo"].isin(combo_keys)
    colors = np.where(is_combo, "#3498DB", COLORS["soc"])

    fig = go.Figure(go.Bar(
        x=df_rmse["RMSE"], y=df_rmse["Modelo"],
        orientation="h", marker_color=colors,
        text=df_rmse["RMSE"].round(4), textposition="outside",
    ))
    fig.update_layout(template="plotly_white", height=max(400, len(df_rmse) * 18),
                      title=f"RMSE por modelo – {serie} {component} h={h}",
                      xaxis_title="RMSE", yaxis_autorange="reversed")

    tbl = dash_table.DataTable(
        data=df_rmse.head(20).round(5).to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_rmse.columns],
        style_cell={"textAlign": "center", "fontSize": "12px"},
        style_header={"fontWeight": "bold"},
    )
    return fig, tbl


# ---------------------------------------------------------------------------
# Callbacks – Tab 4
# ---------------------------------------------------------------------------

@app.callback(
    [Output("t4-heatmap", "figure"),
     Output("t4-table", "children")],
    Input("meta-interval", "n_intervals"),
)
def update_tab4(_):
    empty = go.Figure()
    df_m = load_metrics()

    all_rows = []
    for serie in US_SERIES_NAMES:
        for h in HORIZONS:
            row = {"serie": serie, "h": h, "ratio_SOC": np.nan, "ratio_SOC_opt": np.nan,
                   "rmse_AO": np.nan, "rmse_SOC": np.nan, "n_oos": np.nan}
            if df_m is not None:
                sub = df_m[(df_m["serie"] == serie) & (df_m["h"] == h)]
                if len(sub) > 0:
                    row.update(sub.iloc[0][["ratio_SOC", "ratio_SOC_opt",
                                            "rmse_AO", "rmse_SOC", "n_oos"]].to_dict())
            all_rows.append(row)
    df_all = pd.DataFrame(all_rows)

    # Heatmap: filas = series, columnas = horizontes
    z_soc = []
    z_text = []
    for serie in US_SERIES_NAMES:
        row_z, row_t = [], []
        for h in HORIZONS:
            sub = df_all[(df_all["serie"] == serie) & (df_all["h"] == h)]
            val = float(sub["ratio_SOC"].iloc[0]) if len(sub) > 0 else np.nan
            row_z.append(val)
            row_t.append(f"{val:.3f}" if not np.isnan(val) else "N/D")
        z_soc.append(row_z)
        z_text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z_soc,
        x=[f"h={h}" for h in HORIZONS],
        y=US_SERIES_NAMES,
        text=z_text, texttemplate="%{text}",
        colorscale="RdYlGn_r",
        zmid=1.0, zmin=0.5, zmax=1.5,
        colorbar=dict(title="RMSE SOC/AO"),
    ))
    fig.update_layout(template="plotly_white",
                      title="Ratio RMSE SOC/AO (verde < 1 = SOC supera AO)")

    # Tabla resumen
    df_all["ratio_SOC"]     = df_all["ratio_SOC"].round(3)
    df_all["ratio_SOC_opt"] = df_all["ratio_SOC_opt"].round(3)
    df_all["rmse_AO"]       = df_all["rmse_AO"].round(4)
    df_all["rmse_SOC"]      = df_all["rmse_SOC"].round(4)
    tbl = dash_table.DataTable(
        data=df_all.to_dict("records"),
        columns=[{"name": c, "id": c} for c in
                 ["serie", "h", "rmse_AO", "rmse_SOC", "ratio_SOC", "ratio_SOC_opt", "n_oos"]],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "fontSize": "12px"},
        style_header={"fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"filter_query": "{ratio_SOC} < 1", "column_id": "ratio_SOC"},
             "color": "green", "fontWeight": "bold"},
            {"if": {"filter_query": "{ratio_SOC} >= 1", "column_id": "ratio_SOC"},
             "color": "red"},
        ],
    )
    return fig, tbl


# ---------------------------------------------------------------------------
# Callbacks – Tab 5
# ---------------------------------------------------------------------------

@app.callback(
    [Output("t5-chart", "figure"),
     Output("t5-table", "children")],
    Input("t5-btn", "n_clicks"),
    State("t5-serie", "value"),
    State("t5-history", "value"),
    prevent_initial_call=True,
)
def update_tab5(_, serie, n_history):
    empty = go.Figure()
    empty.update_layout(annotations=[{"text": "Haz clic en 'Calcular pronostico'.", "showarrow": False}])
    try:
        df_future = forecast_future(serie, n_months=12)
        hist = last_observed_inflation(serie, n_history=n_history)
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(annotations=[{"text": f"Error: {e}", "showarrow": False}])
        return fig, html.P(f"Error: {e}", className="text-danger")

    if df_future.empty:
        return empty, html.P("Sin datos OOS para generar pronostico.")

    fig = go.Figure()
    # Historia
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines",
                              line=dict(color=COLORS["actual"], width=2), name="Historia"))
    # Banda 90%
    has_band = "upper_90" in df_future.columns and "lower_90" in df_future.columns
    if has_band:
        fig.add_traces([
            go.Scatter(x=df_future.index, y=df_future["upper_90"],
                       mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=df_future.index, y=df_future["lower_90"],
                       mode="lines", line=dict(width=0), fill="tonexty",
                       fillcolor="rgba(39,174,96,0.2)", name="Banda 90%", hoverinfo="skip"),
        ])
    fig.add_trace(go.Scatter(x=df_future.index, y=df_future["SOC"], mode="lines+markers",
                              line=dict(color=COLORS["soc"], width=2.5, dash="dash"),
                              name="SOC"))
    fig.add_trace(go.Scatter(x=df_future.index, y=df_future["SOC_opt"], mode="lines+markers",
                              line=dict(color=COLORS["soc_opt"], width=2, dash="dot"),
                              name="SOC_opt"))

    if hist.index[-1] < df_future.index[0]:
        fig.add_vline(x=hist.index[-1], line_dash="dash", line_color="gray",
                      annotation_text="Ultimo dato")

    fig.update_layout(template="plotly_white", hovermode="x unified",
                      title=f"Pronostico SOC – {serie} (proximos 12 meses)",
                      xaxis_title="Fecha", yaxis_title="Inflacion (%)")

    # Tabla
    cols_show = ["h", "SOC", "SOC_opt"]
    if has_band:
        cols_show += ["lower_90", "upper_90"]
    for comp in COMPONENT_NAMES:
        col_best = f"best_{comp}"
        if col_best in df_future.columns:
            cols_show.append(col_best)
    df_show = df_future[[c for c in cols_show if c in df_future.columns]].copy()
    df_show.index = df_future.index.strftime("%Y-%m")
    df_show.insert(0, "Fecha", df_show.index)
    tbl = dash_table.DataTable(
        data=df_show.round(4).to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_show.columns],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "fontSize": "12px"},
        style_header={"fontWeight": "bold"},
    )
    return fig, tbl


# ---------------------------------------------------------------------------
# Callbacks – Tab 6
# ---------------------------------------------------------------------------

def _model_type(name: str) -> str:
    if name in ("AR_AIC", "AR_SIC"):
        return "Autorregresivo"
    if name in ("PC1", "PC2", "PC3"):
        return "Phillips Curve"
    if name.startswith("BIV_"):
        return "Bivariado"
    if name in ("PCA", "PLS1", "PLS2"):
        return "Reduccion dim."
    if name in ("LASSO", "ELASTICNET", "RIDGE"):
        return "Penalizado"
    if name.startswith("C_DMSPE"):
        return "Comb. DMSPE"
    if name.startswith("C_"):
        return "Combinacion"
    return "Otro"


@app.callback(
    Output("t6-results", "children"),
    Input("t6-serie",   "value"),
    Input("t6-horizon", "value"),
)
def update_tab6(serie, h):
    if serie is None or h is None:
        return html.P("Selecciona una serie y horizonte.", className="text-muted small")

    df = load_forecasts(serie, h)
    if df is None:
        return dbc.Alert(
            f"Sin resultados para '{serie}' h={h}. "
            "Ejecuta: python -m soc_us.fit_soc_us",
            color="warning",
        )

    actual   = df["actual"].values.astype(float)
    n_obs    = int(np.sum(~np.isnan(actual)))
    date_min = df.index.min().strftime("%Y-%m")
    date_max = df.index.max().strftime("%Y-%m")

    soc_arr = df["SOC"].values.astype(float) if "SOC" in df.columns else np.full_like(actual, np.nan)
    ao_arr  = df["AO"].values.astype(float)  if "AO"  in df.columns else np.full_like(actual, np.nan)
    r_soc   = _rmse(actual, soc_arr)
    r_ao    = _rmse(actual, ao_arr)
    ratio   = r_soc / r_ao if (np.isfinite(r_ao) and r_ao > 0) else np.nan

    kpi_row = dbc.Row([
        dbc.Col(_kpi_card("Obs. OOS",    str(n_obs),                 "secondary"), md=3),
        dbc.Col(_kpi_card("Periodo OOS", f"{date_min} / {date_max}", "secondary"), md=3),
        dbc.Col(_kpi_card("RMSE SOC",    f"{r_soc:.4f}",             "success"),   md=3),
        dbc.Col(_kpi_card(
            "Ratio SOC/AO",
            f"{ratio:.3f}" if np.isfinite(ratio) else "—",
            "success" if np.isfinite(ratio) and ratio < 1 else "danger",
        ), md=3),
    ], className="mb-3 g-2")

    _scale = {"D1": "2-4m", "D2": "4-8m", "D3": "8-16m",
              "D4": "16-32m", "D5": "32-64m", "S5": ">64m"}
    best_rows = []
    for comp in COMPONENT_NAMES:
        bm_col = f"BESTMODEL_{comp}"
        bc_col = f"BEST_{comp}"
        model_name = str(df[bm_col].iloc[0]) if (bm_col in df.columns and len(df) > 0) else "—"
        rmse_str = (f"{_rmse(actual, df[bc_col].values.astype(float)):.4f}"
                    if bc_col in df.columns else "—")
        best_rows.append({
            "Componente": comp, "Escala": _scale.get(comp, ""),
            "Mejor modelo": model_name, "Tipo": _model_type(model_name),
            "RMSE OOS": rmse_str,
        })

    best_table = dash_table.DataTable(
        data=best_rows,
        columns=[{"name": c, "id": c}
                 for c in ["Componente", "Escala", "Mejor modelo", "Tipo", "RMSE OOS"]],
        style_cell={"textAlign": "center", "fontSize": "12px", "padding": "4px 8px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
        style_data_conditional=[
            {"if": {"filter_query": '{Componente} = "D1"'}, "fontStyle": "italic", "color": "#aaa"},
            {"if": {"filter_query": '{Componente} = "S5"'}, "fontWeight": "bold"},
        ] + [
            {"if": {"filter_query": f'{{Tipo}} = "{t}"', "column_id": "Tipo"},
             "color": c, "fontWeight": "bold"}
            for t, c in _TYPE_COLORS.items()
        ],
        style_table={"overflowX": "auto"},
    )

    # Bar chart: RMSE medio por tipo
    type_rmse: dict[str, list[float]] = {}
    for comp in COMPONENT_NAMES:
        for col in [c for c in df.columns if c.startswith(f"{comp}_")]:
            model_name = col[len(comp) + 1:]
            r = _rmse(actual, df[col].values.astype(float))
            if np.isfinite(r):
                tipo = _model_type(model_name)
                type_rmse.setdefault(tipo, []).append(r)

    if type_rmse:
        rows_t = sorted(
            [{"Tipo": t, "RMSE medio": float(np.mean(v)), "N": len(v)}
             for t, v in type_rmse.items()],
            key=lambda x: x["RMSE medio"],
        )
        df_t = pd.DataFrame(rows_t)
        fig_type = go.Figure(go.Bar(
            x=df_t["RMSE medio"], y=df_t["Tipo"], orientation="h",
            marker_color=[_TYPE_COLORS.get(t, "#95A5A6") for t in df_t["Tipo"]],
            text=[f"{v:.4f}  (n={n})" for v, n in zip(df_t["RMSE medio"], df_t["N"])],
            textposition="outside",
        ))
        fig_type.update_layout(
            template="plotly_white", height=300,
            title=dict(text="RMSE medio OOS por tipo de modelo (promediado sobre componentes wavelet)",
                       font=dict(size=12)),
            xaxis_title="RMSE medio", yaxis_autorange="reversed",
            margin=dict(l=140, r=130, t=50, b=40),
        )
        chart_section = dcc.Graph(figure=fig_type, config={"displayModeBar": False})
    else:
        chart_section = html.P("No hay datos de modelos individuales.", className="text-muted small")

    return html.Div([
        kpi_row,
        dbc.Row([
            dbc.Col([
                html.H6("Mejor modelo por componente wavelet", className="fw-bold small mb-1"),
                html.P(
                    f"Serie: {serie}  |  Horizonte: {HORIZON_LABELS.get(h, f'h={h}')}  |  OOS desde {OOS_START}",
                    className="text-muted small mb-2",
                ),
                best_table,
            ], md=6),
            dbc.Col([
                html.H6("Desempeno OOS por tipo de modelo", className="fw-bold small mb-1"),
                html.P("RMSE promediado sobre todos los componentes y modelos dentro de cada tipo.",
                       className="text-muted small mb-2"),
                chart_section,
            ], md=6),
        ]),
    ])


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, port=8052, host="0.0.0.0")
