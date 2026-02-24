"""
dash_app.py – App Plotly Dash SOC para pronostico del INPC Mexico.

Puerto: 8051 (sin conflicto con Streamlit en 8501)
Inicio: python soc/dash_app.py

Tab 1 – Pronostico SOC:
  * Serie temporal con bandas de error +/-1.65 sigma (rolling 24 meses)
  * Scatter Observado vs. Pronosticado con linea 45°, R2 y MAE
  * Grafica de residuos en el tiempo
  * Acordeon con especificacion del modelo

Tab 2 – Componentes Wavelet:  subplots D1..D5, S5
Tab 3 – Comparacion de Modelos: ranking RMSE por componente
Tab 4 – Metricas Globales: heatmap ratio SOC/AO
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

import dash
from dash import dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from soc.config import (
    HORIZONS, INPC_SERIES_NAMES, SAFE_NAMES, COMPONENT_NAMES,
    FORECASTS_DIR, COMPONENTS_DIR, METRICS_DIR, RESULTS_DIR,
    OOS_START, AO_WINDOW, J_LEVELS,
)

# ---------------------------------------------------------------------------
# Helpers de carga y calculo
# ---------------------------------------------------------------------------

def _safe(name: str) -> str:
    return SAFE_NAMES.get(name, name.replace(" ", "_"))


def load_forecasts(serie: str, h: int) -> pd.DataFrame | None:
    path = os.path.join(FORECASTS_DIR, f"{_safe(serie)}_h{h}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def load_components(serie: str, h: int) -> pd.DataFrame | None:
    path = os.path.join(COMPONENTS_DIR, f"{_safe(serie)}_h{h}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def load_metrics() -> pd.DataFrame | None:
    path = os.path.join(METRICS_DIR, "all_metrics.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def load_metadata() -> dict:
    path = os.path.join(RESULTS_DIR, "run_metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    mask = ~np.isnan(actual) & ~np.isnan(forecast)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((actual[mask] - forecast[mask]) ** 2)))


def _mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    mask = ~np.isnan(actual) & ~np.isnan(forecast)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual[mask] - forecast[mask])))


def _r2(actual: np.ndarray, forecast: np.ndarray) -> float:
    mask = ~np.isnan(actual) & ~np.isnan(forecast)
    if mask.sum() < 2:
        return np.nan
    ss_res = np.sum((actual[mask] - forecast[mask]) ** 2)
    ss_tot = np.sum((actual[mask] - actual[mask].mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def _rolling_error_std(actual: np.ndarray, forecast: np.ndarray, window: int = 24) -> np.ndarray:
    """Desviacion estandar rolling de los errores (actual - forecast)."""
    errors = actual - forecast
    s = pd.Series(errors)
    return s.rolling(window, min_periods=max(6, window // 4)).std().values


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="SOC - Pronostico INPC Mexico",
    suppress_callback_exceptions=True,
)

COLORS = {
    "actual":  "#2C3E50",
    "soc":     "#27AE60",
    "soc_opt": "#2980B9",
    "ao":      "#E74C3C",
    "bg":      "#F8F9FA",
    "error":   "#8E44AD",
}

HORIZON_LABELS = {1: "1 mes", 6: "6 meses", 12: "12 meses"}

ROLLING_STD_WINDOW = 24   # meses para bandas de error rolling
CI_MULTIPLIER      = 1.65  # 90% intervalo de confianza (normal)

# ---------------------------------------------------------------------------
# Navbar
# ---------------------------------------------------------------------------

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("SOC - Pronostico INPC Mexico", className="fw-bold"),
        html.Span(
            "Metodo: Sum of the Cycles (Verona, 2026) | Puerto 8051",
            className="text-white ms-3 small",
        ),
    ], fluid=True),
    color="primary", dark=True, className="mb-3",
)

# ---------------------------------------------------------------------------
# Acordeon de especificacion del modelo (estatico)
# ---------------------------------------------------------------------------

MODEL_SPEC_ACCORDION = dbc.Accordion([
    dbc.AccordionItem(
        title="Especificacion del modelo SOC",
        children=[
            dbc.Row([
                dbc.Col([
                    html.H6("1. Transformacion de inflacion", className="fw-bold"),
                    html.P([
                        html.Code("pi^h_t = (1200/h) * ln(P_t / P_{t-h})"),
                        html.Br(),
                        "Inflacion anualizada a horizonte h meses. "
                        "Se aplica a cada una de las 16 series de nivel de precios del INPC.",
                    ], className="small"),

                    html.H6("2. Descomposicion MODWT Haar", className="fw-bold mt-2"),
                    html.P([
                        f"5 niveles de descomposicion (J={J_LEVELS}) via ",
                        html.Code("pywt.swt(..., norm=True)"),
                        ", equivalente al MODWT. Genera 6 componentes:",
                    ], className="small mb-1"),
                    html.Ul([
                        html.Li("D1: ciclos de 2-4 meses (ruido de alta frecuencia)", className="small"),
                        html.Li("D2: ciclos de 4-8 meses", className="small"),
                        html.Li("D3: ciclos de 8-16 meses", className="small"),
                        html.Li("D4: ciclos de 16-32 meses", className="small"),
                        html.Li("D5: ciclos de 32-64 meses", className="small"),
                        html.Li("S5: tendencia de largo plazo (>64 meses)", className="small"),
                    ]),
                    html.P(
                        "Verificacion: la suma de los 6 componentes reconstruye exactamente "
                        "la serie original (error < 1e-6 por observacion).",
                        className="small text-muted",
                    ),
                ], md=6),

                dbc.Col([
                    html.H6("3. Modelos de pronostico por componente", className="fw-bold"),
                    html.P(
                        "Se ajustan individualmente para cada componente wavelet "
                        "usando pronostico directo (no iterado) a h pasos:",
                        className="small mb-1",
                    ),
                    html.Ul([
                        html.Li([html.Strong("AR: "), "AR-AIC, AR-SIC (hasta 12 rezagos)"], className="small"),
                        html.Li([html.Strong("Phillips Curve: "), "PC1 (CETES28), PC2 (USDMXN), PC3 (M2_REAL)"], className="small"),
                        html.Li([html.Strong("Bivariados: "), "AR(1) + cada predictor macro por separado"], className="small"),
                        html.Li([html.Strong("Reduccion de dimension: "), "PCA (3 comp.), PLS (2 comp.)"], className="small"),
                        html.Li([html.Strong("Penalizados: "), "LASSO-CV, ElasticNet-CV, Ridge-CV"], className="small"),
                        html.Li([html.Strong("Combinaciones: "), "Media, Mediana, Media recortada (10%), DMSPE(theta=0.25/0.50/0.75/1.00)"], className="small"),
                    ]),

                    html.H6("4. Evaluacion y construccion del SOC", className="fw-bold mt-2"),
                    html.Ul([
                        html.Li([
                            html.Strong("Ventana expansiva OOS: "),
                            f"desde {OOS_START}. En cada periodo t se entrena con todos los datos hasta t "
                            "y se pronostica t+h."
                        ], className="small"),
                        html.Li([
                            html.Strong("Seleccion del mejor modelo: "),
                            "se elige el de menor RMSE acumulado OOS para cada componente."
                        ], className="small"),
                        html.Li([
                            html.Strong("SOC: "),
                            "suma de los mejores pronosticos de los 6 componentes (D1..D5 + S5)."
                        ], className="small"),
                        html.Li([
                            html.Strong("SOC_opt: "),
                            "igual que SOC pero excluye D1 (ruido de alta frecuencia)."
                        ], className="small"),
                        html.Li([
                            html.Strong("Benchmark AO: "),
                            f"media de los ultimos {AO_WINDOW} meses de inflacion realizada."
                        ], className="small"),
                    ]),
                    html.P([
                        html.Strong("Predictores macro: "),
                        "M2_REAL, CETES28, TIIE28, TMS (diferencial de tasas), MBONO10Y, "
                        "USDMXN, retorno IPC_BMV, nivel de Energeticos. "
                        "Fuente: Banxico SIE. WTI disponible si se agrega FRED_API_KEY al .env.",
                    ], className="small text-muted mt-1"),
                ], md=6),
            ]),
        ],
    ),
], start_collapsed=True, className="mb-3")

# ---------------------------------------------------------------------------
# Tab 1 layout
# ---------------------------------------------------------------------------

def _kpi_card(label: str, value: str, color: str = "primary") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(label, className="card-subtitle text-muted small mb-1"),
            html.H5(value, className=f"text-{color} mb-0"),
        ]),
        className="text-center shadow-sm h-100",
    )


tab1_layout = dbc.Container([
    # Controles
    dbc.Row([
        dbc.Col([
            html.Label("Serie INPC", className="fw-bold"),
            dcc.Dropdown(
                id="t1-serie",
                options=[{"label": s.strip(), "value": s} for s in INPC_SERIES_NAMES],
                value=INPC_SERIES_NAMES[0],
                clearable=False,
            ),
        ], md=6),
        dbc.Col([
            html.Label("Horizonte de pronostico", className="fw-bold"),
            dbc.RadioItems(
                id="t1-horizon",
                options=[{"label": HORIZON_LABELS[h], "value": h} for h in HORIZONS],
                value=1,
                inline=True,
                className="mt-2",
            ),
        ], md=6),
    ], className="mb-3"),

    # KPIs: 4 tarjetas
    dbc.Row([
        dbc.Col(id="kpi-rmse-soc",    md=3),
        dbc.Col(id="kpi-rmse-soc-opt",md=3),
        dbc.Col(id="kpi-rmse-ao",     md=3),
        dbc.Col(id="kpi-ratio",       md=3),
    ], className="mb-3 g-2"),

    # Grafica principal: serie temporal + bandas de error
    html.H6("Serie temporal con bandas de confianza (90%, rolling 24 meses)",
            className="text-muted mb-1"),
    dcc.Graph(id="t1-chart", style={"height": "430px"}),

    # Fila: scatter obs vs pred (izq) + residuos en tiempo (der)
    dbc.Row([
        dbc.Col([
            html.H6("Observado vs. Pronosticado", className="text-muted mb-1"),
            dcc.Graph(id="t1-scatter", style={"height": "380px"}),
        ], md=6),
        dbc.Col([
            html.H6("Errores de pronostico en el tiempo (actual - SOC)",
                    className="text-muted mb-1"),
            dcc.Graph(id="t1-errors", style={"height": "380px"}),
        ], md=6),
    ], className="mt-3"),

    # Acordeon especificacion
    html.Div(MODEL_SPEC_ACCORDION, className="mt-4"),

    # Tabla ultimos 24 periodos
    html.H6("Ultimos 24 periodos OOS", className="mt-2 mb-2 fw-bold"),
    html.Div(id="t1-table"),
], fluid=True)


# ---------------------------------------------------------------------------
# Tab 2 layout
# ---------------------------------------------------------------------------

tab2_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Serie INPC", className="fw-bold"),
            dcc.Dropdown(
                id="t2-serie",
                options=[{"label": s.strip(), "value": s} for s in INPC_SERIES_NAMES],
                value=INPC_SERIES_NAMES[0],
                clearable=False,
            ),
        ], md=6),
        dbc.Col([
            html.Label("Horizonte", className="fw-bold"),
            dbc.RadioItems(
                id="t2-horizon",
                options=[{"label": HORIZON_LABELS[h], "value": h} for h in HORIZONS],
                value=1,
                inline=True,
                className="mt-2",
            ),
        ], md=6),
    ], className="mb-3"),

    dcc.Graph(id="t2-subplots", style={"height": "700px"}),

    html.H6("Mejor modelo seleccionado por componente (menor RMSE OOS)",
            className="mt-3 mb-2 fw-bold"),
    html.Div(id="t2-best-models"),
], fluid=True)


# ---------------------------------------------------------------------------
# Tab 3 layout
# ---------------------------------------------------------------------------

tab3_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Serie INPC", className="fw-bold"),
            dcc.Dropdown(
                id="t3-serie",
                options=[{"label": s.strip(), "value": s} for s in INPC_SERIES_NAMES],
                value=INPC_SERIES_NAMES[0],
                clearable=False,
            ),
        ], md=4),
        dbc.Col([
            html.Label("Horizonte", className="fw-bold"),
            dbc.RadioItems(
                id="t3-horizon",
                options=[{"label": HORIZON_LABELS[h], "value": h} for h in HORIZONS],
                value=1,
                inline=True,
                className="mt-2",
            ),
        ], md=4),
        dbc.Col([
            html.Label("Componente wavelet", className="fw-bold"),
            dcc.Dropdown(
                id="t3-component",
                options=[{"label": c, "value": c} for c in COMPONENT_NAMES],
                value=COMPONENT_NAMES[0],
                clearable=False,
            ),
        ], md=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Switch(id="t3-show-combos", label="Incluir combinaciones de modelos", value=True),
        ]),
    ], className="mb-2"),

    dcc.Graph(id="t3-rmse-bar", style={"height": "520px"}),
], fluid=True)


# ---------------------------------------------------------------------------
# Tab 4 layout
# ---------------------------------------------------------------------------

tab4_layout = dbc.Container([
    html.H6("Ratio RMSE SOC / AO por serie y horizonte (verde = SOC gana al benchmark)",
            className="mb-2"),
    dcc.Graph(id="t4-heatmap", style={"height": "520px"}),

    html.H6("Resumen por serie", className="mt-4 mb-2 fw-bold"),
    html.Div(id="t4-summary"),

    dbc.Row([
        dbc.Col([
            dbc.Button("Exportar metricas a Excel", id="t4-export-btn",
                       color="success", className="mt-3"),
            dcc.Download(id="t4-download"),
        ]),
    ]),
], fluid=True)


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------

app.layout = html.Div([
    navbar,
    dbc.Container([
        dbc.Tabs([
            dbc.Tab(tab1_layout, label="Pronostico SOC",        tab_id="tab1"),
            dbc.Tab(tab2_layout, label="Componentes Wavelet",   tab_id="tab2"),
            dbc.Tab(tab3_layout, label="Comparacion de Modelos",tab_id="tab3"),
            dbc.Tab(tab4_layout, label="Metricas Globales",     tab_id="tab4"),
        ], id="main-tabs", active_tab="tab1"),
    ], fluid=True),
])


# ---------------------------------------------------------------------------
# Callbacks – Tab 1
# ---------------------------------------------------------------------------

@app.callback(
    Output("t1-chart",         "figure"),
    Output("t1-scatter",       "figure"),
    Output("t1-errors",        "figure"),
    Output("kpi-rmse-soc",     "children"),
    Output("kpi-rmse-soc-opt", "children"),
    Output("kpi-rmse-ao",      "children"),
    Output("kpi-ratio",        "children"),
    Output("t1-table",         "children"),
    Input("t1-serie",    "value"),
    Input("t1-horizon",  "value"),
)
def update_tab1(serie, h):
    # ---  Placeholder cuando no hay datos  ---
    def _empty_fig(msg: str) -> go.Figure:
        f = go.Figure()
        f.update_layout(
            title=msg, paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return f

    df = load_forecasts(serie, h)
    if df is None:
        msg = f"Sin datos para '{serie.strip()}' h={h} — ejecuta fit_soc.py"
        na_card = _kpi_card("—", "N/D").children
        return (
            _empty_fig(msg), _empty_fig(msg), _empty_fig(msg),
            na_card, na_card, na_card, na_card,
            html.P("Sin datos. Ejecuta: python -m soc.fit_soc --series 'Indice General' --horizon 1"),
        )

    actual  = df["actual"].values.astype(float)
    soc     = df["SOC"].values.astype(float)
    soc_opt = df["SOC_opt"].values.astype(float)
    ao      = df["AO"].values.astype(float)
    dates   = df.index

    # --- Metricas globales ---
    r_soc  = _rmse(actual, soc)
    r_opt  = _rmse(actual, soc_opt)
    r_ao   = _rmse(actual, ao)
    ratio  = r_soc / r_ao if (np.isfinite(r_ao) and r_ao > 0) else np.nan

    ratio_color = "success" if (np.isfinite(ratio) and ratio < 1) else "danger"

    kpi_soc    = _kpi_card("RMSE SOC",      f"{r_soc:.4f}", "success").children
    kpi_opt    = _kpi_card("RMSE SOC_opt",  f"{r_opt:.4f}", "info").children
    kpi_ao     = _kpi_card("RMSE AO (bench.)", f"{r_ao:.4f}", "danger").children
    kpi_ratio  = _kpi_card("Ratio SOC/AO",  f"{ratio:.3f}", ratio_color).children

    # =========================================================
    # GRAFICA 1: Serie temporal + bandas de error rolling
    # =========================================================
    errors_soc = actual - soc
    roll_std   = _rolling_error_std(actual, soc, window=ROLLING_STD_WINDOW)
    upper_band = soc + CI_MULTIPLIER * roll_std
    lower_band = soc - CI_MULTIPLIER * roll_std

    fig_ts = go.Figure()

    # Banda SOC (fill)
    fig_ts.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(upper_band) + list(lower_band[::-1]),
        fill="toself",
        fillcolor="rgba(39, 174, 96, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"SOC +/-{CI_MULTIPLIER}σ (rolling {ROLLING_STD_WINDOW}m)",
        hoverinfo="skip",
        showlegend=True,
    ))

    # Lineas principales
    fig_ts.add_trace(go.Scatter(
        x=dates, y=actual, name="Realizado",
        line=dict(color=COLORS["actual"], width=2),
    ))
    fig_ts.add_trace(go.Scatter(
        x=dates, y=soc, name="SOC",
        line=dict(color=COLORS["soc"], width=2, dash="dash"),
    ))
    fig_ts.add_trace(go.Scatter(
        x=dates, y=soc_opt, name="SOC_opt",
        line=dict(color=COLORS["soc_opt"], width=1.8, dash="dot"),
    ))
    fig_ts.add_trace(go.Scatter(
        x=dates, y=ao, name="AO (benchmark)",
        line=dict(color=COLORS["ao"], width=1.5, dash="longdash"),
    ))

    fig_ts.update_layout(
        title=dict(
            text=f"Inflacion anualizada — {serie.strip()} (h={h} mes{'es' if h > 1 else ''})",
            font=dict(size=14),
        ),
        xaxis_title="Fecha",
        yaxis_title="Inflacion anualizada (%)",
        hovermode="x unified",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=70, b=40),
    )

    # =========================================================
    # GRAFICA 2: Observado vs. Pronosticado (scatter)
    # =========================================================
    lim_min = float(np.nanmin([actual, soc, soc_opt, ao])) * 1.05
    lim_max = float(np.nanmax([actual, soc, soc_opt, ao])) * 1.05
    diag = [min(lim_min, lim_max), max(lim_min, lim_max)]

    fig_sc = go.Figure()

    # Linea 45 grados (pronostico perfecto)
    fig_sc.add_trace(go.Scatter(
        x=diag, y=diag, mode="lines",
        line=dict(color="gray", dash="dash", width=1),
        name="Pronostico perfecto",
        hoverinfo="skip",
    ))

    r2_soc = _r2(actual, soc)
    mae_soc = _mae(actual, soc)
    r2_opt = _r2(actual, soc_opt)
    mae_opt = _mae(actual, soc_opt)
    r2_ao  = _r2(actual, ao)
    mae_ao  = _mae(actual, ao)

    for y_vals, name, color, r2, mae in [
        (soc,     f"SOC (R²={r2_soc:.3f}, MAE={mae_soc:.3f})",     COLORS["soc"],     r2_soc, mae_soc),
        (soc_opt, f"SOC_opt (R²={r2_opt:.3f}, MAE={mae_opt:.3f})", COLORS["soc_opt"], r2_opt, mae_opt),
        (ao,      f"AO (R²={r2_ao:.3f}, MAE={mae_ao:.3f})",        COLORS["ao"],      r2_ao,  mae_ao),
    ]:
        mask = ~np.isnan(actual) & ~np.isnan(y_vals)
        fig_sc.add_trace(go.Scatter(
            x=actual[mask], y=y_vals[mask],
            mode="markers",
            name=name,
            marker=dict(color=color, size=4, opacity=0.65),
            customdata=dates[mask].strftime("%Y-%m"),
            hovertemplate="Fecha: %{customdata}<br>Realizado: %{x:.3f}<br>Pronostico: %{y:.3f}<extra></extra>",
        ))

    fig_sc.update_layout(
        title=dict(text="Observado vs. Pronosticado", font=dict(size=13)),
        xaxis_title="Inflacion realizada (%)",
        yaxis_title="Inflacion pronosticada (%)",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        legend=dict(orientation="v", font=dict(size=10)),
        margin=dict(t=50, b=40),
    )
    # Ejes cuadrados
    fig_sc.update_xaxes(range=diag)
    fig_sc.update_yaxes(range=diag)

    # =========================================================
    # GRAFICA 3: Errores de pronostico en el tiempo
    # =========================================================
    fig_err = go.Figure()

    # Linea cero
    fig_err.add_hline(y=0, line_width=1, line_dash="solid", line_color="gray")

    # Banda 1-sigma rolling
    fig_err.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(roll_std) + list(-roll_std[::-1]),
        fill="toself",
        fillcolor="rgba(39, 174, 96, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="+/-1σ rolling",
        hoverinfo="skip",
    ))

    # Residuos SOC
    fig_err.add_trace(go.Scatter(
        x=dates, y=errors_soc,
        mode="lines",
        name="Error SOC",
        line=dict(color=COLORS["soc"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(39, 174, 96, 0.08)",
    ))

    # Residuos AO (referencia)
    fig_err.add_trace(go.Scatter(
        x=dates, y=actual - ao,
        mode="lines",
        name="Error AO",
        line=dict(color=COLORS["ao"], width=1.2, dash="dot"),
    ))

    # RMSE recursivo del SOC (acumulado)
    cum_rmse_soc = []
    for t in range(len(actual)):
        mask = ~np.isnan(actual[:t+1]) & ~np.isnan(soc[:t+1])
        if mask.sum() == 0:
            cum_rmse_soc.append(np.nan)
        else:
            cum_rmse_soc.append(float(np.sqrt(np.mean((actual[:t+1][mask] - soc[:t+1][mask])**2))))
    cum_rmse_ao = []
    for t in range(len(actual)):
        mask = ~np.isnan(actual[:t+1]) & ~np.isnan(ao[:t+1])
        if mask.sum() == 0:
            cum_rmse_ao.append(np.nan)
        else:
            cum_rmse_ao.append(float(np.sqrt(np.mean((actual[:t+1][mask] - ao[:t+1][mask])**2))))

    fig_err.add_trace(go.Scatter(
        x=dates, y=cum_rmse_soc,
        mode="lines", name="RMSE recursivo SOC",
        line=dict(color="#1A5276", width=1.8, dash="longdash"),
        yaxis="y2",
    ))
    fig_err.add_trace(go.Scatter(
        x=dates, y=cum_rmse_ao,
        mode="lines", name="RMSE recursivo AO",
        line=dict(color="#922B21", width=1.5, dash="longdash"),
        yaxis="y2",
    ))

    fig_err.update_layout(
        title=dict(text="Residuos y RMSE recursivo", font=dict(size=13)),
        xaxis_title="Fecha",
        yaxis=dict(title="Error (pp)", zeroline=True),
        yaxis2=dict(
            title="RMSE acumulado",
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
        ),
        hovermode="x unified",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        legend=dict(orientation="h", font=dict(size=9), y=-0.25),
        margin=dict(t=50, b=60),
    )

    # =========================================================
    # TABLA: ultimos 24 periodos
    # =========================================================
    n_show = min(24, len(df))
    df_show = df.tail(n_show)[["actual", "SOC", "SOC_opt", "AO"]].copy()
    df_show["Error_SOC"] = (df_show["actual"] - df_show["SOC"]).round(4)
    df_show["Error_AO"]  = (df_show["actual"] - df_show["AO"]).round(4)
    df_show.index = df_show.index.strftime("%Y-%m")
    df_show = df_show.round(4).reset_index()
    df_show.columns = ["Fecha", "Realizado", "SOC", "SOC_opt", "AO",
                        "Error SOC", "Error AO"]

    table = dash_table.DataTable(
        data=df_show.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_show.columns],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "fontSize": "12px", "padding": "4px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
        style_data_conditional=[
            # SOC mas cercano al realizado que AO: verde
            {"if": {
                "filter_query": "{Error SOC} < {Error AO} && {Error SOC} > -{Error AO}",
                "column_id": "SOC",
             }, "color": "#27AE60", "fontWeight": "bold"},
            # Errores positivos = subestimacion
            {"if": {"filter_query": "{Error SOC} > 0", "column_id": "Error SOC"},
             "color": "#E74C3C"},
            {"if": {"filter_query": "{Error SOC} < 0", "column_id": "Error SOC"},
             "color": "#2980B9"},
        ],
        page_size=24,
        sort_action="native",
    )

    return (
        fig_ts, fig_sc, fig_err,
        kpi_soc, kpi_opt, kpi_ao, kpi_ratio,
        table,
    )


# ---------------------------------------------------------------------------
# Callbacks – Tab 2
# ---------------------------------------------------------------------------

@app.callback(
    Output("t2-subplots",   "figure"),
    Output("t2-best-models","children"),
    Input("t2-serie",   "value"),
    Input("t2-horizon", "value"),
)
def update_tab2(serie, h):
    df_comp = load_components(serie, h)
    df_fc   = load_forecasts(serie, h)

    if df_comp is None:
        f = go.Figure()
        f.update_layout(title="Sin datos — ejecuta fit_soc.py")
        return f, html.P("Sin datos.")

    n_comp = len(COMPONENT_NAMES)
    # Descripciones de escala de frecuencia
    scale_desc = {
        "D1": "2-4 meses (ruido)",
        "D2": "4-8 meses",
        "D3": "8-16 meses",
        "D4": "16-32 meses",
        "D5": "32-64 meses",
        "S5": ">64 meses (tendencia)",
    }
    subtitles = [f"{c}  [{scale_desc.get(c,'')}]" for c in COMPONENT_NAMES]

    fig = make_subplots(
        rows=n_comp, cols=1,
        shared_xaxes=True,
        subplot_titles=subtitles,
        vertical_spacing=0.035,
    )

    comp_colors = px.colors.qualitative.Safe
    for i, comp_name in enumerate(COMPONENT_NAMES):
        if comp_name not in df_comp.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df_comp.index,
                y=df_comp[comp_name],
                name=comp_name,
                line=dict(color=comp_colors[i % len(comp_colors)], width=1.4),
            ),
            row=i + 1, col=1,
        )

    fig.update_layout(
        title=dict(
            text=f"Descomposicion MODWT Haar — {serie.strip()}  (h={h})",
            font=dict(size=14),
        ),
        height=720,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        hovermode="x unified",
        showlegend=False,
    )

    # Tabla mejor modelo por componente (con RMSE)
    if df_fc is not None:
        best_cols = [c for c in df_fc.columns if c.startswith("BESTMODEL_")]
        if best_cols:
            actual = df_fc["actual"].values.astype(float)
            rows_tab = []
            for bc in best_cols:
                comp = bc.replace("BESTMODEL_", "")
                model_name = str(df_fc[bc].iloc[0]) if len(df_fc) > 0 else "—"
                # RMSE del mejor modelo para ese componente
                best_fc_col = f"BEST_{comp}"
                if best_fc_col in df_fc.columns:
                    r = _rmse(actual, df_fc[best_fc_col].values.astype(float))
                    rmse_str = f"{r:.4f}" if np.isfinite(r) else "—"
                else:
                    rmse_str = "—"
                rows_tab.append({
                    "Componente": comp,
                    "Escala": scale_desc.get(comp, ""),
                    "Mejor modelo (menor RMSE OOS)": model_name,
                    "RMSE componente": rmse_str,
                })
            df_tab = pd.DataFrame(rows_tab)
            table = dash_table.DataTable(
                data=df_tab.to_dict("records"),
                columns=[{"name": c, "id": c} for c in df_tab.columns],
                style_cell={"textAlign": "center", "fontSize": "13px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
                style_data_conditional=[
                    {"if": {"filter_query": '{Componente} = "D1"'},
                     "fontStyle": "italic", "color": "#666"},
                ],
            )
        else:
            table = html.P("Informacion de modelos no disponible.")
    else:
        table = html.P("Sin datos de pronosticos.")

    return fig, table


# ---------------------------------------------------------------------------
# Callbacks – Tab 3
# ---------------------------------------------------------------------------

@app.callback(
    Output("t3-rmse-bar", "figure"),
    Input("t3-serie",       "value"),
    Input("t3-horizon",     "value"),
    Input("t3-component",   "value"),
    Input("t3-show-combos", "value"),
)
def update_tab3(serie, h, comp_name, show_combos):
    df = load_forecasts(serie, h)
    if df is None:
        f = go.Figure()
        f.update_layout(title="Sin datos — ejecuta fit_soc.py")
        return f

    actual = df["actual"].values.astype(float)
    prefix = f"{comp_name}_"
    model_cols = [c for c in df.columns if c.startswith(prefix)]

    COMBO_PREFIXES = ("C_",)
    best_model_name = (
        str(df[f"BESTMODEL_{comp_name}"].iloc[0])
        if f"BESTMODEL_{comp_name}" in df.columns else None
    )

    rmse_data = []
    for col in model_cols:
        model_name = col[len(prefix):]
        is_combo = model_name.startswith(COMBO_PREFIXES)
        if is_combo and not show_combos:
            continue
        f_arr = df[col].values.astype(float)
        r = _rmse(actual, f_arr)
        if np.isfinite(r):
            rmse_data.append({
                "model": model_name,
                "rmse": r,
                "is_combo": is_combo,
                "is_best": (model_name == best_model_name),
            })

    if not rmse_data:
        f = go.Figure()
        f.update_layout(title=f"Sin modelos con datos para {comp_name}")
        return f

    df_rmse = pd.DataFrame(rmse_data).sort_values("rmse")

    bar_colors = []
    for _, row in df_rmse.iterrows():
        if row["is_best"]:
            bar_colors.append("#27AE60")
        elif row["is_combo"]:
            bar_colors.append("#9B59B6")
        else:
            bar_colors.append("#3498DB")

    fig = go.Figure(go.Bar(
        x=df_rmse["rmse"],
        y=df_rmse["model"],
        orientation="h",
        marker_color=bar_colors,
        text=df_rmse["rmse"].round(4).astype(str),
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(
            text=f"RMSE OOS por modelo — {comp_name} | {serie.strip()} h={h}",
            font=dict(size=13),
        ),
        xaxis_title="RMSE",
        yaxis_title="Modelo",
        height=max(420, 22 * len(df_rmse) + 80),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        margin=dict(l=160, r=80),
    )

    if best_model_name:
        fig.add_annotation(
            text=f"Mejor: {best_model_name} (usado en SOC)",
            xref="paper", yref="paper",
            x=0.98, y=0.02,
            xanchor="right", yanchor="bottom",
            showarrow=False,
            font=dict(color="#27AE60", size=12),
            bgcolor="white",
            bordercolor="#27AE60",
            borderwidth=1,
        )

    return fig


# ---------------------------------------------------------------------------
# Callbacks – Tab 4
# ---------------------------------------------------------------------------

@app.callback(
    Output("t4-heatmap", "figure"),
    Output("t4-summary", "children"),
    Input("main-tabs", "active_tab"),
)
def update_tab4(active_tab):
    if active_tab != "tab4":
        return go.Figure(), html.Div()

    df_metrics = load_metrics()
    if df_metrics is None:
        f = go.Figure()
        f.update_layout(title="Sin metricas — ejecuta fit_soc.py con mas series")
        return f, html.P("Sin datos.")

    pivot = df_metrics.pivot_table(index="serie", columns="h", values="ratio_SOC")
    pivot.index = [str(s).strip() for s in pivot.index]

    z = pivot.values.astype(float)
    z_text = np.where(np.isnan(z), "N/D", np.round(z, 3).astype(str))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"h={h}" for h in pivot.columns],
        y=pivot.index.tolist(),
        text=z_text,
        texttemplate="%{text}",
        colorscale=[
            [0.0,  "#1A9850"],
            [0.45, "#D9EF8B"],
            [0.5,  "#FFFFBF"],
            [0.55, "#FEE08B"],
            [1.0,  "#D73027"],
        ],
        zmid=1.0,
        zmin=0.3,
        zmax=1.7,
        colorbar=dict(title="RMSE SOC/AO", thickness=15),
        hovertemplate="Serie: %{y}<br>Horizonte: %{x}<br>Ratio: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Ratio RMSE SOC / AO   (< 1 = SOC supera al benchmark AO)",
        height=max(420, 32 * len(pivot) + 80),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        margin=dict(l=210, t=60),
    )

    summary_rows = []
    for _, row in df_metrics.iterrows():
        r_soc = row.get("rmse_SOC", np.nan)
        r_opt = row.get("rmse_SOC_opt", np.nan)
        r_ao  = row.get("rmse_AO", np.nan)
        ratio_soc = row.get("ratio_SOC", np.nan)
        ratio_opt = row.get("ratio_SOC_opt", np.nan)
        best_strat = "SOC_opt" if (np.isfinite(ratio_opt) and ratio_opt < ratio_soc) else "SOC"
        summary_rows.append({
            "Serie":          str(row["serie"]).strip(),
            "H":              int(row["h"]),
            "RMSE AO":        round(r_ao,  4) if np.isfinite(r_ao)  else None,
            "RMSE SOC":       round(r_soc, 4) if np.isfinite(r_soc) else None,
            "RMSE SOC_opt":   round(r_opt, 4) if np.isfinite(r_opt) else None,
            "Ratio SOC/AO":   round(ratio_soc, 3) if np.isfinite(ratio_soc) else None,
            "Ratio opt/AO":   round(ratio_opt, 3) if np.isfinite(ratio_opt) else None,
            "Mejor":          best_strat,
        })

    df_sum = pd.DataFrame(summary_rows).sort_values(["H", "Ratio SOC/AO"])
    table = dash_table.DataTable(
        data=df_sum.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_sum.columns],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "fontSize": "12px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#e9ecef"},
        style_data_conditional=[
            {"if": {"filter_query": "{Ratio SOC/AO} < 1", "column_id": "Ratio SOC/AO"},
             "color": "#1A9850", "fontWeight": "bold"},
            {"if": {"filter_query": "{Ratio SOC/AO} > 1", "column_id": "Ratio SOC/AO"},
             "color": "#D73027"},
        ],
        sort_action="native",
        page_size=25,
    )

    return fig, table


@app.callback(
    Output("t4-download", "data"),
    Input("t4-export-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_metrics(_):
    df_metrics = load_metrics()
    if df_metrics is None:
        return None
    return dcc.send_data_frame(df_metrics.to_excel, "soc_metrics.xlsx", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    meta = load_metadata()
    if meta:
        print(f"[dash_app] Resultados disponibles: {meta.get('n_ok', 0)} tareas OK")
    else:
        print("[dash_app] Sin resultados — ejecuta: python -m soc.fit_soc")
    print("[dash_app] Iniciando en http://localhost:8051")
    app.run(debug=False, port=8051, host="0.0.0.0")
