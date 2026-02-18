import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_loader import (fetch_from_banxico, load_data, fetch_from_banxico_q, load_data_q,
                         PONDERADORES, JERARQUIA, COLUMN_ORDER)

st.set_page_config(page_title="Dashboard INPC", layout="wide")

# --- Sidebar común ---
st.sidebar.header("Configuración")

frecuencia = st.sidebar.radio("Frecuencia", ["Mensual", "Quincenal"])
is_q = (frecuencia == "Quincenal")
PERIOD = 24 if is_q else 12
HORIZON = 24 if is_q else 12
FREQ = None if is_q else "MS"  # quincenal (1o y 16) no tiene freq estándar en pandas
N_LAGS_NN = 48 if is_q else 24
N_FEATURES_NN = N_LAGS_NN + 2
var_period_label = "Quincenal" if is_q else "Mensual"
var_short = "q" if is_q else "m"  # para columnas de tablas: Var.q / Var.m


def _next_q_dates(last_date, n):
    """Genera n fechas quincenales (1 y 16 de cada mes) después de last_date."""
    dates = []
    y, m, d = last_date.year, last_date.month, last_date.day
    for _ in range(n):
        if d < 16:
            d = 16
        else:
            d = 1
            m += 1
            if m > 12:
                m = 1
                y += 1
        dates.append(pd.Timestamp(y, m, d))
    return pd.DatetimeIndex(dates)


def _make_idx(index_str):
    """Crea DatetimeIndex, asignando freq sólo si es mensual."""
    idx = pd.DatetimeIndex(index_str)
    if FREQ is not None:
        idx.freq = FREQ
    return idx


def _set_freq(serie):
    """Asigna freq al índice sólo si es mensual."""
    if FREQ is not None:
        serie.index.freq = FREQ


def _fmt_date(dt):
    """Formatea fecha: '1qEne2026' / '2qEne2026' para quincenal, 'Ene 2026' para mensual."""
    if is_q:
        q = 1 if dt.day <= 1 else 2
        return f"{q}q{dt.strftime('%b%Y')}"
    return dt.strftime("%b %Y")


def _fmt_index(idx):
    """Formatea un DatetimeIndex completo para mostrar en tablas."""
    return [_fmt_date(d) for d in idx]

if st.sidebar.button("Actualizar datos desde Banxico"):
    with st.spinner("Descargando datos de Banxico..."):
        try:
            if is_q:
                df = fetch_from_banxico_q()
            else:
                df = fetch_from_banxico()
            st.sidebar.success(f"Datos actualizados. Último: {_fmt_date(df.index.max())}")
            st.cache_data.clear()
        except Exception as e:
            st.sidebar.error(f"Error al descargar: {e}")

# --- Navegación ---
pagina = st.sidebar.radio("Página", ["Serie Original", "Ajuste Estacional", "Pronósticos", "Redes Neuronales", "Manual"])

# Cargar datos
df = load_data_q() if is_q else load_data()

if df is None:
    st.info("No hay datos guardados. Haz clic en 'Actualizar datos desde Banxico' en el sidebar.")
    st.stop()

st.title("Análisis de la Inflación")

serie = st.sidebar.selectbox("Serie", df.columns.tolist())

min_date = df.index.min().to_pydatetime()
max_date = df.index.max().to_pydatetime()

rango = st.sidebar.slider(
    "Rango de fechas",
    min_value=min_date,
    max_value=max_date,
    value=(pd.Timestamp("2000-01-01").to_pydatetime(), max_date),
    format="YYYY-MM",
)

st.sidebar.caption(f"Último dato: {_fmt_date(max_date)}")


# ============================================================
# PAGINA 1: Serie Original
# ============================================================
if pagina == "Serie Original":
    st.header("INPC - Serie Original")

    mask = (df.index >= rango[0]) & (df.index <= rango[1])
    datos = df.loc[mask, serie].dropna()

    serie_completa = df[serie].dropna()
    var_mensual_full = serie_completa.pct_change() * 100
    var_anual_full = serie_completa.pct_change(PERIOD) * 100
    var_mensual = var_mensual_full[(var_mensual_full.index >= rango[0]) & (var_mensual_full.index <= rango[1])].dropna()
    var_anual = var_anual_full[(var_anual_full.index >= rango[0]) & (var_anual_full.index <= rango[1])].dropna()

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=datos.index, y=datos.values, mode="lines", name=serie))
        fig.update_layout(title=f"{serie} (Nivel)", yaxis_title="Índice", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=var_anual.index, y=var_anual.values, mode="lines", name="Var. Anual %", line=dict(color="crimson")))
        fig.update_layout(title=f"{serie} - Variación Anual (%)", yaxis_title="%", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=var_mensual.index, y=var_mensual.values, name=f"Var. {var_period_label} %", marker_color="steelblue"))
        fig.update_layout(title=f"{serie} - Variación {var_period_label} (%)", yaxis_title="%", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Resumen")
        ultimo = datos.iloc[-1]
        fecha_ultimo = _fmt_date(datos.index[-1])
        ponderador = PONDERADORES.get(serie, None)
        resumen = {
            "Último dato": f"{ultimo:.4f}",
            "Fecha": fecha_ultimo,
            "Ponderador (%)": f"{ponderador:.4f}" if ponderador is not None else "N/A",
            f"Var. {var_period_label} (%)": f"{var_mensual.iloc[-1]:.2f}" if len(var_mensual) > 0 else "N/A",
            "Var. Anual (%)": f"{var_anual.iloc[-1]:.2f}" if len(var_anual) > 0 else "N/A",
            "Máx histórico": f"{datos.max():.4f}",
            "Mín en rango": f"{datos.min():.4f}",
        }
        for k, v in resumen.items():
            st.metric(k, v)

    with st.expander("Ver tabla de datos"):
        rango_t1 = st.slider(
            "Rango de fechas (tabla)",
            min_value=min_date, max_value=max_date,
            value=(rango[0], rango[1]),
            format="YYYY-MM",
            key="tabla_original",
        )
        mask_t1 = (serie_completa.index >= rango_t1[0]) & (serie_completa.index <= rango_t1[1])
        datos_t1 = serie_completa[mask_t1]
        vm_t1 = var_mensual_full[mask_t1].dropna()
        va_t1 = var_anual_full[mask_t1].dropna()
        tabla = pd.DataFrame({"Nivel": datos_t1, f"Var. {var_period_label} (%)": vm_t1, "Var. Anual (%)": va_t1})
        tabla = tabla.sort_index(ascending=False)
        st.dataframe(tabla, use_container_width=True)
        st.download_button(
            "Descargar CSV (Excel)", tabla.to_csv(),
            f"{serie.strip()}_original.csv", "text/csv", key="dl_original",
        )


# ============================================================
# PAGINA 2: Ajuste Estacional (STL)
# ============================================================
elif pagina == "Ajuste Estacional":
    from statsmodels.tsa.seasonal import STL

    st.header("INPC - Ajuste Estacional (STL)")
    st.caption(f"Descomposición Seasonal-Trend usando LOESS (STL) con periodo={PERIOD}")

    # STL necesita frecuencia definida y sin NaN
    serie_completa = df[serie].dropna()
    _set_freq(serie_completa)

    # Ajuste estacional
    @st.cache_data
    def run_stl(values, index_str, period, quincenal):
        idx = pd.DatetimeIndex(index_str)
        if not quincenal:
            idx.freq = "MS"
        s = pd.Series(values, index=idx)
        stl = STL(s, period=period, robust=True)
        result = stl.fit()
        return result.seasonal.values, result.trend.values

    seasonal_vals, trend_vals = run_stl(serie_completa.values, serie_completa.index.astype(str).tolist(), PERIOD, is_q)
    seasonal = pd.Series(seasonal_vals, index=serie_completa.index, name="Estacional")
    trend = pd.Series(trend_vals, index=serie_completa.index, name="Tendencia")
    seasadj = serie_completa - seasonal
    seasadj.name = "Desestacionalizada"

    # Variaciones sobre la serie desestacionalizada
    var_mensual_sa = seasadj.pct_change() * 100
    var_anual_sa = seasadj.pct_change(PERIOD) * 100

    # Filtrar por rango
    mask_range = (serie_completa.index >= rango[0]) & (serie_completa.index <= rango[1])
    datos_orig = serie_completa[mask_range]
    datos_sa = seasadj[mask_range]
    datos_trend = trend[mask_range]
    datos_seasonal = seasonal[mask_range]
    vm_sa = var_mensual_sa[mask_range].dropna()
    va_sa = var_anual_sa[mask_range].dropna()

    # --- Gráficas ---
    # 1. Original vs Desestacionalizada vs Tendencia
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=datos_orig.index, y=datos_orig.values, mode="lines", name="Original", opacity=0.5))
        fig.add_trace(go.Scatter(x=datos_sa.index, y=datos_sa.values, mode="lines", name="Desestacionalizada", line=dict(color="darkorange")))
        fig.add_trace(go.Scatter(x=datos_trend.index, y=datos_trend.values, mode="lines", name="Tendencia", line=dict(color="green", dash="dash")))
        fig.update_layout(title=f"{serie} - Original vs Desestacionalizada", yaxis_title="Índice", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=va_sa.index, y=va_sa.values, mode="lines", name="Var. Anual % (SA)", line=dict(color="crimson")))
        fig.update_layout(title=f"{serie} SA - Variación Anual (%)", yaxis_title="%", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=vm_sa.index, y=vm_sa.values, name=f"Var. {var_period_label} % (SA)", marker_color="darkorange"))
        fig.update_layout(title=f"{serie} SA - Variación {var_period_label} (%)", yaxis_title="%", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Componente estacional
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=datos_seasonal.index, y=datos_seasonal.values, mode="lines", name="Comp. Estacional", line=dict(color="purple")))
        fig.update_layout(title=f"{serie} - Componente Estacional", yaxis_title="Índice", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Resumen: Original y Desestacionalizada
    var_mensual_orig = serie_completa.pct_change() * 100
    var_anual_orig = serie_completa.pct_change(PERIOD) * 100
    vm_orig = var_mensual_orig[mask_range].dropna()
    va_orig = var_anual_orig[mask_range].dropna()

    st.subheader("Resumen")
    col_orig, col_sa = st.columns(2)

    with col_orig:
        st.markdown("**Serie Original**")
        st.metric("Último dato", f"{datos_orig.iloc[-1]:.4f}")
        st.metric("Fecha", _fmt_date(datos_orig.index[-1]))
        st.metric(f"Var. {var_period_label} (%)", f"{vm_orig.iloc[-1]:.2f}" if len(vm_orig) > 0 else "N/A")
        st.metric("Var. Anual (%)", f"{va_orig.iloc[-1]:.2f}" if len(va_orig) > 0 else "N/A")

    with col_sa:
        st.markdown("**Serie Desestacionalizada**")
        st.metric("Último dato SA", f"{datos_sa.iloc[-1]:.4f}")
        st.metric("Fecha", _fmt_date(datos_sa.index[-1]))
        st.metric(f"Var. {var_period_label} SA (%)", f"{vm_sa.iloc[-1]:.2f}" if len(vm_sa) > 0 else "N/A")
        st.metric("Var. Anual SA (%)", f"{va_sa.iloc[-1]:.2f}" if len(va_sa) > 0 else "N/A")

    with st.expander("Ver tabla de datos"):
        rango_t2 = st.slider(
            "Rango de fechas (tabla)",
            min_value=min_date, max_value=max_date,
            value=(rango[0], rango[1]),
            format="YYYY-MM",
            key="tabla_estacional",
        )
        mask_t2 = (serie_completa.index >= rango_t2[0]) & (serie_completa.index <= rango_t2[1])
        tabla = pd.DataFrame({
            "Original": serie_completa[mask_t2],
            "Desestacionalizada": seasadj[mask_t2],
            "Tendencia": trend[mask_t2],
            "Comp. Estacional": seasonal[mask_t2],
            f"Var. {var_period_label} SA (%)": var_mensual_sa[mask_t2].dropna(),
            "Var. Anual SA (%)": var_anual_sa[mask_t2].dropna(),
        })
        tabla = tabla.sort_index(ascending=False)
        st.dataframe(tabla, use_container_width=True)
        st.download_button(
            "Descargar CSV (Excel)", tabla.to_csv(),
            f"{serie.strip()}_estacional.csv", "text/csv", key="dl_estacional",
        )


# ============================================================
# PÁGINA 3: Pronósticos ARIMA / SARIMA
# ============================================================
elif pagina == "Pronósticos":
    import json
    import numpy as np
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    st.header("INPC - Pronósticos ARIMA / SARIMA")
    st.caption(f"Especificaciones pre-calibradas (últimos {PERIOD * 4} períodos). "
               "Se recomienda recalibrar anualmente. Comando: `python fit_orders.py" +
               (" --quincenal`" if is_q else "`"))

    ORDERS_PATH = os.path.join(os.path.dirname(__file__), "data",
                                "model_orders_q.json" if is_q else "model_orders.json")
    if not os.path.exists(ORDERS_PATH):
        cmd = "python fit_orders.py --quincenal" if is_q else "python fit_orders.py"
        archivo = "model_orders_q.json" if is_q else "model_orders.json"
        st.error(f"No se encontro data/{archivo}. Ejecuta `{cmd}` primero.")
        st.stop()

    with open(ORDERS_PATH, encoding="utf-8") as f:
        all_orders = json.load(f)

    if serie not in all_orders:
        st.warning(f"No hay órdenes pre-calculados para '{serie}'. Ejecuta fit_orders.py.")
        st.stop()

    serie_completa = df[serie].dropna()
    _set_freq(serie_completa)

    @st.cache_data
    def fit_forecast_single(values, index_str, arima_ord, arima_tr, sarima_ord, sarima_seas, sarima_tr, horizon, quincenal):
        from concurrent.futures import ThreadPoolExecutor
        idx = pd.DatetimeIndex(index_str)
        if not quincenal:
            idx.freq = "MS"
        s = pd.Series(values, index=idx)

        def _fit_arima():
            model = SARIMAX(s, order=arima_ord, trend=arima_tr,
                            enforce_stationarity=False, enforce_invertibility=False)
            return model.fit(disp=False)

        def _fit_sarima():
            model = SARIMAX(s, order=sarima_ord, seasonal_order=sarima_seas,
                            trend=sarima_tr, enforce_stationarity=False,
                            enforce_invertibility=False)
            return model.fit(disp=False)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_arima = pool.submit(_fit_arima)
            fut_sarima = pool.submit(_fit_sarima)
            arima_res = fut_arima.result()
            sarima_res = fut_sarima.result()

        arima_pred = arima_res.get_forecast(steps=horizon)
        sarima_pred = sarima_res.get_forecast(steps=horizon)

        arima_fc = arima_pred.predicted_mean.values
        arima_ci = arima_pred.conf_int().values
        sarima_fc = sarima_pred.predicted_mean.values
        sarima_ci = sarima_pred.conf_int().values

        # Generar índice de fechas del pronóstico
        last_date = s.index[-1]
        if quincenal:
            fc_idx = _next_q_dates(last_date, horizon)
        else:
            fc_idx = arima_pred.predicted_mean.index

        arima_resid = arima_res.resid.values
        sarima_resid = sarima_res.resid.values
        arima_summary = arima_res.summary().as_text()
        sarima_summary = sarima_res.summary().as_text()

        return (arima_fc, arima_ci, sarima_fc, sarima_ci,
                fc_idx.astype(str).tolist(),
                arima_resid, sarima_resid, arima_summary, sarima_summary)

    def _run_forecast(serie_name):
        """Pronostica una serie usando sus órdenes pre-calculados."""
        ord_ = all_orders[serie_name]
        s = df[serie_name].dropna()
        _set_freq(s)
        return fit_forecast_single(
            s.values, s.index.astype(str).tolist(),
            tuple(ord_["arima_order"]),
            "c" if ord_.get("arima_intercept", False) else "n",
            tuple(ord_["sarima_order"]),
            tuple(ord_["sarima_seasonal_order"]),
            "c" if ord_.get("sarima_intercept", False) else "n",
            HORIZON, is_q,
        )

    orders = all_orders[serie]
    arima_order = tuple(orders["arima_order"])
    sarima_order = tuple(orders["sarima_order"])
    sarima_seasonal = tuple(orders["sarima_seasonal_order"])

    with st.spinner("Estimando modelos..."):
        (arima_fc, arima_ci, sarima_fc, sarima_ci,
         fc_index_str,
         arima_resid, sarima_resid, arima_summary_txt, sarima_summary_txt) = _run_forecast(serie)

    fc_index = _make_idx(fc_index_str)
    arima_forecast = pd.Series(arima_fc, index=fc_index)
    sarima_forecast = pd.Series(sarima_fc, index=fc_index)

    # --- Suma ponderada desde subcomponentes ---
    hijos = JERARQUIA.get(serie, None)
    pond_fc_arima = None
    pond_fc_sarima = None

    if hijos and all(h in all_orders for h in hijos):
        with st.spinner("Calculando suma ponderada de subcomponentes..."):
            peso_total = sum(PONDERADORES[h] for h in hijos)
            pond_arima = np.zeros(HORIZON)
            pond_sarima = np.zeros(HORIZON)
            children_info = []
            for h in hijos:
                w = PONDERADORES[h] / peso_total
                res_h = _run_forecast(h)
                pond_arima += w * res_h[0]   # arima_fc
                pond_sarima += w * res_h[2]  # sarima_fc
                ord_h = all_orders[h]
                children_info.append(
                    f"{h.strip()} (w={PONDERADORES[h]:.2f}%): "
                    f"ARIMA{tuple(ord_h['arima_order'])} / "
                    f"SARIMA{tuple(ord_h['sarima_order'])}x{tuple(ord_h['sarima_seasonal_order'])}"
                )
            pond_fc_arima = pd.Series(pond_arima, index=fc_index)
            pond_fc_sarima = pd.Series(pond_sarima, index=fc_index)

    # Info de órdenes
    info_txt = (f"**ARIMA** orden: {arima_order}  |  "
                f"**SARIMA** orden: {sarima_order} x {sarima_seasonal}")
    if pond_fc_arima is not None:
        info_txt += "\n\n**Suma ponderada** construida desde:\n" + "\n".join(
            f"- {ci}" for ci in children_info)
    st.info(info_txt)

    # --- Variaciones historicas ---
    var_period_hist = serie_completa.pct_change() * 100          # quincenal o mensual
    var_anual_hist = serie_completa.pct_change(PERIOD) * 100
    if is_q:
        var_mensual_hist_q = serie_completa.pct_change(2) * 100  # mensual dentro de quincenal

    # --- Variaciones del pronóstico ---
    def calc_variations(forecast, serie_hist, fc_idx):
        combined = pd.concat([serie_hist.iloc[-1:], forecast])
        vp = combined.pct_change().iloc[1:] * 100   # var. periodo (q o m)
        h_period = serie_hist.iloc[-PERIOD:]
        va = pd.Series((forecast.values / h_period.values - 1) * 100, index=fc_idx)
        # var. mensual dentro de quincenal (pct_change(2))
        vm = None
        if is_q:
            combined2 = pd.concat([serie_hist.iloc[-2:], forecast])
            vm = combined2.pct_change(2).iloc[2:] * 100
        return vp, va, vm

    vp_arima, va_arima, vm_arima = calc_variations(arima_forecast, serie_completa, fc_index)
    vp_sarima, va_sarima, vm_sarima = calc_variations(sarima_forecast, serie_completa, fc_index)

    if pond_fc_arima is not None:
        vp_pond_arima, va_pond_arima, vm_pond_arima = calc_variations(pond_fc_arima, serie_completa, fc_index)
        vp_pond_sarima, va_pond_sarima, vm_pond_sarima = calc_variations(pond_fc_sarima, serie_completa, fc_index)

    # --- Filtrar históricos por rango (gráficas) ---
    rango_chart_arima = st.slider(
        "Rango de historia en graficas",
        min_value=min_date, max_value=max_date,
        value=(pd.Timestamp("2015-01-01").to_pydatetime(), max_date),
        format="YYYY-MM", key="rango_chart_arima",
    )
    mask_range = (serie_completa.index >= rango_chart_arima[0]) & (serie_completa.index <= rango_chart_arima[1])
    datos_hist = serie_completa[mask_range]
    va_hist = var_anual_hist[mask_range].dropna()
    vp_hist = var_period_hist[mask_range].dropna()

    # --- Gráfica 1: Nivel histórico + pronósticos ---
    st.subheader("Nivel")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datos_hist.index, y=datos_hist.values,
                             mode="lines", name="Histórico"))
    # ARIMA forecast + CI
    fig.add_trace(go.Scatter(x=fc_index, y=arima_ci[:, 1], mode="lines",
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fc_index, y=arima_ci[:, 0], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(31,119,180,0.15)", showlegend=False))
    fig.add_trace(go.Scatter(x=fc_index, y=arima_forecast.values,
                             mode="lines+markers", name="ARIMA",
                             line=dict(color="rgb(31,119,180)", dash="dash")))
    # SARIMA forecast + CI
    fig.add_trace(go.Scatter(x=fc_index, y=sarima_ci[:, 1], mode="lines",
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fc_index, y=sarima_ci[:, 0], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(255,127,14,0.15)", showlegend=False))
    fig.add_trace(go.Scatter(x=fc_index, y=sarima_forecast.values,
                             mode="lines+markers", name="SARIMA",
                             line=dict(color="rgb(255,127,14)", dash="dash")))
    if pond_fc_arima is not None:
        fig.add_trace(go.Scatter(x=fc_index, y=pond_fc_arima.values,
                                 mode="lines+markers", name="Pond. ARIMA",
                                 line=dict(color="rgb(44,160,44)", dash="dot")))
        fig.add_trace(go.Scatter(x=fc_index, y=pond_fc_sarima.values,
                                 mode="lines+markers", name="Pond. SARIMA",
                                 line=dict(color="rgb(214,39,40)", dash="dot")))
    fig.update_layout(title=f"{serie} - Nivel + Pronóstico {HORIZON} períodos",
                      yaxis_title="Índice", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # --- Gráficas de variaciones ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Variación Anual (%)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=va_hist.index, y=va_hist.values,
                                 mode="lines", name="Histórico", line=dict(color="crimson")))
        fig.add_trace(go.Scatter(x=fc_index, y=va_arima.values,
                                 mode="lines+markers", name="ARIMA",
                                 line=dict(color="rgb(31,119,180)", dash="dash")))
        fig.add_trace(go.Scatter(x=fc_index, y=va_sarima.values,
                                 mode="lines+markers", name="SARIMA",
                                 line=dict(color="rgb(255,127,14)", dash="dash")))
        if pond_fc_arima is not None:
            fig.add_trace(go.Scatter(x=fc_index, y=va_pond_arima.values,
                                     mode="lines+markers", name="Pond. ARIMA",
                                     line=dict(color="rgb(44,160,44)", dash="dot")))
            fig.add_trace(go.Scatter(x=fc_index, y=va_pond_sarima.values,
                                     mode="lines+markers", name="Pond. SARIMA",
                                     line=dict(color="rgb(214,39,40)", dash="dot")))
        fig.update_layout(yaxis_title="%", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Variación {var_period_label} (%)")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=vp_hist.index, y=vp_hist.values,
                             name="Histórico", marker_color="steelblue"))
        fig.add_trace(go.Scatter(x=fc_index, y=vp_arima.values,
                                 mode="lines+markers", name="ARIMA",
                                 line=dict(color="rgb(31,119,180)", dash="dash")))
        fig.add_trace(go.Scatter(x=fc_index, y=vp_sarima.values,
                                 mode="lines+markers", name="SARIMA",
                                 line=dict(color="rgb(255,127,14)", dash="dash")))
        if pond_fc_arima is not None:
            fig.add_trace(go.Scatter(x=fc_index, y=vp_pond_arima.values,
                                     mode="lines+markers", name="Pond. ARIMA",
                                     line=dict(color="rgb(44,160,44)", dash="dot")))
            fig.add_trace(go.Scatter(x=fc_index, y=vp_pond_sarima.values,
                                     mode="lines+markers", name="Pond. SARIMA",
                                     line=dict(color="rgb(214,39,40)", dash="dot")))
        fig.update_layout(yaxis_title="%", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # --- Tabla de pronosticos ---
    st.subheader("Tabla de Pronósticos")
    n_hist_arima = st.slider("Periodos de historia a incluir", 0, 120, PERIOD,
                             key="hist_arima")

    all_cols_arima = ["Nivel ARIMA", "Nivel SARIMA"]
    if pond_fc_arima is not None:
        all_cols_arima += ["Nivel Pond. ARIMA", "Nivel Pond. SARIMA"]
    all_cols_arima += [f"Var.{var_short} ARIMA (%)", f"Var.{var_short} SARIMA (%)"]
    if pond_fc_arima is not None:
        all_cols_arima += [f"Var.{var_short} Pond.A (%)", f"Var.{var_short} Pond.S (%)"]
    if is_q:
        all_cols_arima += ["Var.m ARIMA (%)", "Var.m SARIMA (%)"]
        if pond_fc_arima is not None:
            all_cols_arima += ["Var.m Pond.A (%)", "Var.m Pond.S (%)"]
    all_cols_arima += ["Var.a ARIMA (%)", "Var.a SARIMA (%)"]
    if pond_fc_arima is not None:
        all_cols_arima += ["Var.a Pond.A (%)", "Var.a Pond.S (%)"]

    cols_visibles_arima = st.multiselect(
        "Columnas visibles", all_cols_arima, default=all_cols_arima, key="cols_arima")

    # Construir parte historica
    if n_hist_arima > 0:
        hist_slice = serie_completa.iloc[-n_hist_arima:]
        hist_rows = pd.DataFrame({"Nivel ARIMA": hist_slice.values,
                                  "Nivel SARIMA": hist_slice.values}, index=hist_slice.index)
        if pond_fc_arima is not None:
            hist_rows["Nivel Pond. ARIMA"] = hist_slice.values
            hist_rows["Nivel Pond. SARIMA"] = hist_slice.values
        hist_rows[f"Var.{var_short} ARIMA (%)"] = var_period_hist.reindex(hist_slice.index)
        hist_rows[f"Var.{var_short} SARIMA (%)"] = var_period_hist.reindex(hist_slice.index)
        if pond_fc_arima is not None:
            hist_rows[f"Var.{var_short} Pond.A (%)"] = var_period_hist.reindex(hist_slice.index)
            hist_rows[f"Var.{var_short} Pond.S (%)"] = var_period_hist.reindex(hist_slice.index)
        if is_q:
            hist_rows["Var.m ARIMA (%)"] = var_mensual_hist_q.reindex(hist_slice.index)
            hist_rows["Var.m SARIMA (%)"] = var_mensual_hist_q.reindex(hist_slice.index)
            if pond_fc_arima is not None:
                hist_rows["Var.m Pond.A (%)"] = var_mensual_hist_q.reindex(hist_slice.index)
                hist_rows["Var.m Pond.S (%)"] = var_mensual_hist_q.reindex(hist_slice.index)
        hist_rows["Var.a ARIMA (%)"] = var_anual_hist.reindex(hist_slice.index)
        hist_rows["Var.a SARIMA (%)"] = var_anual_hist.reindex(hist_slice.index)
        if pond_fc_arima is not None:
            hist_rows["Var.a Pond.A (%)"] = var_anual_hist.reindex(hist_slice.index)
            hist_rows["Var.a Pond.S (%)"] = var_anual_hist.reindex(hist_slice.index)

    # Construir parte pronóstico
    tabla_data = {
        "Nivel ARIMA": arima_forecast.values,
        "Nivel SARIMA": sarima_forecast.values,
    }
    if pond_fc_arima is not None:
        tabla_data["Nivel Pond. ARIMA"] = pond_fc_arima.values
        tabla_data["Nivel Pond. SARIMA"] = pond_fc_sarima.values
    tabla_data.update({
        f"Var.{var_short} ARIMA (%)": vp_arima.values,
        f"Var.{var_short} SARIMA (%)": vp_sarima.values,
    })
    if pond_fc_arima is not None:
        tabla_data[f"Var.{var_short} Pond.A (%)"] = vp_pond_arima.values
        tabla_data[f"Var.{var_short} Pond.S (%)"] = vp_pond_sarima.values
    if is_q:
        tabla_data["Var.m ARIMA (%)"] = vm_arima.values
        tabla_data["Var.m SARIMA (%)"] = vm_sarima.values
        if pond_fc_arima is not None:
            tabla_data["Var.m Pond.A (%)"] = vm_pond_arima.values
            tabla_data["Var.m Pond.S (%)"] = vm_pond_sarima.values
    tabla_data.update({
        "Var.a ARIMA (%)": va_arima.values,
        "Var.a SARIMA (%)": va_sarima.values,
    })
    if pond_fc_arima is not None:
        tabla_data["Var.a Pond.A (%)"] = va_pond_arima.values
        tabla_data["Var.a Pond.S (%)"] = va_pond_sarima.values
    tabla_fc = pd.DataFrame(tabla_data, index=fc_index)

    if n_hist_arima > 0:
        tabla_fc = pd.concat([hist_rows, tabla_fc])

    tabla_fc.index.name = "Fecha"
    tabla_fc_display = tabla_fc[[c for c in cols_visibles_arima if c in tabla_fc.columns]].copy()
    tabla_fc_display.index = _fmt_index(tabla_fc_display.index)
    st.dataframe(tabla_fc_display.style.format("{:.4f}"), use_container_width=True)
    st.download_button(
        "Descargar CSV (Excel)", tabla_fc.to_csv(),
        f"{serie.strip()}_pronostico_arima.csv", "text/csv", key="dl_arima",
    )

    # --- Gráficas de residuales (errores estandarizados) ---
    st.subheader("Residuales de los modelos")

    arima_std_resid = (arima_resid - np.mean(arima_resid)) / np.std(arima_resid)
    sarima_std_resid = (sarima_resid - np.mean(sarima_resid)) / np.std(sarima_resid)
    resid_index = serie_completa.index[:len(arima_resid)]

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=resid_index, y=arima_std_resid,
                                 mode="lines", name="Residuales",
                                 line=dict(color="rgb(31,119,180)")))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=2, line_dash="dot", line_color="red", opacity=0.5)
        fig.add_hline(y=-2, line_dash="dot", line_color="red", opacity=0.5)
        fig.update_layout(title=f"ARIMA {arima_order} - Residuales estandarizados",
                          yaxis_title="Desv. estándar", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        sarima_resid_index = serie_completa.index[:len(sarima_resid)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sarima_resid_index, y=sarima_std_resid,
                                 mode="lines", name="Residuales",
                                 line=dict(color="rgb(255,127,14)")))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=2, line_dash="dot", line_color="red", opacity=0.5)
        fig.add_hline(y=-2, line_dash="dot", line_color="red", opacity=0.5)
        fig.update_layout(title=f"SARIMA {sarima_order}x{sarima_seasonal} - Residuales estandarizados",
                          yaxis_title="Desv. estándar", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Histogramas de residuales
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=arima_std_resid, nbinsx=40,
                                   name="ARIMA", marker_color="rgb(31,119,180)", opacity=0.75))
        fig.update_layout(title=f"ARIMA {arima_order} - Distribucion de residuales",
                          xaxis_title="Desv. estándar", yaxis_title="Frecuencia",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col_h2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=sarima_std_resid, nbinsx=40,
                                   name="SARIMA", marker_color="rgb(255,127,14)", opacity=0.75))
        fig.update_layout(title=f"SARIMA {sarima_order}x{sarima_seasonal} - Distribucion de residuales",
                          xaxis_title="Desv. estándar", yaxis_title="Frecuencia",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # --- Especificacion de modelos ---
    st.subheader("Especificacion de los modelos")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown(f"**ARIMA {arima_order}**")
        st.code(arima_summary_txt, language=None)

    with col_s2:
        st.markdown(f"**SARIMA {sarima_order} x {sarima_seasonal}**")
        st.code(sarima_summary_txt, language=None)


# ============================================================
# PAGINA 4: Redes Neuronales (MLP + LSTM)
# ============================================================
elif pagina == "Redes Neuronales":
    import json
    import numpy as np
    import joblib
    import torch
    import torch.nn as tnn

    st.header("INPC - Pronósticos con Redes Neuronales")
    st.caption("Modelos pre-entrenados (MLP + LSTM). "
               "Se recomienda recalibrar anualmente. Comando: `python fit_nn.py" +
               (" --quincenal`" if is_q else "`"))

    NN_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data",
                                   "nn_config_q.json" if is_q else "nn_config.json")
    NN_MODELS_DIR = os.path.join(os.path.dirname(__file__), "data",
                                  "nn_models_q" if is_q else "nn_models")

    if not os.path.exists(NN_CONFIG_PATH):
        cmd = "python fit_nn.py --quincenal" if is_q else "python fit_nn.py"
        archivo = "nn_config_q.json" if is_q else "nn_config.json"
        st.error(f"No se encontro data/{archivo}. Ejecuta `{cmd}` primero.")
        st.stop()

    with open(NN_CONFIG_PATH, encoding="utf-8") as f:
        nn_config = json.load(f)

    if serie not in nn_config:
        st.warning(f"No hay modelos pre-entrenados para '{serie}'. Ejecuta fit_nn.py.")
        st.stop()

    serie_completa = df[serie].dropna()
    _set_freq(serie_completa)

    N_LAGS = N_LAGS_NN

    # --- LSTM model class (must match fit_nn.py) ---
    class LSTMModel(tnn.Module):
        def __init__(self, input_size=N_FEATURES_NN, hidden_size=32, num_layers=1):
            super().__init__()
            self.lstm = tnn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                 num_layers=num_layers, batch_first=True)
            self.fc = tnn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    # --- Load models ---
    @st.cache_resource
    def load_nn_models(safe_name, n_features, hidden_size, num_layers):
        series_dir = os.path.join(NN_MODELS_DIR, safe_name)
        mlp_model = joblib.load(os.path.join(series_dir, "mlp_model.pkl"))
        mlp_scaler_X = joblib.load(os.path.join(series_dir, "mlp_scaler_X.pkl"))
        mlp_scaler_y = joblib.load(os.path.join(series_dir, "mlp_scaler_y.pkl"))
        lstm_model = LSTMModel(input_size=n_features, hidden_size=hidden_size,
                                num_layers=num_layers)
        lstm_model.load_state_dict(
            torch.load(os.path.join(series_dir, "lstm_model.pt"), weights_only=True))
        lstm_model.eval()
        lstm_scaler_X = joblib.load(os.path.join(series_dir, "lstm_scaler_X.pkl"))
        lstm_scaler_y = joblib.load(os.path.join(series_dir, "lstm_scaler_y.pkl"))
        return (mlp_model, mlp_scaler_X, mlp_scaler_y,
                lstm_model, lstm_scaler_X, lstm_scaler_y)

    # --- Recursive forecast ---
    def nn_recursive_forecast(model_type, model, scaler_X, scaler_y,
                              last_values, last_month, n_steps=HORIZON):
        """Pronóstico recursivo de n_steps usando ventana de N_LAGS rezagos."""
        preds = []
        window = list(last_values[-N_LAGS:])
        for i in range(n_steps):
            month = (last_month + i) % 12 + 1
            lags = list(reversed(window[-N_LAGS:]))
            sin_m = np.sin(2 * np.pi * month / PERIOD)
            cos_m = np.cos(2 * np.pi * month / PERIOD)
            features = np.array(lags + [sin_m, cos_m]).reshape(1, -1)
            features_scaled = scaler_X.transform(features)

            if model_type == "mlp":
                pred_scaled = model.predict(features_scaled)
                pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
            else:  # lstm
                x_tensor = torch.FloatTensor(features_scaled).unsqueeze(1)
                with torch.no_grad():
                    pred_scaled = model(x_tensor).item()
                pred = scaler_y.inverse_transform([[pred_scaled]])[0, 0]

            preds.append(pred)
            window.append(pred)
        return np.array(preds)

    # --- Run forecast for a given series ---
    @st.cache_data
    def run_nn_forecast(serie_name, values, index_str, quincenal):
        cfg = nn_config[serie_name]
        models = load_nn_models(cfg["safe_name"], cfg["n_features"],
                                cfg["lstm_hidden_size"], cfg["lstm_num_layers"])
        mlp_model, mlp_sX, mlp_sY, lstm_model, lstm_sX, lstm_sY = models

        idx = pd.DatetimeIndex(index_str)
        if not quincenal:
            idx.freq = "MS"
        s = pd.Series(values, index=idx)
        last_month = s.index[-1].month

        mlp_fc = nn_recursive_forecast("mlp", mlp_model, mlp_sX, mlp_sY,
                                       s.values, last_month)
        lstm_fc = nn_recursive_forecast("lstm", lstm_model, lstm_sX, lstm_sY,
                                        s.values, last_month)

        if quincenal:
            fc_index = _next_q_dates(s.index[-1], HORIZON)
        else:
            fc_index = pd.date_range(s.index[-1] + pd.DateOffset(months=1),
                                     periods=HORIZON, freq="MS")
        return mlp_fc, lstm_fc, fc_index.astype(str).tolist()

    def _run_nn(serie_name):
        s = df[serie_name].dropna()
        _set_freq(s)
        return run_nn_forecast(serie_name, s.values, s.index.astype(str).tolist(), is_q)

    # --- Run main forecast ---
    with st.spinner("Generando pronosticos con redes neuronales..."):
        mlp_fc, lstm_fc, fc_index_str = _run_nn(serie)

    fc_index = _make_idx(fc_index_str)
    mlp_forecast = pd.Series(mlp_fc, index=fc_index)
    lstm_forecast = pd.Series(lstm_fc, index=fc_index)

    # --- Suma ponderada desde subcomponentes ---
    hijos = JERARQUIA.get(serie, None)
    pond_fc_mlp = None
    pond_fc_lstm = None

    if hijos and all(h in nn_config for h in hijos):
        with st.spinner("Calculando suma ponderada de subcomponentes..."):
            peso_total = sum(PONDERADORES[h] for h in hijos)
            pond_mlp = np.zeros(HORIZON)
            pond_lstm = np.zeros(HORIZON)
            children_info = []
            for h in hijos:
                w = PONDERADORES[h] / peso_total
                res_h = _run_nn(h)
                pond_mlp += w * res_h[0]
                pond_lstm += w * res_h[1]
                cfg_h = nn_config[h]
                children_info.append(
                    f"{h.strip()} (w={PONDERADORES[h]:.2f}%): "
                    f"MLP{cfg_h['mlp_params']['hidden_layer_sizes']} / LSTM(32)"
                )
            pond_fc_mlp = pd.Series(pond_mlp, index=fc_index)
            pond_fc_lstm = pd.Series(pond_lstm, index=fc_index)

    # Info de modelos
    cfg = nn_config[serie]
    info_txt = (f"**MLP** capas: {cfg['mlp_params']['hidden_layer_sizes']}, "
                f"alpha: {cfg['mlp_params']['alpha']}, CV MSE: {cfg['mlp_cv_mse']:.6f}  |  "
                f"**LSTM** hidden: {cfg['lstm_hidden_size']}, "
                f"val MSE: {cfg['lstm_val_mse']:.6f}")
    if pond_fc_mlp is not None:
        info_txt += "\n\n**Suma ponderada** construida desde:\n" + "\n".join(
            f"- {ci}" for ci in children_info)
    st.info(info_txt)

    # --- Variaciones historicas ---
    var_period_hist = serie_completa.pct_change() * 100
    var_anual_hist = serie_completa.pct_change(PERIOD) * 100
    if is_q:
        var_mensual_hist_q = serie_completa.pct_change(2) * 100

    # --- Variaciones del pronóstico ---
    def calc_variations(forecast, serie_hist, fc_idx):
        combined = pd.concat([serie_hist.iloc[-1:], forecast])
        vp = combined.pct_change().iloc[1:] * 100
        h_period = serie_hist.iloc[-PERIOD:]
        va = pd.Series((forecast.values / h_period.values - 1) * 100, index=fc_idx)
        vm = None
        if is_q:
            combined2 = pd.concat([serie_hist.iloc[-2:], forecast])
            vm = combined2.pct_change(2).iloc[2:] * 100
        return vp, va, vm

    vp_mlp, va_mlp, vm_mlp = calc_variations(mlp_forecast, serie_completa, fc_index)
    vp_lstm, va_lstm, vm_lstm = calc_variations(lstm_forecast, serie_completa, fc_index)

    if pond_fc_mlp is not None:
        vp_pond_mlp, va_pond_mlp, vm_pond_mlp = calc_variations(pond_fc_mlp, serie_completa, fc_index)
        vp_pond_lstm, va_pond_lstm, vm_pond_lstm = calc_variations(pond_fc_lstm, serie_completa, fc_index)

    # --- Filtrar históricos por rango (gráficas) ---
    rango_chart_nn = st.slider(
        "Rango de historia en graficas",
        min_value=min_date, max_value=max_date,
        value=(pd.Timestamp("2015-01-01").to_pydatetime(), max_date),
        format="YYYY-MM", key="rango_chart_nn",
    )
    mask_range = (serie_completa.index >= rango_chart_nn[0]) & (serie_completa.index <= rango_chart_nn[1])
    datos_hist = serie_completa[mask_range]
    va_hist = var_anual_hist[mask_range].dropna()
    vp_hist = var_period_hist[mask_range].dropna()

    # --- Gráfica 1: Nivel histórico + pronósticos ---
    st.subheader("Nivel")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datos_hist.index, y=datos_hist.values,
                             mode="lines", name="Histórico"))
    fig.add_trace(go.Scatter(x=fc_index, y=mlp_forecast.values,
                             mode="lines+markers", name="MLP",
                             line=dict(color="rgb(148,103,189)", dash="dash")))
    fig.add_trace(go.Scatter(x=fc_index, y=lstm_forecast.values,
                             mode="lines+markers", name="LSTM",
                             line=dict(color="rgb(227,119,194)", dash="dash")))
    if pond_fc_mlp is not None:
        fig.add_trace(go.Scatter(x=fc_index, y=pond_fc_mlp.values,
                                 mode="lines+markers", name="Pond. MLP",
                                 line=dict(color="rgb(44,160,44)", dash="dot")))
        fig.add_trace(go.Scatter(x=fc_index, y=pond_fc_lstm.values,
                                 mode="lines+markers", name="Pond. LSTM",
                                 line=dict(color="rgb(214,39,40)", dash="dot")))
    fig.update_layout(title=f"{serie} - Nivel + Pronóstico {HORIZON} períodos (Redes Neuronales)",
                      yaxis_title="Índice", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # --- Gráficas de variaciones ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Variación Anual (%)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=va_hist.index, y=va_hist.values,
                                 mode="lines", name="Histórico", line=dict(color="crimson")))
        fig.add_trace(go.Scatter(x=fc_index, y=va_mlp.values,
                                 mode="lines+markers", name="MLP",
                                 line=dict(color="rgb(148,103,189)", dash="dash")))
        fig.add_trace(go.Scatter(x=fc_index, y=va_lstm.values,
                                 mode="lines+markers", name="LSTM",
                                 line=dict(color="rgb(227,119,194)", dash="dash")))
        if pond_fc_mlp is not None:
            fig.add_trace(go.Scatter(x=fc_index, y=va_pond_mlp.values,
                                     mode="lines+markers", name="Pond. MLP",
                                     line=dict(color="rgb(44,160,44)", dash="dot")))
            fig.add_trace(go.Scatter(x=fc_index, y=va_pond_lstm.values,
                                     mode="lines+markers", name="Pond. LSTM",
                                     line=dict(color="rgb(214,39,40)", dash="dot")))
        fig.update_layout(yaxis_title="%", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Variación {var_period_label} (%)")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=vp_hist.index, y=vp_hist.values,
                             name="Histórico", marker_color="steelblue"))
        fig.add_trace(go.Scatter(x=fc_index, y=vp_mlp.values,
                                 mode="lines+markers", name="MLP",
                                 line=dict(color="rgb(148,103,189)", dash="dash")))
        fig.add_trace(go.Scatter(x=fc_index, y=vp_lstm.values,
                                 mode="lines+markers", name="LSTM",
                                 line=dict(color="rgb(227,119,194)", dash="dash")))
        if pond_fc_mlp is not None:
            fig.add_trace(go.Scatter(x=fc_index, y=vp_pond_mlp.values,
                                     mode="lines+markers", name="Pond. MLP",
                                     line=dict(color="rgb(44,160,44)", dash="dot")))
            fig.add_trace(go.Scatter(x=fc_index, y=vp_pond_lstm.values,
                                     mode="lines+markers", name="Pond. LSTM",
                                     line=dict(color="rgb(214,39,40)", dash="dot")))
        fig.update_layout(yaxis_title="%", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # --- Tabla de pronosticos ---
    st.subheader("Tabla de Pronósticos")
    n_hist_nn = st.slider("Periodos de historia a incluir", 0, 120, PERIOD,
                          key="hist_nn")

    all_cols_nn = ["Nivel MLP", "Nivel LSTM"]
    if pond_fc_mlp is not None:
        all_cols_nn += ["Nivel Pond. MLP", "Nivel Pond. LSTM"]
    all_cols_nn += [f"Var.{var_short} MLP (%)", f"Var.{var_short} LSTM (%)"]
    if pond_fc_mlp is not None:
        all_cols_nn += [f"Var.{var_short} Pond.M (%)", f"Var.{var_short} Pond.L (%)"]
    if is_q:
        all_cols_nn += ["Var.m MLP (%)", "Var.m LSTM (%)"]
        if pond_fc_mlp is not None:
            all_cols_nn += ["Var.m Pond.M (%)", "Var.m Pond.L (%)"]
    all_cols_nn += ["Var.a MLP (%)", "Var.a LSTM (%)"]
    if pond_fc_mlp is not None:
        all_cols_nn += ["Var.a Pond.M (%)", "Var.a Pond.L (%)"]

    cols_visibles_nn = st.multiselect(
        "Columnas visibles", all_cols_nn, default=all_cols_nn, key="cols_nn")

    # Construir parte historica
    if n_hist_nn > 0:
        hist_slice_nn = serie_completa.iloc[-n_hist_nn:]
        hist_rows_nn = pd.DataFrame({"Nivel MLP": hist_slice_nn.values,
                                     "Nivel LSTM": hist_slice_nn.values}, index=hist_slice_nn.index)
        if pond_fc_mlp is not None:
            hist_rows_nn["Nivel Pond. MLP"] = hist_slice_nn.values
            hist_rows_nn["Nivel Pond. LSTM"] = hist_slice_nn.values
        hist_rows_nn[f"Var.{var_short} MLP (%)"] = var_period_hist.reindex(hist_slice_nn.index)
        hist_rows_nn[f"Var.{var_short} LSTM (%)"] = var_period_hist.reindex(hist_slice_nn.index)
        if pond_fc_mlp is not None:
            hist_rows_nn[f"Var.{var_short} Pond.M (%)"] = var_period_hist.reindex(hist_slice_nn.index)
            hist_rows_nn[f"Var.{var_short} Pond.L (%)"] = var_period_hist.reindex(hist_slice_nn.index)
        if is_q:
            hist_rows_nn["Var.m MLP (%)"] = var_mensual_hist_q.reindex(hist_slice_nn.index)
            hist_rows_nn["Var.m LSTM (%)"] = var_mensual_hist_q.reindex(hist_slice_nn.index)
            if pond_fc_mlp is not None:
                hist_rows_nn["Var.m Pond.M (%)"] = var_mensual_hist_q.reindex(hist_slice_nn.index)
                hist_rows_nn["Var.m Pond.L (%)"] = var_mensual_hist_q.reindex(hist_slice_nn.index)
        hist_rows_nn["Var.a MLP (%)"] = var_anual_hist.reindex(hist_slice_nn.index)
        hist_rows_nn["Var.a LSTM (%)"] = var_anual_hist.reindex(hist_slice_nn.index)
        if pond_fc_mlp is not None:
            hist_rows_nn["Var.a Pond.M (%)"] = var_anual_hist.reindex(hist_slice_nn.index)
            hist_rows_nn["Var.a Pond.L (%)"] = var_anual_hist.reindex(hist_slice_nn.index)

    # Construir parte pronóstico
    tabla_data = {
        "Nivel MLP": mlp_forecast.values,
        "Nivel LSTM": lstm_forecast.values,
    }
    if pond_fc_mlp is not None:
        tabla_data["Nivel Pond. MLP"] = pond_fc_mlp.values
        tabla_data["Nivel Pond. LSTM"] = pond_fc_lstm.values
    tabla_data.update({
        f"Var.{var_short} MLP (%)": vp_mlp.values,
        f"Var.{var_short} LSTM (%)": vp_lstm.values,
    })
    if pond_fc_mlp is not None:
        tabla_data[f"Var.{var_short} Pond.M (%)"] = vp_pond_mlp.values
        tabla_data[f"Var.{var_short} Pond.L (%)"] = vp_pond_lstm.values
    if is_q:
        tabla_data["Var.m MLP (%)"] = vm_mlp.values
        tabla_data["Var.m LSTM (%)"] = vm_lstm.values
        if pond_fc_mlp is not None:
            tabla_data["Var.m Pond.M (%)"] = vm_pond_mlp.values
            tabla_data["Var.m Pond.L (%)"] = vm_pond_lstm.values
    tabla_data.update({
        "Var.a MLP (%)": va_mlp.values,
        "Var.a LSTM (%)": va_lstm.values,
    })
    if pond_fc_mlp is not None:
        tabla_data["Var.a Pond.M (%)"] = va_pond_mlp.values
        tabla_data["Var.a Pond.L (%)"] = va_pond_lstm.values
    tabla_fc = pd.DataFrame(tabla_data, index=fc_index)

    if n_hist_nn > 0:
        tabla_fc = pd.concat([hist_rows_nn, tabla_fc])

    tabla_fc.index.name = "Fecha"
    tabla_fc_display = tabla_fc[[c for c in cols_visibles_nn if c in tabla_fc.columns]].copy()
    tabla_fc_display.index = _fmt_index(tabla_fc_display.index)
    st.dataframe(tabla_fc_display.style.format("{:.4f}"), use_container_width=True)
    st.download_button(
        "Descargar CSV (Excel)", tabla_fc.to_csv(),
        f"{serie.strip()}_pronostico_nn.csv", "text/csv", key="dl_nn",
    )

    # --- Residuales estandarizados (in-sample one-step-ahead) ---
    @st.cache_data
    def compute_nn_residuals(serie_name, values, index_str):
        """Calcula residuales in-sample (one-step-ahead) para MLP y LSTM."""
        cfg_r = nn_config[serie_name]
        models = load_nn_models(cfg_r["safe_name"], cfg_r["n_features"],
                                cfg_r["lstm_hidden_size"], cfg_r["lstm_num_layers"])
        mlp_m, mlp_sX, mlp_sY, lstm_m, lstm_sX, lstm_sY = models

        s = pd.Series(values, index=pd.DatetimeIndex(index_str))

        # Construir tabla supervisada
        dfr = pd.DataFrame({"y": s.values}, index=s.index)
        for i in range(1, N_LAGS + 1):
            dfr[f"lag_{i}"] = s.shift(i).values
        dfr["sin_month"] = np.sin(2 * np.pi * s.index.month / PERIOD)
        dfr["cos_month"] = np.cos(2 * np.pi * s.index.month / PERIOD)
        dfr = dfr.dropna()
        X = dfr.drop("y", axis=1).values
        y_real = dfr["y"].values
        resid_index = dfr.index.astype(str).tolist()

        # MLP predictions
        X_scaled = mlp_sX.transform(X)
        mlp_pred_scaled = mlp_m.predict(X_scaled)
        mlp_pred = mlp_sY.inverse_transform(mlp_pred_scaled.reshape(-1, 1)).ravel()
        mlp_resid = y_real - mlp_pred

        # LSTM predictions
        X_lstm_scaled = lstm_sX.transform(X)
        X_tensor = torch.FloatTensor(X_lstm_scaled).unsqueeze(1)
        lstm_m.eval()
        with torch.no_grad():
            lstm_pred_scaled = lstm_m(X_tensor).squeeze().numpy()
        lstm_pred = lstm_sY.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).ravel()
        lstm_resid = y_real - lstm_pred

        return mlp_resid, lstm_resid, resid_index

    mlp_resid, lstm_resid, resid_index_str = compute_nn_residuals(
        serie, serie_completa.values, serie_completa.index.astype(str).tolist())
    resid_index = _make_idx(resid_index_str)

    mlp_std_resid = (mlp_resid - np.mean(mlp_resid)) / np.std(mlp_resid)
    lstm_std_resid = (lstm_resid - np.mean(lstm_resid)) / np.std(lstm_resid)

    st.subheader("Residuales de los modelos")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=resid_index, y=mlp_std_resid,
                                 mode="lines", name="Residuales",
                                 line=dict(color="rgb(148,103,189)")))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=2, line_dash="dot", line_color="red", opacity=0.5)
        fig.add_hline(y=-2, line_dash="dot", line_color="red", opacity=0.5)
        fig.update_layout(title=f"MLP {cfg['mlp_params']['hidden_layer_sizes']} - Residuales estandarizados",
                          yaxis_title="Desv. estándar", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=resid_index, y=lstm_std_resid,
                                 mode="lines", name="Residuales",
                                 line=dict(color="rgb(227,119,194)")))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=2, line_dash="dot", line_color="red", opacity=0.5)
        fig.add_hline(y=-2, line_dash="dot", line_color="red", opacity=0.5)
        fig.update_layout(title=f"LSTM({cfg['lstm_hidden_size']}) - Residuales estandarizados",
                          yaxis_title="Desv. estándar", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Histogramas de residuales
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=mlp_std_resid, nbinsx=40,
                                   name="MLP", marker_color="rgb(148,103,189)", opacity=0.75))
        fig.update_layout(title=f"MLP {cfg['mlp_params']['hidden_layer_sizes']} - Distribucion de residuales",
                          xaxis_title="Desv. estándar", yaxis_title="Frecuencia",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col_h2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=lstm_std_resid, nbinsx=40,
                                   name="LSTM", marker_color="rgb(227,119,194)", opacity=0.75))
        fig.update_layout(title=f"LSTM({cfg['lstm_hidden_size']}) - Distribucion de residuales",
                          xaxis_title="Desv. estándar", yaxis_title="Frecuencia",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # --- Especificacion de modelos ---
    st.subheader("Especificacion de los modelos")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown(f"**MLP (sklearn)**")
        mlp_info = (
            f"Capas ocultas: {cfg['mlp_params']['hidden_layer_sizes']}\n"
            f"Regularizacion alpha: {cfg['mlp_params']['alpha']}\n"
            f"Activacion: relu\n"
            f"Max iter: 500, early stopping\n"
            f"Features: {N_LAGS} rezagos + sin(mes) + cos(mes) = {cfg['n_features']}\n"
            f"CV MSE (TimeSeriesSplit, 3 folds): {cfg['mlp_cv_mse']:.6f}\n"
            f"Escalado: StandardScaler en X e y"
        )
        st.code(mlp_info, language=None)

    with col_s2:
        st.markdown(f"**LSTM (PyTorch)**")
        lstm_info = (
            f"Arquitectura: LSTM({cfg['n_features']}, {cfg['lstm_hidden_size']}) -> Linear({cfg['lstm_hidden_size']}, 1)\n"
            f"Capas LSTM: {cfg['lstm_num_layers']}\n"
            f"Optimizador: Adam(lr=0.001)\n"
            f"Loss: MSELoss\n"
            f"Epochs: 100 (early stopping, patience=10)\n"
            f"Features: {N_LAGS} rezagos + sin(mes) + cos(mes) = {cfg['n_features']}\n"
            f"Validation MSE (últimos 12 meses): {cfg['lstm_val_mse']:.6f}\n"
            f"Escalado: StandardScaler en X e y"
        )
        st.code(lstm_info, language=None)


# ============================================================
# PÁGINA 5: Pronósticos Manuales
# ============================================================
elif pagina == "Manual":
    import json
    import numpy as np
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    st.header("Pronósticos Manuales")
    st.caption(f"Edita directamente la variación {var_period_label.lower()} (%) de cada subcomponente y observa cómo se propaga a los agregados")

    # --- Paths y configs ---
    ORDERS_PATH = os.path.join(os.path.dirname(__file__), "data",
                                "model_orders_q.json" if is_q else "model_orders.json")
    NN_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data",
                                   "nn_config_q.json" if is_q else "nn_config.json")
    NN_MODELS_DIR = os.path.join(os.path.dirname(__file__), "data",
                                  "nn_models_q" if is_q else "nn_models")

    modelos_disponibles = []
    all_orders = {}
    nn_config = {}

    if os.path.exists(ORDERS_PATH):
        with open(ORDERS_PATH, encoding="utf-8") as f:
            all_orders = json.load(f)
        modelos_disponibles.extend(["ARIMA", "SARIMA"])

    if os.path.exists(NN_CONFIG_PATH):
        with open(NN_CONFIG_PATH, encoding="utf-8") as f:
            nn_config = json.load(f)
        modelos_disponibles.extend(["MLP", "LSTM"])

    if not modelos_disponibles:
        flag = " --quincenal" if is_q else ""
        st.error(f"No hay modelos pre-entrenados. Ejecuta `python fit_orders.py{flag}` y/o `python fit_nn.py{flag}`.")
        st.stop()

    # Hojas = subcomponentes de menor nivel (sin hijos en JERARQUIA)
    hojas = [col for col in COLUMN_ORDER if col not in JERARQUIA]

    # --- Funciones de pronóstico ---
    @st.cache_data
    def _forecast_arima_sarima(serie_name, values, index_str, orders_dict, quincenal):
        idx = pd.DatetimeIndex(index_str)
        if not quincenal:
            idx.freq = "MS"
        s = pd.Series(values, index=idx)
        results = {}

        arima_ord = tuple(orders_dict["arima_order"])
        arima_tr = "c" if orders_dict.get("arima_intercept", False) else "n"
        model_a = SARIMAX(s, order=arima_ord, trend=arima_tr,
                          enforce_stationarity=False, enforce_invertibility=False)
        res_a = model_a.fit(disp=False)
        pred_a = res_a.get_forecast(steps=HORIZON)
        results["ARIMA"] = pred_a.predicted_mean.values

        sarima_ord = tuple(orders_dict["sarima_order"])
        sarima_seas = tuple(orders_dict["sarima_seasonal_order"])
        sarima_tr = "c" if orders_dict.get("sarima_intercept", False) else "n"
        model_s = SARIMAX(s, order=sarima_ord, seasonal_order=sarima_seas,
                          trend=sarima_tr, enforce_stationarity=False,
                          enforce_invertibility=False)
        res_s = model_s.fit(disp=False)
        pred_s = res_s.get_forecast(steps=HORIZON)
        results["SARIMA"] = pred_s.predicted_mean.values

        if quincenal:
            fc_idx = _next_q_dates(s.index[-1], HORIZON).astype(str).tolist()
        else:
            fc_idx = pred_a.predicted_mean.index.astype(str).tolist()
        return results, fc_idx

    N_LAGS = N_LAGS_NN

    if nn_config:
        import torch
        import torch.nn as tnn
        import joblib

        class LSTMModelManual(tnn.Module):
            def __init__(self, input_size=N_FEATURES_NN, hidden_size=32, num_layers=1):
                super().__init__()
                self.lstm = tnn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                     num_layers=num_layers, batch_first=True)
                self.fc = tnn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        @st.cache_resource
        def _load_nn_models(safe_name, n_features, hidden_size, num_layers):
            series_dir = os.path.join(NN_MODELS_DIR, safe_name)
            mlp_model = joblib.load(os.path.join(series_dir, "mlp_model.pkl"))
            mlp_sX = joblib.load(os.path.join(series_dir, "mlp_scaler_X.pkl"))
            mlp_sY = joblib.load(os.path.join(series_dir, "mlp_scaler_y.pkl"))
            lstm_model = LSTMModelManual(input_size=n_features, hidden_size=hidden_size,
                                         num_layers=num_layers)
            lstm_model.load_state_dict(
                torch.load(os.path.join(series_dir, "lstm_model.pt"), weights_only=True))
            lstm_model.eval()
            lstm_sX = joblib.load(os.path.join(series_dir, "lstm_scaler_X.pkl"))
            lstm_sY = joblib.load(os.path.join(series_dir, "lstm_scaler_y.pkl"))
            return mlp_model, mlp_sX, mlp_sY, lstm_model, lstm_sX, lstm_sY

        def _nn_recursive(model_type, model, scaler_X, scaler_y, last_values, last_month):
            preds = []
            window = list(last_values[-N_LAGS:])
            for i in range(HORIZON):
                month = (last_month + i) % 12 + 1
                lags = list(reversed(window[-N_LAGS:]))
                sin_m = np.sin(2 * np.pi * month / PERIOD)
                cos_m = np.cos(2 * np.pi * month / PERIOD)
                features = np.array(lags + [sin_m, cos_m]).reshape(1, -1)
                features_scaled = scaler_X.transform(features)
                if model_type == "mlp":
                    pred_scaled = model.predict(features_scaled)
                    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
                else:
                    x_tensor = torch.FloatTensor(features_scaled).unsqueeze(1)
                    with torch.no_grad():
                        pred_scaled = model(x_tensor).item()
                    pred = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
                preds.append(pred)
                window.append(pred)
            return np.array(preds)

        @st.cache_data
        def _forecast_nn(serie_name, values, index_str, quincenal):
            cfg = nn_config[serie_name]
            models = _load_nn_models(cfg["safe_name"], cfg["n_features"],
                                     cfg["lstm_hidden_size"], cfg["lstm_num_layers"])
            mlp_m, mlp_sX, mlp_sY, lstm_m, lstm_sX, lstm_sY = models
            s = pd.Series(values, index=pd.DatetimeIndex(index_str))
            last_month = s.index[-1].month
            mlp_fc = _nn_recursive("mlp", mlp_m, mlp_sX, mlp_sY, s.values, last_month)
            lstm_fc = _nn_recursive("lstm", lstm_m, lstm_sX, lstm_sY, s.values, last_month)
            if quincenal:
                fc_idx = _next_q_dates(s.index[-1], HORIZON)
            else:
                fc_idx = pd.date_range(s.index[-1] + pd.DateOffset(months=1),
                                       periods=HORIZON, freq="MS")
            return {"MLP": mlp_fc, "LSTM": lstm_fc}, fc_idx.astype(str).tolist()

    # --- Selección de modelo ---
    st.subheader("1. Modelo base")

    modelo_global = st.selectbox("Modelo base (aplica a todos por defecto)",
                                 modelos_disponibles, key="manual_modelo_global")

    usar_individual = st.checkbox("Seleccionar modelo por subcomponente", value=False)

    modelos_por_hoja = {}
    if usar_individual:
        cols_sel = st.columns(3)
        for i, hoja in enumerate(hojas):
            with cols_sel[i % 3]:
                modelos_por_hoja[hoja] = st.selectbox(
                    hoja.strip(), modelos_disponibles,
                    index=modelos_disponibles.index(modelo_global),
                    key=f"manual_modelo_{hoja}")
    else:
        for hoja in hojas:
            modelos_por_hoja[hoja] = modelo_global

    # --- Calcular pronósticos base de cada hoja ---
    with st.spinner("Calculando pronósticos base..."):
        base_forecasts = {}
        fc_index = None

        for hoja in hojas:
            s = df[hoja].dropna()
            _set_freq(s)
            modelo = modelos_por_hoja[hoja]

            if modelo in ("ARIMA", "SARIMA") and hoja in all_orders:
                res, fc_str = _forecast_arima_sarima(
                    hoja, s.values, s.index.astype(str).tolist(), all_orders[hoja], is_q)
                base_forecasts[hoja] = res[modelo]
                if fc_index is None:
                    fc_index = _make_idx(fc_str)
            elif modelo in ("MLP", "LSTM") and nn_config and hoja in nn_config:
                res, fc_str = _forecast_nn(hoja, s.values, s.index.astype(str).tolist(), is_q)
                base_forecasts[hoja] = res[modelo]
                if fc_index is None:
                    fc_index = _make_idx(fc_str)
            else:
                st.warning(f"No hay modelo {modelo} para '{hoja.strip()}'. Ejecuta el script correspondiente.")
                st.stop()

    # --- Solo mes t+1: variación mensual base de cada hoja ---
    mes_pronostico = fc_index[0]
    mes_label = _fmt_date(mes_pronostico)

    base_var_t1 = {}
    for hoja in hojas:
        last_val = df[hoja].dropna().iloc[-1]
        fc_t1 = base_forecasts[hoja][0]
        base_var_t1[hoja] = (fc_t1 / last_val - 1) * 100

    # --- Tabla editable traspuesta: drift directo por subcomponente ---
    st.subheader(f"2. Ajuste para {mes_label}")
    st.caption("Edita el Drift (pp) de cada subcomponente. Ajustado = Modelo + Drift.")

    nombres = [h.strip() for h in hojas]

    # Boton para resetear drifts
    def _reset_drifts():
        if "manual_drift_t" in st.session_state:
            del st.session_state["manual_drift_t"]

    st.button("Resetear drifts a 0", key="reset_drifts", on_click=_reset_drifts)

    # Calcular último dato de var. periodo y promedio histórico de la misma quincena/mes
    mes_target = mes_pronostico.month
    dia_target = mes_pronostico.day
    ultimo_vm = {}
    prom_mes_10a = {}
    for hoja, nombre in zip(hojas, nombres):
        s = df[hoja].dropna()
        vm_hist = s.pct_change() * 100
        ultimo_vm[nombre] = round(vm_hist.dropna().iloc[-1], 4)
        if is_q:
            # Filtrar misma quincena (mismo mes y mismo dia 1 o 16)
            vm_misma_q = vm_hist[(vm_hist.index.month == mes_target) & (vm_hist.index.day == dia_target)].dropna()
            vm_ultimo_10 = vm_misma_q[vm_misma_q.index.year >= (mes_pronostico.year - 10)]
        else:
            vm_mismo_mes = vm_hist[vm_hist.index.month == mes_target].dropna()
            vm_ultimo_10 = vm_mismo_mes[vm_mismo_mes.index.year >= (mes_pronostico.year - 10)]
        prom_mes_10a[nombre] = round(vm_ultimo_10.mean(), 4) if len(vm_ultimo_10) > 0 else np.nan

    # Tabla de referencia (no editable): histórico + modelo
    prom_label = f"Prom. {mes_pronostico.strftime('%b')}{(' Q' + str(1 if dia_target == 1 else 2)) if is_q else ''} 10a (%)"
    ref_t = pd.DataFrame({
        nombre: {
            f"Ult. Var.{var_short} (%)": ultimo_vm[nombre],
            prom_label: prom_mes_10a[nombre],
            "Modelo (%)": round(base_var_t1[hoja], 4),
        }
        for hoja, nombre in zip(hojas, nombres)
    })
    st.dataframe(ref_t.style.format("{:.4f}"), use_container_width=True)

    # Tabla editable: solo Drift
    drift_t = pd.DataFrame({
        nombre: {"Drift (pp)": 0.0}
        for nombre in nombres
    })

    edited_drift = st.data_editor(
        drift_t,
        use_container_width=True,
        key="manual_drift_t",
        num_rows="fixed",
        disabled={"_index": True},
        column_config={
            nombre: st.column_config.NumberColumn(format="%.4f")
            for nombre in nombres
        },
    )

    # Recalcular ajustado y niveles
    adjusted_forecasts = {}  # nivel t+1
    base_levels = {}         # nivel t (hojas: dato real; padres: suma ponderada)
    ajustado_row = {}

    for hoja, nombre in zip(hojas, nombres):
        last_val = df[hoja].dropna().iloc[-1]
        base_levels[hoja] = last_val
        modelo_var = base_var_t1[hoja]
        drift_val = edited_drift.loc["Drift (pp)", nombre]
        ajustado_var = modelo_var + drift_val
        ajustado_nivel = last_val * (1 + ajustado_var / 100)
        adjusted_forecasts[hoja] = np.array([ajustado_nivel])
        ajustado_row[nombre] = round(ajustado_var, 4)

    # Mostrar fila Ajustado debajo de la tabla editable
    ajustado_df = pd.DataFrame(ajustado_row, index=["Ajustado (%)"])
    st.dataframe(ajustado_df.style.format("{:.4f}"), use_container_width=True)

    # --- Agregar hacia arriba por jerarquia ---
    # Calcula nivel t (base) y t+1 (forecast) con la misma formula
    # para que las variaciones sean internamente consistentes
    def aggregate_recursive(parent):
        hijos = JERARQUIA[parent]
        for hijo in hijos:
            if hijo in JERARQUIA:
                aggregate_recursive(hijo)
        peso_parent = PONDERADORES[parent]
        fc_result = np.zeros(1)
        base_result = 0.0
        for hijo in hijos:
            fc_result += adjusted_forecasts[hijo] * PONDERADORES[hijo]
            base_result += base_levels[hijo] * PONDERADORES[hijo]
        adjusted_forecasts[parent] = fc_result / peso_parent
        base_levels[parent] = base_result / peso_parent

    aggregate_recursive("Indice General")

    # --- Tabla de resultados: todos los componentes ---
    st.subheader("3. Pronóstico agregado")

    # Tambien calcular base_levels_12m para var. anual con misma formula
    base_levels_12m = {}
    for hoja in hojas:
        s = df[hoja].dropna()
        base_levels_12m[hoja] = s.iloc[-PERIOD] if len(s) >= PERIOD else np.nan

    def aggregate_12m(parent):
        hijos = JERARQUIA[parent]
        for hijo in hijos:
            if hijo in JERARQUIA:
                aggregate_12m(hijo)
        peso_parent = PONDERADORES[parent]
        result = 0.0
        for hijo in hijos:
            if np.isnan(base_levels_12m.get(hijo, np.nan)):
                base_levels_12m[parent] = np.nan
                return
            result += base_levels_12m[hijo] * PONDERADORES[hijo]
        base_levels_12m[parent] = result / peso_parent

    aggregate_12m("Indice General")

    agg_data = {}
    for col in COLUMN_ORDER:
        if col in adjusted_forecasts:
            base_val = base_levels[col]
            fc_val = adjusted_forecasts[col][0]
            vm = (fc_val / base_val - 1) * 100
            val_12m = base_levels_12m.get(col, np.nan)
            va = (fc_val / val_12m - 1) * 100 if not np.isnan(val_12m) else np.nan
            agg_data[col] = {
                "Pond. (%)": PONDERADORES.get(col, np.nan),
                "Nivel t": round(base_val, 4),
                f"Nivel {mes_label}": round(fc_val, 4),
                f"Var. {var_period_label} (%)": round(vm, 4),
                "Var. Anual (%)": round(va, 4),
            }

    agg_df = pd.DataFrame(agg_data)

    def _bold_vm(styler):
        return styler.apply(lambda row: [
            "font-weight: bold" if row.name == f"Var. {var_period_label} (%)" else ""
            for _ in row], axis=1)

    st.dataframe(_bold_vm(agg_df.style).format("{:.4f}"), use_container_width=True)
    st.download_button("Descargar CSV", agg_df.to_csv(),
                       f"pronostico_manual_{mes_pronostico.strftime('%Y_%m')}.csv",
                       "text/csv", key="dl_man_agg")
