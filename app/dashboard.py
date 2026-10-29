"""Dashboard del MVP: streamlit run app/dashboard.py

Lee los archivos generados en outputs/ por `agroscore demo` (o procesar/entrenar/scorear).
Estilo según el manual de marca de Advanta (colores, Noto Sans, logo). El tema está en .streamlit/config.toml.
"""

import json
import os
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from agroscore.config import RAIZ_PROYECTO, cargar_config
from agroscore.score import ParametrosScore, calcular_score

# Carpeta de resultados; otra con: AGROSCORE_SALIDA=outputs/aliado streamlit run app/dashboard.py
SALIDA = Path(os.environ.get("AGROSCORE_SALIDA", RAIZ_PROYECTO / "outputs"))
ASSETS = Path(__file__).parent / "assets"

# Paleta de marca. Cultivos: azul (soja) y Gold, el color de maíz en el sistema de productos.
# Validada para daltonismo y contraste en ambos modos.
PALETA = {
    "light": {"soja": "#1961AC", "maiz": "#EA580C", "linea": "#1961AC", "texto": "#0C1C49",
              "tenue": "#8A93A6", "grilla": "rgba(12,28,73,0.08)", "marco": "#FFFFFF"},
    "dark": {"soja": "#4F93E3", "maiz": "#EA580C", "linea": "#4F93E3", "texto": "#EEF2F8",
             "tenue": "#8C9BBD", "grilla": "rgba(238,242,248,0.10)", "marco": "#0B1735"},
}
VERDE_ADVANTA, GOLD_CLARO = "#13A538", "#FBBA00"
# Riesgo: siempre con ícono + texto, nunca solo color.
RIESGO = {
    "alto": {"color": "#E7344C", "badge": "red", "icono": ":material/error:", "emoji": "🔴"},
    "medio": {"color": "#FBBA00", "badge": "yellow", "icono": ":material/warning:", "emoji": "🟡"},
    "bajo": {"color": "#13A538", "badge": "green", "icono": ":material/check_circle:", "emoji": "🟢"},
    "sin dato": {"color": "#8A93A6", "badge": "gray", "icono": ":material/help:", "emoji": "⚪"},
}
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
FUENTE = "Noto Sans, Arial, sans-serif"

st.set_page_config(page_title="Scoring agro · Advanta", page_icon=str(ASSETS / "advanta_isotipo.png"),
                   layout="wide")


# --- Formato --------------------------------------------------------------------------------------
def num(x, dec: int = 0) -> str:
    """Número con formato argentino: 1.418 y 0,55."""
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{dec}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def paleta() -> dict:
    tipo = getattr(st.context.theme, "type", None)
    return PALETA["dark" if tipo == "dark" else "light"]


def estilo(fig: go.Figure, alto: int, **kw) -> go.Figure:
    p = paleta()
    base = dict(height=alto, margin=dict(l=8, r=8, t=28, b=8), separators=",.",
                font=dict(family=FUENTE, color=p["texto"], size=13),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(font_family=FUENTE))
    fig.update_layout(**(base | kw))
    fig.update_xaxes(gridcolor=p["grilla"], zeroline=False, linecolor=p["grilla"])
    fig.update_yaxes(gridcolor=p["grilla"], zeroline=False, linecolor=p["grilla"])
    return fig


def ticks_mensuales(desde, hasta):
    meses = pd.date_range(pd.Timestamp(desde).to_period("M").to_timestamp(), hasta, freq="MS")
    return list(meses), [f"{MESES[m.month - 1]} {m.strftime('%y')}" for m in meses]


# --- Datos ----------------------------------------------------------------------------------------
@st.cache_data
def cargar_salidas(directorio: str):
    d = Path(directorio)
    leer = lambda nombre, **kw: pd.read_csv(d / nombre, dtype={"lote_id": str, "campania": str}, **kw)  # noqa: E731
    return {
        "scores": leer("scores.csv", parse_dates=["fecha_pico", "fecha_siembra"]),
        "series": leer("series.csv", parse_dates=["fecha"]),
        "obs": leer("observaciones.csv", parse_dates=["fecha"]),
        "oof": leer("predicciones_validacion.csv"),
        "por_campania": pd.read_csv(d / "metricas_por_campania.csv", dtype={"campania_test": str}),
        "metricas": json.loads((d / "metricas.json").read_text(encoding="utf-8")),
    }


# --- Gráficos -------------------------------------------------------------------------------------
def grafico_ndvi(serie: pd.DataFrame, obs: pd.DataFrame, fila, cfg) -> go.Figure:
    p = paleta()
    fig = go.Figure()
    if pd.notna(fila["fecha_pico"]):
        pc = cfg["periodo_critico"]
        fig.add_vrect(x0=fila["fecha_pico"] - pd.Timedelta(days=pc["dias_antes_pico"]),
                      x1=fila["fecha_pico"] + pd.Timedelta(days=pc["dias_despues_pico"]),
                      fillcolor=GOLD_CLARO, opacity=0.14, line_width=0, layer="below",
                      annotation_text="período crítico", annotation_position="top left",
                      annotation_font=dict(color=p["tenue"], size=11))
    largo = serie["hueco_largo"].astype(bool)
    tramos = (largo != largo.shift()).cumsum()[largo]
    for _, t in serie[largo].groupby(tramos):
        fig.add_vrect(x0=t["fecha"].min(), x1=t["fecha"].max(), fillcolor=p["tenue"], opacity=0.14,
                      line_width=0, layer="below", annotation_text="sin imágenes válidas",
                      annotation_position="bottom left", annotation_font=dict(color=p["tenue"], size=11))
    if pd.notna(fila.get("fecha_siembra")):
        fig.add_vline(x=fila["fecha_siembra"], line=dict(color=p["tenue"], width=1, dash="dot"))
        fig.add_annotation(x=fila["fecha_siembra"], y=0.95, text="siembra", showarrow=False, xanchor="left",
                           xshift=4, font=dict(color=p["tenue"], size=11))

    usadas, descartadas = obs[obs["usada"]], obs[~obs["usada"]]
    fig.add_trace(go.Scatter(
        x=descartadas["fecha"], y=descartadas["ndvi"], mode="markers", name="Descartada por nubes",
        marker=dict(size=8, color="rgba(0,0,0,0)", line=dict(color=p["tenue"], width=1.5)),
        customdata=descartadas["frac_valida"] * 100,
        hovertemplate="%{x|%d/%m/%Y}<br>NDVI %{y:.2f}<br>Píxeles válidos %{customdata:.0f} %<extra>Descartada</extra>"))
    fig.add_trace(go.Scatter(
        x=serie["fecha"], y=serie["suave"], mode="lines", name="Curva suavizada",
        line=dict(color=p["linea"], width=2), hovertemplate="%{x|%d/%m/%Y}<br>NDVI %{y:.2f}<extra>Curva</extra>"))
    fig.add_trace(go.Scatter(
        x=usadas["fecha"], y=usadas["ndvi"], mode="markers", name="Imagen válida",
        marker=dict(size=8, color=p["linea"], line=dict(color=p["marco"], width=2)),
        customdata=usadas["frac_valida"] * 100,
        hovertemplate="%{x|%d/%m/%Y}<br>NDVI %{y:.2f}<br>Píxeles válidos %{customdata:.0f} %<extra>Válida</extra>"))
    if pd.notna(fila["fecha_pico"]):
        fig.add_annotation(x=fila["fecha_pico"], y=fila["ndvi_pico"], text=f"<b>pico {num(fila['ndvi_pico'], 2)}</b>",
                           showarrow=True, arrowhead=0, arrowcolor=p["tenue"], ay=-28,
                           font=dict(color=p["texto"], size=12))
    valores, etiquetas = ticks_mensuales(serie["fecha"].min(), serie["fecha"].max())
    return estilo(fig, 340, hovermode="closest",
                  yaxis=dict(title="NDVI", range=[-0.05, 1.0]),
                  xaxis=dict(tickvals=valores, ticktext=etiquetas, showgrid=False),
                  legend=dict(orientation="h", y=-0.12, x=0))


def grafico_score(score: float, p_score: ParametrosScore) -> go.Figure:
    """Barra 0-1000 con las zonas de riesgo y la posición del lote."""
    p = paleta()
    rango = p_score.cobertura_score_1000 - p_score.cobertura_score_0
    a_score = lambda c: min(1000, max(0, 1000 * (c - p_score.cobertura_score_0) / rango))  # noqa: E731
    limite_alto, limite_bajo = a_score(p_score.umbral_riesgo_alto), a_score(p_score.umbral_riesgo_bajo)
    fig = go.Figure()
    for nombre, x0, x1 in [("alto", 0, limite_alto), ("medio", limite_alto, limite_bajo), ("bajo", limite_bajo, 1000)]:
        if x1 <= x0:
            continue
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1, line=dict(color=p["marco"], width=2),
                      fillcolor=RIESGO[nombre]["color"], opacity=0.25, layer="below")
        fig.add_annotation(x=(x0 + x1) / 2, y=0.5, text=f"Riesgo {nombre}", showarrow=False,
                           font=dict(color=p["texto"], size=12))
    if pd.notna(score):
        fig.add_shape(type="line", x0=score, x1=score, y0=-0.15, y1=1.15, line=dict(color=p["texto"], width=3))
        fig.add_annotation(x=score, y=1.15, text=f"<b>{num(score)}</b>", showarrow=False, yanchor="bottom",
                           font=dict(color=p["texto"], size=14))
    return estilo(fig, 110, showlegend=False, margin=dict(l=8, r=8, t=26, b=4),
                  xaxis=dict(range=[-8, 1008], tickvals=[0, 250, 500, 750, 1000], showgrid=False, fixedrange=True),
                  yaxis=dict(range=[-0.2, 1.4], visible=False, fixedrange=True))


def grafico_ingreso_costo(ingreso: float, costo: float) -> go.Figure:
    p = paleta()
    fig = go.Figure(go.Bar(
        y=["Costo o deuda", "Ingreso con margen"], x=[costo, ingreso], orientation="h", width=0.55,
        marker=dict(color=[p["tenue"], p["linea"]], cornerradius=4, line=dict(color=p["marco"], width=2)),
        text=[f"{num(costo)} USD/ha", f"{num(ingreso)} USD/ha"], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:,.0f} USD/ha<extra></extra>"))
    return estilo(fig, 150, showlegend=False, margin=dict(l=8, r=90, t=10, b=8),
                  xaxis=dict(visible=False, range=[0, max(ingreso, costo) * 1.05]),
                  yaxis=dict(showgrid=False))


def grafico_estimado_vs_real(oof: pd.DataFrame, seleccion) -> go.Figure:
    p = paleta()
    fig = go.Figure()
    lim = [0, max(oof["rinde_kg_ha"].max(), oof["rinde_validacion_kg_ha"].max()) * 1.05]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Estimado = real",
                             line=dict(color=p["tenue"], width=1, dash="dot"), hoverinfo="skip"))
    for cultivo, g in oof.groupby("cultivo"):
        fig.add_trace(go.Scatter(
            x=g["rinde_kg_ha"], y=g["rinde_validacion_kg_ha"], mode="markers",
            name="Maíz" if cultivo == "maiz" else cultivo.capitalize(),
            marker=dict(size=8, color=p.get(cultivo, p["tenue"]), opacity=0.8, line=dict(color=p["marco"], width=1)),
            customdata=g[["lote_id", "campania"]],
            hovertemplate="%{customdata[0]} · %{customdata[1]}<br>Real %{x:,.0f} kg/ha<br>"
                          "Estimado %{y:,.0f} kg/ha<extra></extra>"))
    sel = oof[(oof["lote_id"] == seleccion["lote_id"]) & (oof["campania"] == seleccion["campania"])]
    if not sel.empty:
        fig.add_trace(go.Scatter(x=sel["rinde_kg_ha"], y=sel["rinde_validacion_kg_ha"], mode="markers",
                                 name="Lote seleccionado", hoverinfo="skip",
                                 marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color=p["texto"], width=2))))
    return estilo(fig, 400, xaxis=dict(title="Rinde real (kg/ha)", range=lim, tickformat=",d"),
                  yaxis=dict(title="Rinde estimado (kg/ha)", range=lim, tickformat=",d"),
                  legend=dict(orientation="h", y=-0.18, x=0))


def grafico_cartera(scores: pd.DataFrame) -> go.Figure:
    p = paleta()
    conteo = scores.groupby(["campania", "riesgo"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for nivel in ["alto", "medio", "bajo"]:
        if nivel in conteo:
            fig.add_trace(go.Bar(
                x=conteo.index, y=conteo[nivel], name=f"Riesgo {nivel}",
                marker=dict(color=RIESGO[nivel]["color"], line=dict(color=p["marco"], width=2)),
                hovertemplate="%{x}<br>%{y} lotes<extra>Riesgo " + nivel + "</extra>"))
    return estilo(fig, 320, barmode="stack", bargap=0.45, yaxis=dict(title="Lotes"),
                  xaxis=dict(showgrid=False), legend=dict(orientation="h", y=-0.15, x=0, traceorder="normal"))


def mapa(geometria: str, color: str):
    capa = folium.GeoJson(json.loads(geometria),
                          style_function=lambda _: {"color": color, "weight": 3, "fillOpacity": 0.12})
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Calles").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri — Esri, Maxar, Earthstar Geographics", name="Satélite (Esri)", show=False,
    ).add_to(m)
    capa.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(capa.get_bounds())
    return m


# --- Componentes ----------------------------------------------------------------------------------
def tarjeta_paso(numero: int, titulo: str, valor: str, detalle: str):
    with st.container(border=True):
        st.caption(f"**{numero}** · {titulo}")
        st.markdown(f"<div style='font-size:1.55rem;font-weight:700;line-height:1.2'>{valor}</div>",
                    unsafe_allow_html=True)
        st.caption(detalle)


def badge_riesgo(riesgo: str):
    r = RIESGO.get(riesgo, RIESGO["sin dato"])
    st.badge(f"Riesgo {riesgo}", icon=r["icono"], color=r["badge"])


def nombre_cultivo(cultivo: str) -> str:
    return "Maíz" if cultivo == "maiz" else cultivo.capitalize()


# --- Página ---------------------------------------------------------------------------------------
if not (SALIDA / "scores.csv").exists():
    st.error("No hay resultados en `outputs/`. Corré primero: `agroscore demo`")
    st.stop()

datos = cargar_salidas(str(SALIDA))
cfg = cargar_config()
scores = datos["scores"]
metricas = datos["metricas"]

with st.sidebar:
    st.image(str(ASSETS / "advanta_logo_tarjeta.svg"), width=170)
    st.subheader("Lote")
    campanias = sorted(scores["campania"].unique())
    campania = st.selectbox("Campaña", campanias, index=len(campanias) - 1)
    cultivos = {"Todos": None, "Soja": "soja", "Maíz": "maiz"}
    filtro_cultivo = cultivos[st.segmented_control("Cultivo", list(cultivos), default="Todos") or "Todos"]
    en_campania = scores[scores["campania"] == campania]
    if filtro_cultivo:
        en_campania = en_campania[en_campania["cultivo"] == filtro_cultivo]
    if en_campania.empty:
        st.warning("No hay lotes con ese filtro.")
        st.stop()
    por_id = en_campania.set_index("lote_id")
    lote_id = st.selectbox(
        "Lote", por_id.index.tolist(),
        format_func=lambda i: f"{i} · {nombre_cultivo(por_id.loc[i, 'cultivo'])} · "
                              f"{RIESGO[por_id.loc[i, 'riesgo']]['emoji']} {por_id.loc[i, 'riesgo']}")
    fila = por_id.loc[lote_id].copy()
    fila["lote_id"] = lote_id

    st.subheader("Supuestos del score")
    precio = st.number_input("Precio de referencia (USD/t)", min_value=1.0, value=float(fila["precio_usd_t"]), step=5.0,
                             format="%.0f")
    costo = st.number_input("Costo o deuda (USD/ha)", min_value=1.0, value=float(fila["costo_ref_usd_ha"]),
                            step=10.0, format="%.0f")
    k = st.slider("Margen por error del modelo (× error típico)", 0.0, 2.0, float(cfg["score"]["k_incertidumbre"]), 0.1)
    st.caption("Estos valores solo cambian la vista del lote; no modifican la configuración.")
    st.divider()
    st.caption("Herramienta de apoyo al análisis, **no** una decisión de crédito. Con datos reales: contiene "
               "datos modificados de Copernicus Sentinel; clima de NASA POWER.")

p_score = ParametrosScore(**{**ParametrosScore.desde_config(cfg).__dict__, "k_incertidumbre": k})
rmse = metricas["rmse_por_cultivo"].get(fila["cultivo"])
resultado = calcular_score(fila["rinde_estimado_kg_ha"], rmse, precio, costo, p_score)

st.markdown(f"# Scoring de riesgo agro <span style='color:{VERDE_ADVANTA}'>//</span>", unsafe_allow_html=True)
st.caption(":material/info: MVP · Rinde estimado con Sentinel-2 y NASA POWER · "
           "Herramienta de apoyo, no una decisión de crédito")

tab_lote, tab_cartera, tab_modelo = st.tabs([":material/agriculture: Lote", ":material/view_list: Cartera",
                                             ":material/insights: Modelo"])

with tab_lote:
    encabezado, etiquetas = st.columns([2, 3], vertical_alignment="center")
    encabezado.markdown(f"## Lote {lote_id}")
    with etiquetas, st.container(horizontal=True, horizontal_alignment="right"):
        st.badge(nombre_cultivo(fila["cultivo"]), icon=":material/grass:", color="blue")
        st.badge(f"Campaña {campania}", icon=":material/calendar_month:", color="gray")
        st.badge(f"{num(fila['area_ha'])} ha", icon=":material/straighten:", color="gray")
        badge_riesgo(resultado["riesgo"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rinde estimado (kg/ha)", num(fila["rinde_estimado_kg_ha"]), border=True,
              help=f"Origen: {fila['origen_estimacion']}")
    if pd.isna(fila["rinde_kg_ha"]):
        m2.metric("Rinde real (kg/ha)", "sin dato", border=True)
    else:
        diferencia = fila["rinde_estimado_kg_ha"] - fila["rinde_kg_ha"]
        m2.metric("Rinde real (kg/ha)", num(fila["rinde_kg_ha"]), border=True, delta_color="off",
                  delta=f"{'+' if diferencia > 0 else ''}{num(diferencia)} estimado",
                  help="Diferencia: rinde estimado − rinde real")
    m3.metric("Score (de 1000)", num(resultado["score"]), border=True)
    m4.metric("Cobertura del costo", f"{num(resultado['cobertura'], 2)} ×", border=True,
              help="Ingreso estimado con margen dividido por el costo o deuda por hectárea")

    st.plotly_chart(grafico_score(resultado["score"], p_score), width="stretch", config={"displayModeBar": False})

    st.markdown("#### Cómo se calcula")
    pasos, barras = st.columns([3, 2], vertical_alignment="center")
    with pasos:
        a, b = st.columns(2)
        with a:
            tarjeta_paso(1, "Rinde estimado", f"{num(fila['rinde_estimado_kg_ha'])} kg/ha",
                         "Modelo de rinde con NDVI y clima")
            tarjeta_paso(3, "Ingreso", f"{num(resultado['ingreso_usd_ha'])} USD/ha",
                         f"{num(resultado['rinde_pesimista_kg_ha'] / 1000, 2)} t/ha × {num(precio)} USD/t")
        with b:
            tarjeta_paso(2, "Rinde con margen", f"{num(resultado['rinde_pesimista_kg_ha'])} kg/ha",
                         f"− {num(k, 1)} × error típico ({num(rmse)} kg/ha)")
            tarjeta_paso(4, "Cobertura", f"{num(resultado['cobertura'], 2)} ×",
                         f"{num(resultado['ingreso_usd_ha'])} ÷ {num(costo)} USD/ha de costo")
    with barras:
        st.plotly_chart(grafico_ingreso_costo(resultado["ingreso_usd_ha"], costo), width="stretch",
                        config={"displayModeBar": False})

    col_mapa, col_curva = st.columns([2, 3])
    with col_mapa:
        st.markdown("#### Ubicación")
        st_folium(mapa(fila["geometria"], RIESGO[resultado["riesgo"]]["color"]), height=340,
                  use_container_width=True, returned_objects=[])
    with col_curva:
        st.markdown("#### Curva de NDVI")
        clave = lambda df: df[(df["lote_id"] == lote_id) & (df["campania"] == campania)]  # noqa: E731
        st.plotly_chart(grafico_ndvi(clave(datos["series"]), clave(datos["obs"]), fila, cfg), width="stretch")

    st.markdown("#### Confianza de los datos")
    c1, c2, c3 = st.columns(3)
    c1.metric("Imágenes válidas", num(fila["n_obs_validas"]), border=True)
    c2.metric("Mayor hueco sin imágenes", f"{num(fila['max_hueco_dias'])} días", border=True)
    c3.metric("Días interpolados en huecos largos", f"{num(fila['frac_dias_hueco_largo'] * 100)} %", border=True)
    if fila["max_hueco_dias"] > cfg["series"]["max_hueco_dias"]:
        st.warning(f"Este lote tiene huecos de más de {cfg['series']['max_hueco_dias']} días sin imágenes: la "
                   "curva en esos tramos es interpolada y la estimación es menos confiable.", icon=":material/cloud:")

with tab_cartera:
    st.markdown(f"## Cartera · campaña {campania}")
    st.caption("Calculado con los precios y costos de referencia de la configuración.")
    total = len(en_campania)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lotes", num(total), border=True)
    for col, nivel in zip([k2, k3, k4], ["alto", "medio", "bajo"]):
        n = int((en_campania["riesgo"] == nivel).sum())
        col.metric(f"{RIESGO[nivel]['emoji']} Riesgo {nivel}", num(n), border=True,
                   delta=f"{num(100 * n / total)} % de la cartera", delta_color="off", delta_arrow="off")

    graf, tabla = st.columns([1, 2])
    with graf:
        st.markdown("#### Riesgo por campaña")
        base = scores if filtro_cultivo is None else scores[scores["cultivo"] == filtro_cultivo]
        st.plotly_chart(grafico_cartera(base), width="stretch", config={"displayModeBar": False})
    with tabla:
        st.markdown("#### Detalle")
        vista = pd.DataFrame({
            "Lote": en_campania["lote_id"],
            "Cultivo": en_campania["cultivo"].map(nombre_cultivo),
            "Estimado": en_campania["rinde_estimado_kg_ha"].map(num),
            "Real": en_campania["rinde_kg_ha"].map(num),
            "Cobertura": en_campania["cobertura"].map(lambda c: num(c, 2)),
            "Score": en_campania["score"],
            "Riesgo": en_campania["riesgo"].map(lambda r: f"{RIESGO[r]['emoji']} {r.capitalize()}"),
        }).sort_values("Score")
        st.dataframe(vista, hide_index=True, width="stretch", height=360, column_config={
            "Estimado": st.column_config.TextColumn("Estimado", help="Rinde estimado (kg/ha)", width="small"),
            "Real": st.column_config.TextColumn("Real", help="Rinde real (kg/ha)", width="small"),
            "Cobertura": st.column_config.TextColumn("Cobertura", width="small"),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1000, format="%d",
                                                     width="small"),
        })
        st.caption("Rindes en kg/ha. Ordenado de menor a mayor score.")

with tab_modelo:
    st.markdown("## Rendimiento del modelo")
    st.caption(f"Validación por campaña: cada campaña se evalúa con un modelo entrenado sin ella. Modelo elegido: "
               f"**{metricas['tipo']}** · {metricas['n_lotes']} lotes con rinde conocido · campañas "
               f"{', '.join(metricas['campanias'])}")

    por_cultivo = metricas["metricas_por_cultivo"][metricas["tipo"]]
    baseline = metricas["metricas_por_cultivo"]["baseline"]
    for col, (cultivo, m) in zip(st.columns(len(por_cultivo)), por_cultivo.items()):
        with col, st.container(border=True):
            st.markdown(f"#### {nombre_cultivo(cultivo)}")
            st.caption("Errores en kg/ha. La flecha compara con predecir el rinde promedio del cultivo.")
            a, b, c = st.columns(3)
            a.metric("Error medio", num(m["mae_kg_ha"]),
                     delta=num(m["mae_kg_ha"] - baseline[cultivo]["mae_kg_ha"]), delta_color="inverse",
                     help="MAE: error absoluto promedio")
            b.metric("RMSE", num(m["rmse_kg_ha"]), help="Castiga más los errores grandes. Lo usa el score.")
            c.metric("R²", num(m["r2"], 2), help="1 = perfecto · 0 = igual que predecir el promedio")

    izq, der = st.columns([3, 2])
    with izq:
        st.markdown("#### Rinde estimado vs. real")
        st.caption("Cada punto se estimó con un modelo que no vio su campaña.")
        st.plotly_chart(grafico_estimado_vs_real(datos["oof"], fila), width="stretch")
    with der:
        st.markdown("#### Por campaña evaluada")
        tabla = datos["por_campania"]
        tabla = tabla[tabla["modelo"] == metricas["tipo"]]
        st.dataframe(pd.DataFrame({
            "Campaña": tabla["campania_test"],
            "Lotes": tabla["n"],
            "Error medio": tabla["mae_kg_ha"].map(num),
            "R²": tabla["r2"].map(lambda v: num(v, 2)),
            "Sesgo": tabla["sesgo_kg_ha"].map(lambda v: ("+" if v > 0 else "") + num(v)),
        }), hide_index=True, width="stretch", column_config={
            "Sesgo": st.column_config.TextColumn("Sesgo", help="Positivo = el modelo sobreestima (kg/ha)")})
        st.caption("Un sesgo grande en una campaña extrema (por ejemplo, una sequía) indica que el modelo no vio "
                   "años parecidos al entrenar.")
        with st.expander("Comparación entre modelos"):
            comp = pd.DataFrame(metricas["metricas_globales"]).T
            st.dataframe(pd.DataFrame({
                "Error medio": comp["mae_kg_ha"].map(num),
                "RMSE": comp["rmse_kg_ha"].map(num),
                "R²": comp["r2"].map(lambda v: num(v, 2)),
            }, index=comp.index.rename("Modelo")), width="stretch")
