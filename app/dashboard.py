"""Dashboard del MVP: streamlit run app/dashboard.py

Lee los archivos generados en outputs/ por `agroscore demo` (o procesar/entrenar/scorear).
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
# Paleta: categórica (identidad) y de estado (riesgo). El riesgo siempre va con ícono + texto.
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#8a8985"
COLOR_CULTIVO = {"soja": AZUL, "maiz": NARANJA}
ESTADO_RIESGO = {"bajo": ("🟢", "#0ca30c"), "medio": ("🟡", "#fab219"), "alto": ("🔴", "#d03b3b"),
                 "sin dato": ("⚪", GRIS)}

st.set_page_config(page_title="Scoring agro MVP", page_icon="🌱", layout="wide")


@st.cache_data
def cargar_salidas(directorio: str):
    d = Path(directorio)
    leer = lambda nombre, **kw: pd.read_csv(d / nombre, dtype={"lote_id": str, "campania": str}, **kw)  # noqa: E731
    return {
        "scores": leer("scores.csv"),
        "series": leer("series.csv", parse_dates=["fecha"]),
        "obs": leer("observaciones.csv", parse_dates=["fecha"]),
        "oof": leer("predicciones_validacion.csv"),
        "por_campania": pd.read_csv(d / "metricas_por_campania.csv", dtype={"campania_test": str}),
        "metricas": json.loads((d / "metricas.json").read_text(encoding="utf-8")),
    }


def etiqueta_riesgo(riesgo: str) -> str:
    return f"{ESTADO_RIESGO.get(riesgo, ESTADO_RIESGO['sin dato'])[0]} {riesgo.capitalize()}"


def grafico_ndvi(serie: pd.DataFrame, obs: pd.DataFrame, fila) -> go.Figure:
    fig = go.Figure()
    # Huecos largos sin observaciones válidas: sombreado de fondo.
    largo = serie["hueco_largo"].astype(bool)
    tramos = (largo != largo.shift()).cumsum()[largo]
    for _, t in serie[largo].groupby(tramos):
        fig.add_vrect(x0=t["fecha"].min(), x1=t["fecha"].max(), fillcolor=GRIS, opacity=0.15, line_width=0,
                      annotation_text="hueco largo", annotation_position="top left",
                      annotation_font_color=GRIS)
    usadas, descartadas = obs[obs["usada"]], obs[~obs["usada"]]
    fig.add_trace(go.Scatter(
        x=descartadas["fecha"], y=descartadas["ndvi"], mode="markers", name="Descartada (nubes)",
        marker=dict(size=8, color="rgba(0,0,0,0)", line=dict(color=GRIS, width=1.5)),
        customdata=descartadas["frac_valida"],
        hovertemplate="%{x|%d/%m/%Y}<br>NDVI %{y:.2f}<br>Píxeles válidos %{customdata:.0%}<extra>Descartada</extra>"))
    fig.add_trace(go.Scatter(
        x=serie["fecha"], y=serie["suave"], mode="lines", name="Curva suavizada",
        line=dict(color=AZUL, width=2), hovertemplate="%{x|%d/%m/%Y}<br>NDVI %{y:.2f}<extra>Suavizada</extra>"))
    fig.add_trace(go.Scatter(
        x=usadas["fecha"], y=usadas["ndvi"], mode="markers", name="Observación válida",
        marker=dict(size=8, color=AZUL, line=dict(color="white", width=2)),
        customdata=usadas["frac_valida"],
        hovertemplate="%{x|%d/%m/%Y}<br>NDVI %{y:.2f}<br>Píxeles válidos %{customdata:.0%}<extra>Válida</extra>"))
    if pd.notna(fila.get("fecha_pico")):
        fig.add_annotation(x=fila["fecha_pico"], y=fila["ndvi_pico"], text=f"Pico {fila['ndvi_pico']:.2f}",
                           showarrow=True, arrowhead=0, ay=-30)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), hovermode="closest",
                      yaxis=dict(title="NDVI", range=[-0.05, 1.0], gridcolor="rgba(128,128,128,0.15)"),
                      xaxis=dict(showgrid=False), legend=dict(orientation="h", y=-0.15))
    return fig


def grafico_estimado_vs_real(oof: pd.DataFrame, seleccion) -> go.Figure:
    fig = go.Figure()
    lim = [0, max(oof["rinde_kg_ha"].max(), oof["rinde_validacion_kg_ha"].max()) * 1.05]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Estimado = real",
                             line=dict(color=GRIS, width=1, dash="dot"), hoverinfo="skip"))
    for cultivo, g in oof.groupby("cultivo"):
        fig.add_trace(go.Scatter(
            x=g["rinde_kg_ha"], y=g["rinde_validacion_kg_ha"], mode="markers", name=cultivo.capitalize(),
            marker=dict(size=8, color=COLOR_CULTIVO.get(cultivo, GRIS), opacity=0.75,
                        line=dict(color="white", width=1)),
            customdata=g[["lote_id", "campania"]],
            hovertemplate="%{customdata[0]} · %{customdata[1]}<br>Real %{x:,.0f}<br>Estimado %{y:,.0f} kg/ha"
                          "<extra></extra>"))
    sel = oof[(oof["lote_id"] == seleccion["lote_id"]) & (oof["campania"] == seleccion["campania"])]
    if not sel.empty:
        fig.add_trace(go.Scatter(x=sel["rinde_kg_ha"], y=sel["rinde_validacion_kg_ha"], mode="markers",
                                 name="Lote seleccionado", hoverinfo="skip",
                                 marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="#0b0b0b", width=2))))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(title="Rinde real (kg/ha)", range=lim, gridcolor="rgba(128,128,128,0.15)"),
                      yaxis=dict(title="Rinde estimado fuera de muestra (kg/ha)", range=lim,
                                 gridcolor="rgba(128,128,128,0.15)"),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def mapa(geometria: str, color: str):
    geo = json.loads(geometria)
    capa = folium.GeoJson(geo, style_function=lambda _: {"color": color, "weight": 3, "fillOpacity": 0.15})
    m = folium.Map(tiles="OpenStreetMap")
    capa.add_to(m)
    m.fit_bounds(capa.get_bounds())
    return m


# ---------------------------------------------------------------------------
st.title("Scoring de riesgo agropecuario · MVP")
st.warning("**Herramienta de apoyo, no una decisión de crédito.** Las estimaciones tienen error "
           "(ver métricas abajo) y dependen de la disponibilidad de imágenes sin nubes.", icon="⚠️")

if not (SALIDA / "scores.csv").exists():
    st.error("No hay resultados en `outputs/`. Corré primero: `agroscore demo`")
    st.stop()

datos = cargar_salidas(str(SALIDA))
cfg = cargar_config()
scores = datos["scores"].copy()
scores["fecha_pico"] = pd.to_datetime(scores["fecha_pico"])
metricas = datos["metricas"]

with st.sidebar:
    st.header("Lote")
    campania = st.selectbox("Campaña", sorted(scores["campania"].unique()), index=len(scores["campania"].unique()) - 1)
    en_campania = scores[scores["campania"] == campania]
    lote_id = st.selectbox("Lote", en_campania["lote_id"].tolist(),
                           format_func=lambda i: f"{i} · {en_campania.set_index('lote_id').loc[i, 'cultivo']}")
    fila = en_campania[en_campania["lote_id"] == lote_id].iloc[0]

    st.header("Parámetros del score")
    st.caption("Cambialos para ver cómo se mueve el score. No modifican el archivo de configuración.")
    precio = st.number_input("Precio de referencia (USD/t)", min_value=1.0, value=float(fila["precio_usd_t"]), step=5.0)
    costo = st.number_input("Costo o deuda (USD/ha)", min_value=1.0, value=float(fila["costo_ref_usd_ha"]), step=10.0)
    k = st.slider("Margen por error del modelo (× RMSE)", 0.0, 2.0, float(cfg["score"]["k_incertidumbre"]), 0.1)

p = ParametrosScore.desde_config(cfg)
p = ParametrosScore(**{**p.__dict__, "k_incertidumbre": k})
rmse = metricas["rmse_por_cultivo"].get(fila["cultivo"])
resultado = calcular_score(fila["rinde_estimado_kg_ha"], rmse, precio, costo, p)

st.subheader(f"Lote {lote_id} · {fila['cultivo'].capitalize()} · campaña {campania} · {fila['area_ha']:.0f} ha")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rinde estimado", f"{fila['rinde_estimado_kg_ha']:,.0f} kg/ha", help=f"Origen: {fila['origen_estimacion']}")
c2.metric("Rinde real", "sin dato" if pd.isna(fila["rinde_kg_ha"]) else f"{fila['rinde_kg_ha']:,.0f} kg/ha",
          delta=None if pd.isna(fila["rinde_kg_ha"])
          else f"{fila['rinde_estimado_kg_ha'] - fila['rinde_kg_ha']:+,.0f} kg/ha estimado − real",
          delta_color="off")
c3.metric("Score", f"{resultado['score']} / 1000")
c4.metric("Riesgo", etiqueta_riesgo(resultado["riesgo"]), help=f"Cobertura ingreso/costo: {resultado['cobertura']:.2f}")
st.info(resultado["explicacion"], icon="🧮")

col_mapa, col_curva = st.columns([2, 3])
with col_mapa:
    st.markdown("**Ubicación del lote**")
    st_folium(mapa(fila["geometria"], ESTADO_RIESGO[resultado["riesgo"]][1]), height=360,
              use_container_width=True, returned_objects=[])
with col_curva:
    st.markdown("**Curva de NDVI de la campaña**")
    clave = lambda df: df[(df["lote_id"] == lote_id) & (df["campania"] == campania)]  # noqa: E731
    st.plotly_chart(grafico_ndvi(clave(datos["series"]), clave(datos["obs"]), fila), width="stretch")

confianza = st.columns(3)
confianza[0].metric("Observaciones válidas", int(fila["n_obs_validas"]))
confianza[1].metric("Mayor hueco sin datos", f"{int(fila['max_hueco_dias'])} días")
confianza[2].metric("Días interpolados en huecos largos", f"{fila['frac_dias_hueco_largo']:.0%}")
if fila["max_hueco_dias"] > cfg["series"]["max_hueco_dias"]:
    st.caption(f"⚠️ Este lote tiene huecos mayores a {cfg['series']['max_hueco_dias']} días: "
               "la curva en esos tramos es interpolada y la estimación es menos confiable.")

st.divider()
st.header("Rendimiento del modelo")
st.caption(f"Validación por campaña (se entrena sin la campaña evaluada). Modelo elegido: **{metricas['tipo']}** · "
           f"{metricas['n_lotes']} lotes con rinde conocido · campañas {', '.join(metricas['campanias'])} · "
           f"entrenado {metricas['fecha_entrenamiento']}")

por_cultivo = metricas["metricas_por_cultivo"][metricas["tipo"]]
baseline = metricas["metricas_por_cultivo"]["baseline"]
cols = st.columns(len(por_cultivo))
for col, (cultivo, m) in zip(cols, por_cultivo.items()):
    with col:
        st.markdown(f"**{cultivo.capitalize()}**")
        a, b, c = st.columns(3)
        a.metric("MAE (kg/ha)", f"{m['mae_kg_ha']:,.0f}",
                 delta=f"{m['mae_kg_ha'] - baseline[cultivo]['mae_kg_ha']:+,.0f} vs. promedio",
                 delta_color="inverse", help="Error medio absoluto. El delta lo compara con predecir el rinde promedio del cultivo")
        b.metric("RMSE (kg/ha)", f"{m['rmse_kg_ha']:,.0f}")
        c.metric("R²", f"{m['r2']:.2f}")

izq, der = st.columns([3, 2])
with izq:
    st.markdown("**Rinde estimado vs. real (fuera de muestra)**")
    st.plotly_chart(grafico_estimado_vs_real(datos["oof"], fila), width="stretch")
with der:
    st.markdown("**Métricas por campaña evaluada**")
    tabla = datos["por_campania"].rename(columns={
        "campania_test": "Campaña", "modelo": "Modelo", "n": "Lotes", "mae_kg_ha": "MAE", "rmse_kg_ha": "RMSE",
        "r2": "R²", "mape_pct": "MAPE %", "sesgo_kg_ha": "Sesgo"})
    st.dataframe(tabla.drop(columns=["Modelo"])[tabla["Modelo"] == metricas["tipo"]].round(2),
                 hide_index=True, width="stretch")
    st.caption("Sesgo positivo = el modelo sobreestima. Un sesgo grande en una campaña extrema "
               "(ej. sequía) indica que el modelo no vio años parecidos al entrenar.")
    with st.expander("Comparación entre modelos"):
        st.dataframe(pd.DataFrame(metricas["metricas_globales"]).T.round(2), width="stretch")

with st.expander("Todos los lotes de la campaña"):
    st.dataframe(en_campania[["lote_id", "cultivo", "rinde_estimado_kg_ha", "rinde_kg_ha", "cobertura", "score",
                              "riesgo"]].round(2), hide_index=True, width="stretch")
