"""Variables por lote a partir de la curva de NDVI y del clima de la campaña."""

import numpy as np
import pandas as pd

VARIABLES_CURVA = [
    "ndvi_pico",
    "dias_al_pico",
    "auc_ndvi",
    "pendiente_crecimiento",
    "caida_pre_pico",
    "caida_post_pico_30d",
    "n_obs_validas",
    "max_hueco_dias",
    "frac_dias_hueco_largo",
]
VARIABLES_CLIMA = [
    "lluvia_total_mm",
    "lluvia_critico_mm",
    "tmax_media",
    "dias_calor_critico",
    "radiacion_total_mj",
    "gdd",
]


def features_curva(serie: pd.DataFrame, inicio, ndvi_base: float = 0.2, ventana_pendiente: int = 10) -> dict:
    """Variables de la curva suavizada (columna 'suave', índice diario).

    - ndvi_pico: máximo de la curva suavizada.
    - dias_al_pico: días desde el inicio de la ventana (o siembra - 15) hasta el pico.
    - auc_ndvi: área bajo la curva por encima de ndvi_base (NDVI·día). Resume biomasa acumulada.
    - pendiente_crecimiento: máxima pendiente (NDVI/día) en ventanas de N días antes del pico.
    - caida_pre_pico: máxima caída respecto del máximo previo, antes del pico (estrés durante el crecimiento).
    - caida_post_pico_30d: NDVI(pico) - NDVI(pico + 30 días): senescencia rápida o estrés en llenado.
    """
    s = serie["suave"].dropna()
    if len(s) < 3:
        return {k: np.nan for k in ["ndvi_pico", "dias_al_pico", "auc_ndvi", "pendiente_crecimiento",
                                    "caida_pre_pico", "caida_post_pico_30d"]} | {"fecha_pico": pd.NaT}
    fecha_pico = s.idxmax()
    v = s.to_numpy()
    i_pico = int(np.argmax(v))

    crecimiento = v[: i_pico + 1]
    if len(crecimiento) > ventana_pendiente:
        pendiente = float(np.max((crecimiento[ventana_pendiente:] - crecimiento[:-ventana_pendiente]) / ventana_pendiente))
    else:
        pendiente = np.nan

    caida_pre = float(np.max(np.maximum.accumulate(crecimiento) - crecimiento))
    fecha_post = fecha_pico + pd.Timedelta(days=30)
    caida_post = float(v[i_pico] - s.loc[fecha_post]) if fecha_post in s.index else np.nan

    return {
        "ndvi_pico": float(v[i_pico]),
        "fecha_pico": fecha_pico,
        "dias_al_pico": int((fecha_pico - pd.Timestamp(inicio).normalize()).days),
        "auc_ndvi": float(np.clip(v - ndvi_base, 0, None).sum()),  # paso diario: suma = integral
        "pendiente_crecimiento": pendiente,
        "caida_pre_pico": caida_pre,
        "caida_post_pico_30d": caida_post,
    }


def features_clima(clima: pd.DataFrame, inicio, fin, fecha_pico, cfg: dict) -> dict:
    """Variables climáticas de la campaña.

    clima: DataFrame con fecha, lluvia_mm, tmax, tmin, radiacion_mj (diario).
    El período crítico se define relativo al pico de NDVI (config: periodo_critico).
    """
    c = clima.copy()
    c["fecha"] = pd.to_datetime(c["fecha"])
    camp = c[(c["fecha"] >= pd.Timestamp(inicio)) & (c["fecha"] <= pd.Timestamp(fin))]
    if camp.empty:
        return {k: np.nan for k in VARIABLES_CLIMA}

    pc = cfg["periodo_critico"]
    if pd.notna(fecha_pico):
        desde = fecha_pico - pd.Timedelta(days=pc["dias_antes_pico"])
        hasta = fecha_pico + pd.Timedelta(days=pc["dias_despues_pico"])
        critico = camp[(camp["fecha"] >= desde) & (camp["fecha"] <= hasta)]
    else:
        critico = camp.iloc[0:0]

    t_base = cfg["clima"]["temp_base_gdd"]
    t_media = (camp["tmax"] + camp["tmin"]) / 2
    return {
        "lluvia_total_mm": float(camp["lluvia_mm"].sum()),
        "lluvia_critico_mm": float(critico["lluvia_mm"].sum()) if len(critico) else np.nan,
        "tmax_media": float(camp["tmax"].mean()),
        "dias_calor_critico": int((critico["tmax"] > cfg["clima"]["temp_estres_calor"]).sum()) if len(critico) else np.nan,
        "radiacion_total_mj": float(camp["radiacion_mj"].sum()),
        "gdd": float(np.clip(t_media - t_base, 0, None).sum()),
    }
