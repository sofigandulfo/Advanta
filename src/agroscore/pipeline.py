"""Orquestación: lotes -> observaciones -> series -> variables."""

import json
import logging

import numpy as np
import pandas as pd
from shapely.geometry import mapping

from agroscore.features import VARIABLES_CLIMA, VARIABLES_CURVA, features_clima, features_curva
from agroscore.lotes import ventana_campania
from agroscore.timeseries import filtrar_validas, interpolar_diaria, max_hueco

log = logging.getLogger(__name__)

COLUMNAS_LOTE = ["lote_id", "campania", "cultivo", "area_ha", "lat", "lon", "fecha_siembra",
                 "rinde_kg_ha", "costo_usd_ha"]


def procesar_lote(fila, prov_img, prov_clima, cfg) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    inicio, fin = ventana_campania(fila, cfg)
    obs = prov_img.observaciones(fila, inicio, fin)
    min_frac = cfg["nubes"]["min_fraccion_valida"]
    validas = filtrar_validas(obs, min_frac)

    s = cfg["series"]
    serie = interpolar_diaria(validas["fecha"], validas["ndvi"], inicio, fin, s["max_hueco_dias"],
                              s["savgol_ventana_dias"], s["savgol_orden"])
    curva = features_curva(serie, inicio, s["ndvi_base"])
    clima = prov_clima.clima(fila, inicio, fin)
    feats = curva | features_clima(clima, inicio, fin, curva["fecha_pico"], cfg) | {
        "n_obs_validas": int(len(validas)),
        "max_hueco_dias": max_hueco(validas["fecha"], inicio, fin),
        "frac_dias_hueco_largo": float(serie["hueco_largo"].mean()),
        "inicio": inicio,
        "fin": fin,
    }
    obs = obs.assign(usada=(obs["frac_valida"] >= min_frac) & obs["ndvi"].notna())
    return feats, serie.reset_index(), obs


def procesar(lotes: pd.DataFrame, prov_img, prov_clima, cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve (features, series diarias, observaciones crudas) para todos los lotes.

    Un error en un lote no corta el proceso: queda registrado en la columna 'error'.
    """
    filas, series, observaciones = [], [], []
    for _, fila in lotes.iterrows():
        base = {c: fila.get(c) for c in COLUMNAS_LOTE} | {"geometria": json.dumps(mapping(fila["geometry"]))}
        clave = {"lote_id": fila["lote_id"], "campania": fila["campania"]}
        try:
            feats, serie, obs = procesar_lote(fila, prov_img, prov_clima, cfg)
            filas.append(base | feats | {"error": ""})
            series.append(serie.assign(**clave))
            observaciones.append(obs.assign(**clave))
        except Exception as e:  # noqa: BLE001 - se registra y se sigue con el próximo lote
            log.error("Lote %s %s: %s", fila["lote_id"], fila["campania"], e)
            vacias = {k: np.nan for k in VARIABLES_CURVA + VARIABLES_CLIMA}
            filas.append(base | vacias | {"error": str(e)})
    juntar = lambda partes: pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()  # noqa: E731
    return pd.DataFrame(filas), juntar(series), juntar(observaciones)
