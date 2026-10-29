"""Clima diario por lote: NASA POWER (real) o archivo CSV (sintético).

Todos los proveedores devuelven: fecha, lluvia_mm, tmax, tmin, radiacion_mj.
"""

import logging
import time

import numpy as np
import pandas as pd
import requests

from agroscore import cache

log = logging.getLogger(__name__)

COLUMNAS = ["fecha", "lluvia_mm", "tmax", "tmin", "radiacion_mj"]
URL_POWER = "https://power.larc.nasa.gov/api/temporal/daily/point"
# Comunidad AG: radiación en MJ/m²/día. PRECTOTCORR = precipitación corregida (mm/día).
PARAMETROS_POWER = {
    "PRECTOTCORR": "lluvia_mm",
    "T2M_MAX": "tmax",
    "T2M_MIN": "tmin",
    "ALLSKY_SFC_SW_DWN": "radiacion_mj",
}
VALOR_FALTANTE = -999


class ProveedorClimaArchivo:
    """Clima precalculado en un CSV con lote_id, campania y las COLUMNAS."""

    def __init__(self, ruta_csv):
        self.datos = pd.read_csv(ruta_csv, dtype={"lote_id": str, "campania": str}, parse_dates=["fecha"])

    def clima(self, fila, inicio, fin) -> pd.DataFrame:
        d = self.datos[(self.datos["lote_id"] == fila["lote_id"]) & (self.datos["campania"] == fila["campania"])]
        return d[(d["fecha"] >= inicio) & (d["fecha"] <= fin)].reindex(columns=COLUMNAS).reset_index(drop=True)


class ProveedorNasaPower:
    """NASA POWER, endpoint diario por punto (centroide del lote). No requiere clave.

    La resolución de POWER es de ~0,5° x 0,625°: lotes cercanos comparten la misma celda,
    por eso la clave de caché redondea las coordenadas a 0,01°.
    Se espera `pausa_s` entre consultas reales para respetar el uso equitativo del servicio.
    """

    def __init__(self, pausa_s: float = 1.0, reintentos: int = 3, timeout_s: int = 60):
        self.pausa_s = pausa_s
        self.reintentos = reintentos
        self.timeout_s = timeout_s

    def clima(self, fila, inicio, fin) -> pd.DataFrame:
        lat, lon = round(float(fila["lat"]), 2), round(float(fila["lon"]), 2)
        k = cache.clave("nasa_power", lat, lon, str(inicio.date()), str(fin.date()), sorted(PARAMETROS_POWER))
        datos = cache.leer_json("nasa_power", k)
        if datos is None:
            datos = self._consultar(lat, lon, inicio, fin)
            cache.guardar_json("nasa_power", k, datos)
        return _a_dataframe(datos)

    def _consultar(self, lat, lon, inicio, fin) -> dict:
        params = {
            "parameters": ",".join(PARAMETROS_POWER),
            "community": "AG",
            "latitude": lat,
            "longitude": lon,
            "start": f"{inicio:%Y%m%d}",
            "end": f"{fin:%Y%m%d}",
            "format": "JSON",
        }
        for intento in range(1, self.reintentos + 1):
            try:
                r = requests.get(URL_POWER, params=params, timeout=self.timeout_s)
                r.raise_for_status()
                time.sleep(self.pausa_s)
                return r.json()["properties"]["parameter"]
            except (requests.RequestException, KeyError, ValueError) as e:
                log.warning("NASA POWER falló (intento %d/%d): %s", intento, self.reintentos, e)
                if intento == self.reintentos:
                    raise
                time.sleep(2 ** intento)


def _a_dataframe(parametros: dict) -> pd.DataFrame:
    df = pd.DataFrame({PARAMETROS_POWER[p]: pd.Series(v) for p, v in parametros.items() if p in PARAMETROS_POWER})
    df = df.replace(VALOR_FALTANTE, np.nan)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "fecha"
    return df.reset_index().reindex(columns=COLUMNAS)


def crear_proveedor_clima(nombre: str, ruta_clima=None):
    if nombre == "archivo":
        if ruta_clima is None:
            raise ValueError("La fuente de clima 'archivo' necesita --clima con el CSV")
        return ProveedorClimaArchivo(ruta_clima)
    if nombre == "nasa_power":
        return ProveedorNasaPower()
    raise ValueError(f"Fuente de clima desconocida: {nombre}")
