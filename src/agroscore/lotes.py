"""Lectura y validación del CSV de lotes.

Columnas obligatorias: lote_id, cultivo, campania (formato "2022/23") y una geometría:
  - geometria: GeoJSON (Polygon, MultiPolygon o Feature) en EPSG:4326, o
  - lat, lon (+ buffer_m opcional, por defecto 100 m): se genera un cuadrado alrededor del punto.
Opcionales: rinde_kg_ha, fecha_siembra (AAAA-MM-DD), costo_usd_ha.
"""

import json
import re

import numpy as np
import pandas as pd
import shapely
from pyproj import Geod, Transformer
from shapely.geometry import box, shape

OBLIGATORIAS = ["lote_id", "cultivo", "campania"]
PATRON_CAMPANIA = re.compile(r"^(\d{4})/(\d{2})$")
_GEOD = Geod(ellps="WGS84")


def anio_inicio(campania: str) -> int:
    m = PATRON_CAMPANIA.match(str(campania).strip())
    if not m:
        raise ValueError(f"Campaña inválida '{campania}': se espera el formato AAAA/AA, ej. 2022/23")
    inicio = int(m.group(1))
    if (inicio + 1) % 100 != int(m.group(2)):
        raise ValueError(f"Campaña inválida '{campania}': los años no son consecutivos")
    return inicio


def _geometria_desde_geojson(texto: str):
    obj = json.loads(texto)
    if obj.get("type") == "Feature":
        obj = obj["geometry"]
    geom = shape(obj)
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Se espera Polygon o MultiPolygon, llegó {geom.geom_type}")
    return geom


def rectangulo_metros(lat: float, lon: float, ancho_m: float, alto_m: float):
    """Rectángulo centrado en el punto, construido en una proyección local en metros y devuelto en EPSG:4326."""
    local = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +units=m"
    a_wgs84 = Transformer.from_crs(local, "EPSG:4326", always_xy=True)
    rect = box(-ancho_m / 2, -alto_m / 2, ancho_m / 2, alto_m / 2)
    return shapely.transform(rect, lambda x, y: a_wgs84.transform(x, y), interleaved=False)


def area_ha(geom) -> float:
    area_m2, _ = _GEOD.geometry_area_perimeter(geom)
    return abs(area_m2) / 10000.0


def leer_lotes(ruta) -> pd.DataFrame:
    df = pd.read_csv(ruta, dtype={"lote_id": str, "campania": str})
    faltan = [c for c in OBLIGATORIAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias en {ruta}: {faltan}")
    tiene_geojson = "geometria" in df.columns
    tiene_punto = {"lat", "lon"} <= set(df.columns)
    if not (tiene_geojson or tiene_punto):
        raise ValueError("El CSV necesita una columna 'geometria' (GeoJSON) o columnas 'lat' y 'lon'")

    df["cultivo"] = df["cultivo"].str.strip().str.lower()
    df["campania"] = df["campania"].str.strip()
    df["anio_inicio"] = df["campania"].map(anio_inicio)

    geoms = []
    for _, fila in df.iterrows():
        if tiene_geojson and isinstance(fila.get("geometria"), str) and fila["geometria"].strip():
            geoms.append(_geometria_desde_geojson(fila["geometria"]))
        elif tiene_punto and pd.notna(fila.get("lat")) and pd.notna(fila.get("lon")):
            buffer_m = fila.get("buffer_m", 100)
            buffer_m = 100 if pd.isna(buffer_m) else float(buffer_m)
            geoms.append(rectangulo_metros(float(fila["lat"]), float(fila["lon"]), 2 * buffer_m, 2 * buffer_m))
        else:
            raise ValueError(f"El lote {fila['lote_id']} ({fila['campania']}) no tiene geometría")
    df["geometry"] = geoms
    df["area_ha"] = [area_ha(g) for g in geoms]
    df["lat"] = [g.centroid.y for g in geoms]
    df["lon"] = [g.centroid.x for g in geoms]

    df["rinde_kg_ha"] = pd.to_numeric(df["rinde_kg_ha"], errors="coerce") if "rinde_kg_ha" in df else np.nan
    df["fecha_siembra"] = pd.to_datetime(df["fecha_siembra"], errors="coerce") if "fecha_siembra" in df else pd.NaT
    df["costo_usd_ha"] = pd.to_numeric(df["costo_usd_ha"], errors="coerce") if "costo_usd_ha" in df else np.nan

    duplicados = df.duplicated(["lote_id", "campania"])
    if duplicados.any():
        raise ValueError(f"Hay lotes repetidos en la misma campaña: {df.loc[duplicados, 'lote_id'].tolist()}")
    return df


def ventana_campania(fila, cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Fechas de inicio y fin de la campaña para un lote."""
    params = cfg["cultivos"].get(fila["cultivo"])
    if params is None:
        raise ValueError(f"Cultivo '{fila['cultivo']}' sin parámetros en config.yaml (sección cultivos)")
    if pd.notna(fila.get("fecha_siembra")):
        siembra = pd.Timestamp(fila["fecha_siembra"])
        return siembra - pd.Timedelta(days=15), siembra + pd.Timedelta(days=params["duracion_dias"])
    anio = int(fila["anio_inicio"])
    return (
        pd.Timestamp(f"{anio}-{params['inicio']}"),
        pd.Timestamp(f"{anio + 1}-{params['fin']}"),
    )

