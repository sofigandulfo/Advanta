"""Observaciones Sentinel-2 L2A por lote: NDVI, EVI y NDWI con máscara de nubes SCL.

Proveedores (todos devuelven un DataFrame con fecha, ndvi, evi, ndwi, frac_valida, n_pixeles):
  - ProveedorArchivo: lee observaciones ya calculadas de un CSV (dataset sintético u otra fuente).
  - ProveedorPlanetary: Microsoft Planetary Computer (STAC). Opción principal.
  - ProveedorCopernicus: Copernicus Data Space. Alternativa, solo documentada en el MVP.

Por cada lote se guarda en caché la serie ya resumida (una fila por fecha, no los rásters),
así que una segunda corrida con los mismos parámetros no vuelve a descargar nada.
"""

import logging

import numpy as np
import pandas as pd
from shapely.geometry import mapping

from agroscore import cache
from agroscore.indices import evi, ndvi, ndwi, reflectancia

log = logging.getLogger(__name__)

COLUMNAS = ["fecha", "ndvi", "evi", "ndwi", "frac_valida", "n_pixeles"]
STAC_PLANETARY = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLECCION_S2 = "sentinel-2-l2a"
BANDAS = ["B02", "B03", "B04", "B08", "B11", "SCL"]


class ProveedorArchivo:
    """Observaciones precalculadas en un CSV con lote_id, campania y las COLUMNAS."""

    def __init__(self, ruta_csv):
        self.obs = pd.read_csv(ruta_csv, dtype={"lote_id": str, "campania": str}, parse_dates=["fecha"])

    def observaciones(self, fila, inicio, fin) -> pd.DataFrame:
        o = self.obs[(self.obs["lote_id"] == fila["lote_id"]) & (self.obs["campania"] == fila["campania"])]
        o = o[(o["fecha"] >= inicio) & (o["fecha"] <= fin)]
        return o.reindex(columns=COLUMNAS).reset_index(drop=True)


class ProveedorPlanetary:
    """Sentinel-2 L2A desde Microsoft Planetary Computer.

    Requiere los extras: pip install -e ".[satelital]".
    Acceso anónimo: planetary_computer firma las URLs con un token SAS temporal.
    Con la variable de entorno PC_SDK_SUBSCRIPTION_KEY se obtienen límites más amplios.
    """

    def __init__(self, cfg: dict):
        n = cfg["nubes"]
        self.max_nubosidad = n["max_nubosidad_escena"]
        self.scl_validas = list(n["scl_validas"])

    def observaciones(self, fila, inicio, fin) -> pd.DataFrame:
        geom = fila["geometry"]
        k = cache.clave("planetary", COLECCION_S2, geom.wkt, str(inicio.date()), str(fin.date()),
                        self.max_nubosidad, self.scl_validas)
        en_cache = cache.leer_csv("sentinel2", k, parse_dates=["fecha"])
        if en_cache is not None:
            return en_cache

        items = self._buscar(geom, inicio, fin)
        log.info("Lote %s %s: %d escenas", fila["lote_id"], fila["campania"], len(items))
        obs = self._resumir(items, geom) if items else pd.DataFrame(columns=COLUMNAS)
        cache.guardar_csv("sentinel2", k, obs)
        return obs

    def _buscar(self, geom, inicio, fin):
        import planetary_computer
        import pystac_client

        catalogo = pystac_client.Client.open(STAC_PLANETARY, modifier=planetary_computer.sign_inplace)
        busqueda = catalogo.search(
            collections=[COLECCION_S2],
            intersects=mapping(geom),
            datetime=f"{inicio:%Y-%m-%d}/{fin:%Y-%m-%d}",
            query={"eo:cloud_cover": {"lt": self.max_nubosidad}},
        )
        return list(busqueda.items())

    def _resumir(self, items, geom) -> pd.DataFrame:
        import odc.stac
        from odc.geo.geom import Geometry
        from odc.geo.xr import rasterize

        poligono = Geometry(mapping(geom), crs="EPSG:4326")
        ds = odc.stac.load(items, bands=BANDAS, geopolygon=poligono, crs="utm", resolution=10,
                           groupby="solar_day", resampling="nearest")
        dentro = rasterize(poligono, ds.odc.geobox).values
        # El baseline de procesamiento define el offset radiométrico (ver indices.reflectancia).
        baselines = {pd.Timestamp(it.datetime).tz_localize(None).normalize(): it.properties.get("s2:processing_baseline")
                     for it in items}

        filas = []
        for i, t in enumerate(ds.time.values):
            fecha = pd.Timestamp(t).normalize()
            scl = ds["SCL"].isel(time=i).values
            valido = dentro & np.isin(scl, self.scl_validas)
            frac = float(valido.sum() / max(dentro.sum(), 1))
            fila = {"fecha": fecha, "frac_valida": frac, "n_pixeles": int(valido.sum())}
            if valido.sum() > 0:
                b = baselines.get(fecha)
                r = {banda: reflectancia(ds[banda].isel(time=i).values[valido], b) for banda in BANDAS[:-1]}
                fila |= {
                    "ndvi": float(np.nanmedian(ndvi(r["B08"], r["B04"]))),
                    "evi": float(np.nanmedian(evi(r["B08"], r["B04"], r["B02"]))),
                    "ndwi": float(np.nanmedian(ndwi(r["B08"], r["B11"]))),
                }
            filas.append(fila)
        return pd.DataFrame(filas).reindex(columns=COLUMNAS)


class ProveedorCopernicus:
    """Alternativa: Copernicus Data Space Ecosystem (no implementada en el MVP).

    Pasos para implementarla si Planetary Computer deja de estar disponible:
      1. Crear una cuenta gratuita en dataspace.copernicus.eu y generar claves S3
         (variables de entorno CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY, nunca en el repo).
      2. Buscar con pystac-client en el STAC de CDSE (al momento: https://stac.dataspace.copernicus.eu/v1,
         colección sentinel-2-l2a; verificar la URL vigente en documentation.dataspace.copernicus.eu).
      3. Leer los assets desde s3://eodata con endpoint https://eodata.dataspace.copernicus.eu
         (rasterio/GDAL con AWS_S3_ENDPOINT y AWS_VIRTUAL_HOSTING=FALSE).
      4. Reutilizar ProveedorPlanetary._resumir: la máscara SCL y los índices son iguales.
    Respetar las cuotas de la cuenta gratuita (ver README).
    """

    def __init__(self, cfg: dict):
        pass

    def observaciones(self, fila, inicio, fin) -> pd.DataFrame:
        raise NotImplementedError(
            "El proveedor Copernicus Data Space no está implementado en el MVP. "
            "Ver la docstring de ProveedorCopernicus para los pasos."
        )


def crear_proveedor(nombre: str, cfg: dict, ruta_obs=None):
    if nombre == "archivo":
        if ruta_obs is None:
            raise ValueError("La fuente 'archivo' necesita --obs con el CSV de observaciones")
        return ProveedorArchivo(ruta_obs)
    if nombre == "planetary":
        return ProveedorPlanetary(cfg)
    if nombre == "copernicus":
        return ProveedorCopernicus(cfg)
    raise ValueError(f"Fuente de imágenes desconocida: {nombre}")
