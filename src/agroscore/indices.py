"""Índices espectrales sobre reflectancias Sentinel-2 (escala 0-1).

Bandas Sentinel-2 usadas: B02 azul, B03 verde, B04 rojo, B08 NIR, B11 SWIR 1.6 µm.
"""

import numpy as np

# Desde el processing baseline 04.00 (enero 2022) los productos L2A traen un
# offset de -1000 en los números digitales: reflectancia = (DN - 1000) / 10000.
OFFSET_DN_BASELINE_04 = -1000
ESCALA = 10000.0


def reflectancia(dn, processing_baseline: str | float | None = None):
    """Convierte números digitales L2A a reflectancia, aplicando el offset según el baseline.

    DN = 0 es "sin dato" y se devuelve como NaN.
    """
    dn = np.asarray(dn, dtype="float64")
    offset = 0.0
    if processing_baseline is not None and float(processing_baseline) >= 4.0:
        offset = OFFSET_DN_BASELINE_04
    refl = (dn + offset) / ESCALA
    return np.where(dn == 0, np.nan, refl)


def _cociente_normalizado(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    suma = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(suma == 0, np.nan, (a - b) / suma)


def ndvi(nir, rojo):
    """NDVI = (NIR - Rojo) / (NIR + Rojo)."""
    return _cociente_normalizado(nir, rojo)


def evi(nir, rojo, azul, g=2.5, c1=6.0, c2=7.5, l=1.0):
    """EVI = G * (NIR - Rojo) / (NIR + C1*Rojo - C2*Azul + L). Menos saturación que NDVI en canopeos densos."""
    nir = np.asarray(nir, dtype="float64")
    rojo = np.asarray(rojo, dtype="float64")
    azul = np.asarray(azul, dtype="float64")
    denom = nir + c1 * rojo - c2 * azul + l
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom == 0, np.nan, g * (nir - rojo) / denom)


def ndwi(nir, swir):
    """NDWI (Gao, 1996) = (NIR - SWIR) / (NIR + SWIR). Sensible al contenido de agua del canopeo.

    Nota: no confundir con el NDWI de McFeeters (verde/NIR), que mide agua libre.
    """
    return _cociente_normalizado(nir, swir)
