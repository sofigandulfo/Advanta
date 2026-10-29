"""Limpieza, interpolación y suavizado de series de índices.

Procedimiento (documentado también en el README):
1. Se descartan fechas con fracción de píxeles válidos (según SCL) menor al umbral.
   Si quedan dos observaciones el mismo día, se promedian.
2. Se arma una serie diaria entre inicio y fin de campaña y se interpola linealmente
   entre observaciones válidas. Antes de la primera y después de la última observación
   no se extrapola: esos días quedan como NaN.
3. Los días dentro de un hueco entre observaciones mayor a max_hueco_dias se interpolan
   igual pero se marcan con hueco_largo=True; las variables de confianza los cuentan.
4. Se suaviza con Savitzky-Golay (conserva mejor la altura del pico que una media móvil).
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def filtrar_validas(obs: pd.DataFrame, min_fraccion: float, columna: str = "ndvi") -> pd.DataFrame:
    """Devuelve las observaciones utilizables, una por día."""
    ok = obs[(obs["frac_valida"] >= min_fraccion) & obs[columna].notna()].copy()
    ok["fecha"] = pd.to_datetime(ok["fecha"]).dt.normalize()
    return ok.groupby("fecha", as_index=False)[columna].mean().sort_values("fecha")


def interpolar_diaria(
    fechas,
    valores,
    inicio,
    fin,
    max_hueco_dias: int = 30,
    savgol_ventana: int = 21,
    savgol_orden: int = 2,
) -> pd.DataFrame:
    """Serie diaria con columnas: obs, interp, suave, hueco_largo."""
    dias = pd.date_range(pd.Timestamp(inicio).normalize(), pd.Timestamp(fin).normalize(), freq="D")
    serie = pd.Series(np.asarray(valores, dtype="float64"), index=pd.DatetimeIndex(pd.to_datetime(fechas)).normalize())
    serie = serie[(serie.index >= dias[0]) & (serie.index <= dias[-1])]
    serie = serie.groupby(level=0).mean()

    out = pd.DataFrame(index=dias)
    out.index.name = "fecha"
    out["obs"] = serie.reindex(dias)
    out["interp"] = out["obs"].interpolate(method="linear", limit_area="inside")
    out["hueco_largo"] = _marcar_huecos_largos(dias, serie.index, max_hueco_dias)
    out["suave"] = _suavizar(out["interp"].to_numpy(), savgol_ventana, savgol_orden)
    return out


def _marcar_huecos_largos(dias: pd.DatetimeIndex, fechas_obs: pd.DatetimeIndex, max_hueco: int) -> np.ndarray:
    marca = np.zeros(len(dias), dtype=bool)
    fechas_obs = fechas_obs.sort_values()
    for a, b in zip(fechas_obs[:-1], fechas_obs[1:]):
        if (b - a).days > max_hueco:
            marca |= (dias > a) & (dias < b)
    return marca


def _suavizar(valores: np.ndarray, ventana: int, orden: int) -> np.ndarray:
    """Savitzky-Golay sobre el tramo sin NaN; los extremos sin dato quedan NaN."""
    out = np.full_like(valores, np.nan)
    validos = np.flatnonzero(~np.isnan(valores))
    if len(validos) == 0:
        return out
    a, b = validos[0], validos[-1] + 1
    tramo = valores[a:b]
    ventana = min(ventana, len(tramo) if len(tramo) % 2 == 1 else len(tramo) - 1)
    if ventana <= orden:
        out[a:b] = tramo
        return out
    out[a:b] = savgol_filter(tramo, window_length=ventana, polyorder=orden)
    return out


def max_hueco(fechas_obs, inicio, fin) -> int:
    """Hueco más largo (días) entre observaciones válidas, incluyendo los bordes de la campaña."""
    puntos = sorted({pd.Timestamp(inicio).normalize(), pd.Timestamp(fin).normalize(), *pd.DatetimeIndex(pd.to_datetime(fechas_obs)).normalize()})
    return int(max((b - a).days for a, b in zip(puntos[:-1], puntos[1:])))
