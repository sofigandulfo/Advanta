"""Caché en disco para no repetir consultas a servicios externos.

La clave es un hash de los parámetros de la consulta (geometría, fechas, umbrales),
de modo que cambiar cualquier parámetro genera una consulta nueva.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from agroscore.config import RAIZ_PROYECTO

DIR_CACHE = RAIZ_PROYECTO / "data" / "cache"


def clave(*partes) -> str:
    texto = json.dumps(partes, sort_keys=True, default=str)
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()[:20]


def ruta(espacio: str, k: str, extension: str, dir_cache: Path | None = None) -> Path:
    base = Path(dir_cache or DIR_CACHE) / espacio
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{k}.{extension}"


def leer_json(espacio: str, k: str, dir_cache: Path | None = None):
    p = ruta(espacio, k, "json", dir_cache)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def guardar_json(espacio: str, k: str, datos, dir_cache: Path | None = None) -> None:
    ruta(espacio, k, "json", dir_cache).write_text(json.dumps(datos), encoding="utf-8")


def leer_csv(espacio: str, k: str, dir_cache: Path | None = None, **kwargs) -> pd.DataFrame | None:
    p = ruta(espacio, k, "csv", dir_cache)
    if p.exists():
        return pd.read_csv(p, **kwargs)
    return None


def guardar_csv(espacio: str, k: str, df: pd.DataFrame, dir_cache: Path | None = None) -> None:
    df.to_csv(ruta(espacio, k, "csv", dir_cache), index=False)
