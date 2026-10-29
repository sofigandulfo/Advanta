"""Carga de la configuración YAML."""

from pathlib import Path

import yaml

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
CONFIG_DEFECTO = RAIZ_PROYECTO / "config" / "config.yaml"


def cargar_config(ruta: str | Path | None = None) -> dict:
    with open(ruta or CONFIG_DEFECTO, encoding="utf-8") as f:
        return yaml.safe_load(f)
