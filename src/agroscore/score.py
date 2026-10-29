"""Conversión del rinde estimado en un score de riesgo explicable.

Lógica (todo por hectárea):
  rinde_pesimista = max(0, rinde_estimado - k * RMSE_validación)
  ingreso         = rinde_pesimista [t/ha] * precio [USD/t]
  cobertura       = ingreso / costo_o_deuda [USD/ha]
  score           = 1000 * (cobertura - c0) / (c1 - c0), recortado a [0, 1000]
  riesgo          = alto si cobertura < u_alto; bajo si cobertura >= u_bajo; medio en el medio

El score es una herramienta de apoyo al análisis, NO una decisión de crédito.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParametrosScore:
    k_incertidumbre: float = 1.0
    cobertura_score_0: float = 0.8
    cobertura_score_1000: float = 1.6
    umbral_riesgo_alto: float = 1.0
    umbral_riesgo_bajo: float = 1.3

    def __post_init__(self):
        if self.cobertura_score_1000 <= self.cobertura_score_0:
            raise ValueError("cobertura_score_1000 debe ser mayor que cobertura_score_0")
        if self.umbral_riesgo_bajo < self.umbral_riesgo_alto:
            raise ValueError("umbral_riesgo_bajo debe ser mayor o igual que umbral_riesgo_alto")
        if self.k_incertidumbre < 0:
            raise ValueError("k_incertidumbre no puede ser negativo")

    @classmethod
    def desde_config(cls, cfg: dict) -> "ParametrosScore":
        s = cfg["score"]
        return cls(**{campo: s[campo] for campo in cls.__dataclass_fields__})


def categoria_riesgo(cobertura: float, p: ParametrosScore) -> str:
    if np.isnan(cobertura):
        return "sin dato"
    if cobertura < p.umbral_riesgo_alto:
        return "alto"
    if cobertura < p.umbral_riesgo_bajo:
        return "medio"
    return "bajo"


def calcular_score(
    rinde_estimado_kg_ha: float,
    rmse_kg_ha: float,
    precio_usd_t: float,
    costo_usd_ha: float,
    p: ParametrosScore = ParametrosScore(),
) -> dict:
    """Score de un lote con su desglose paso a paso."""
    if costo_usd_ha is None or not costo_usd_ha > 0:
        raise ValueError("El costo o deuda de referencia debe ser mayor a 0 USD/ha")
    if precio_usd_t is None or not precio_usd_t > 0:
        raise ValueError("El precio de referencia debe ser mayor a 0 USD/t")
    if rinde_estimado_kg_ha is None or np.isnan(rinde_estimado_kg_ha):
        return {"rinde_estimado_kg_ha": np.nan, "rinde_pesimista_kg_ha": np.nan, "ingreso_usd_ha": np.nan,
                "cobertura": np.nan, "score": np.nan, "riesgo": "sin dato",
                "explicacion": "Sin rinde estimado: faltan datos satelitales o climáticos para este lote."}

    rmse = 0.0 if rmse_kg_ha is None or np.isnan(rmse_kg_ha) else float(rmse_kg_ha)
    pesimista = max(0.0, float(rinde_estimado_kg_ha) - p.k_incertidumbre * rmse)
    ingreso = pesimista / 1000.0 * precio_usd_t
    cobertura = ingreso / costo_usd_ha
    fraccion = (cobertura - p.cobertura_score_0) / (p.cobertura_score_1000 - p.cobertura_score_0)
    score = int(round(1000 * min(1.0, max(0.0, fraccion))))
    riesgo = categoria_riesgo(cobertura, p)
    explicacion = (
        f"Rinde estimado {rinde_estimado_kg_ha:,.0f} kg/ha − {p.k_incertidumbre:g} × error típico del modelo "
        f"({rmse:,.0f} kg/ha) = rinde pesimista {pesimista:,.0f} kg/ha. "
        f"Ingreso = {pesimista / 1000:,.2f} t/ha × {precio_usd_t:,.0f} USD/t = {ingreso:,.0f} USD/ha. "
        f"Cobertura = {ingreso:,.0f} / {costo_usd_ha:,.0f} USD/ha = {cobertura:.2f}. "
        f"Score {score}/1000 (0 con cobertura ≤ {p.cobertura_score_0:g}, 1000 con ≥ {p.cobertura_score_1000:g}). "
        f"Riesgo {riesgo} (alto < {p.umbral_riesgo_alto:g} ≤ medio < {p.umbral_riesgo_bajo:g} ≤ bajo)."
    )
    return {
        "rinde_estimado_kg_ha": float(rinde_estimado_kg_ha),
        "rinde_pesimista_kg_ha": pesimista,
        "ingreso_usd_ha": ingreso,
        "cobertura": cobertura,
        "score": score,
        "riesgo": riesgo,
        "explicacion": explicacion,
    }


def score_dataframe(df: pd.DataFrame, rmse_por_cultivo: dict, cfg: dict, p: ParametrosScore | None = None) -> pd.DataFrame:
    """Aplica calcular_score a cada fila (necesita cultivo, rinde_estimado_kg_ha y opcionalmente costo_usd_ha)."""
    p = p or ParametrosScore.desde_config(cfg)
    filas = []
    for _, f in df.iterrows():
        precio = cfg["score"]["precio_usd_t"][f["cultivo"]]
        costo = f.get("costo_usd_ha")
        if costo is None or pd.isna(costo):
            costo = cfg["score"]["costo_usd_ha"][f["cultivo"]]
        r = calcular_score(f["rinde_estimado_kg_ha"], rmse_por_cultivo.get(f["cultivo"], np.nan), precio, costo, p)
        r.pop("rinde_estimado_kg_ha")
        filas.append(r | {"precio_usd_t": precio, "costo_ref_usd_ha": costo})
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(filas)], axis=1)
