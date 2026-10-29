"""Generador de un dataset sintético de ejemplo (lotes, observaciones satelitales y clima).

Sirve para probar el circuito completo sin conexión ni datos reales. NO sirve para medir
la calidad del modelo: el rinde se genera a partir de las mismas variables que el modelo
observa, así que las métricas sobre este dataset salen buenas por construcción.

Supuestos del generador (simplificados a propósito):
- Lotes en la región pampeana, rotación soja/maíz, 4 campañas con distinta lluvia
  (2022/23 muy seca, imitando la sequía real de esa campaña).
- Rinde = potencial del cultivo × vigor (suelo, agua temprana) × estrés (agua y calor
  en el período crítico) × factor de zona × ruido.
- Curva de NDVI doble logística; el estrés adelanta la senescencia y baja el pico.
- Revisita cada 5 días; ~35 % de fechas nubladas (NDVI sesgado hacia abajo y baja
  fracción válida) y algunos lotes con períodos nublados largos.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import mapping

from agroscore.lotes import rectangulo_metros

# zona: (lat, lon, factor de productividad)
ZONAS = {
    "Pergamino": (-33.89, -60.57, 1.00),
    "Junín": (-34.58, -60.95, 0.97),
    "Marcos Juárez": (-32.70, -62.10, 0.98),
    "Venado Tuerto": (-33.75, -61.97, 0.99),
    "Rojas": (-34.20, -60.73, 0.98),
    "9 de Julio": (-35.44, -60.88, 0.92),
    "Río Cuarto": (-33.12, -64.35, 0.88),
}
# Factor de lluvia por campaña (1 = normal).
CAMPANIAS = {"2019/20": 1.0, "2020/21": 0.85, "2021/22": 0.8, "2022/23": 0.5}
POTENCIAL_KG_HA = {"soja": 4200, "maiz": 11000}
SIEMBRA_MM_DD = {"soja": "11-10", "maiz": "10-01"}
DIAS_AL_PICO = {"soja": 80, "maiz": 85}
DURACION_DIAS = {"soja": 170, "maiz": 190}

PROB_LLUVIA = 0.22
LLUVIA_MEDIA_MM = 13.0


def _clima(rng, fechas, factor_lluvia):
    n = len(fechas)
    # Estacionalidad con máximo térmico a mediados de enero (día del año ~15).
    estacion = np.cos(2 * np.pi * (fechas.dayofyear.to_numpy() - 15) / 365.25)
    llueve = rng.random(n) < PROB_LLUVIA
    lluvia = np.where(llueve, rng.exponential(LLUVIA_MEDIA_MM * factor_lluvia, n), 0.0)
    tmax = 26 + 5 * estacion + rng.normal(0, 3, n) + (1 - factor_lluvia) * 4
    tmin = tmax - 12 + rng.normal(0, 2, n)
    rad = np.clip(17 + 8 * estacion + rng.normal(0, 3, n), 3, None)
    return pd.DataFrame({"fecha": fechas, "lluvia_mm": lluvia.round(1), "tmax": tmax.round(1),
                         "tmin": tmin.round(1), "radiacion_mj": rad.round(1)})


def _doble_logistica(t, base, amplitud, t_sube, t_baja, k_sube=8.0, k_baja=7.0):
    return base + amplitud * (1 / (1 + np.exp(-(t - t_sube) / k_sube)) - 1 / (1 + np.exp(-(t - t_baja) / k_baja)))


def generar(n_lotes: int = 60, semilla: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve (lotes, observaciones, clima)."""
    rng = np.random.default_rng(semilla)
    nombres_zonas = list(ZONAS)
    lotes, observaciones, climas = [], [], []

    for i in range(n_lotes):
        lote_id = f"L{i + 1:03d}"
        zona = nombres_zonas[i % len(nombres_zonas)]
        lat0, lon0, factor_zona = ZONAS[zona]
        lat, lon = lat0 + rng.uniform(-0.15, 0.15), lon0 + rng.uniform(-0.15, 0.15)
        geom = rectangulo_metros(lat, lon, rng.uniform(400, 700), rng.uniform(500, 900))
        suelo = rng.uniform(0.8, 1.1)
        primer_cultivo = rng.integers(2)

        for j, (campania, factor_camp) in enumerate(CAMPANIAS.items()):
            cultivo = ["soja", "maiz"][(primer_cultivo + j) % 2]
            anio = int(campania[:4])
            siembra = pd.Timestamp(f"{anio}-{SIEMBRA_MM_DD[cultivo]}") + pd.Timedelta(days=int(rng.normal(0, 8)))
            inicio = siembra - pd.Timedelta(days=15)
            fechas = pd.date_range(inicio, siembra + pd.Timedelta(days=DURACION_DIAS[cultivo]), freq="D")
            factor_lluvia = factor_camp * rng.uniform(0.85, 1.15)
            clima = _clima(rng, fechas, factor_lluvia)

            # Variables "verdaderas" que generan el rinde.
            t = (fechas - inicio).days.to_numpy()
            dia_pico = 15 + DIAS_AL_PICO[cultivo] + rng.normal(0, 5)
            normal_mm = PROB_LLUVIA * LLUVIA_MEDIA_MM
            temprana = clima["lluvia_mm"][t < dia_pico - 20].sum() / (normal_mm * max(dia_pico - 20, 1))
            en_critico = (t >= dia_pico - 20) & (t <= dia_pico + 30)
            agua_critico = clima["lluvia_mm"][en_critico].sum() / (normal_mm * 51)
            dias_calor = int((clima["tmax"][en_critico] > 35).sum())

            vigor = suelo * (0.6 + 0.4 * min(1.0, temprana))
            estres = float(np.clip(0.35 + 0.65 * min(1.0, agua_critico) - 0.02 * dias_calor, 0.2, 1.0))
            rinde = POTENCIAL_KG_HA[cultivo] * vigor * estres * factor_zona * rng.lognormal(0, 0.07)

            curva = _doble_logistica(t, 0.18, 0.72 * vigor * (0.85 + 0.15 * estres),
                                     dia_pico - 30, dia_pico + 30 + 25 * estres)
            observaciones.append(_observar(rng, lote_id, campania, fechas, curva, estres))
            climas.append(clima.assign(lote_id=lote_id, campania=campania))

            lotes.append({
                "lote_id": lote_id,
                "cultivo": cultivo,
                "campania": campania,
                "zona": zona,
                "fecha_siembra": siembra.date().isoformat(),
                "geometria": json.dumps(mapping(geom)),
                "rinde_kg_ha": round(rinde, -1),
            })

    lotes = pd.DataFrame(lotes)
    # ~10 % de lotes sin rinde informado, como pasaría con datos reales.
    lotes.loc[rng.random(len(lotes)) < 0.1, "rinde_kg_ha"] = np.nan
    obs = pd.concat(observaciones, ignore_index=True)
    clima = pd.concat(climas, ignore_index=True)
    return lotes, obs, clima[["lote_id", "campania", "fecha", "lluvia_mm", "tmax", "tmin", "radiacion_mj"]]


def _observar(rng, lote_id, campania, fechas, curva, estres):
    """Simula pasadas de Sentinel-2 cada 5 días con nubes."""
    idx = np.arange(rng.integers(5), len(fechas), 5)
    n = len(idx)
    frac = rng.uniform(0.85, 1.0, n)
    sesgo = np.ones(n)
    nublado = rng.random(n) < 0.35
    parcial = ~nublado & (rng.random(n) < 0.1)
    if rng.random() < 0.15:  # período nublado largo (~35 días)
        desde = rng.integers(n)
        nublado[desde:desde + 7] = True
    frac[nublado] = rng.uniform(0.0, 0.4, nublado.sum())
    sesgo[nublado] = rng.uniform(0.2, 0.6, nublado.sum())
    frac[parcial] = rng.uniform(0.4, 0.6, parcial.sum())
    sesgo[parcial] = rng.uniform(0.7, 0.9, parcial.sum())

    v = np.clip(curva[idx] * sesgo + rng.normal(0, 0.02, n), -0.1, 0.95)
    return pd.DataFrame({
        "lote_id": lote_id,
        "campania": campania,
        "fecha": fechas[idx],
        "ndvi": v.round(4),
        "evi": np.clip(0.85 * v - 0.06 + rng.normal(0, 0.02, n), -0.1, 0.9).round(4),
        "ndwi": np.clip(0.75 * v - 0.2 + 0.1 * (estres - 0.7) + rng.normal(0, 0.02, n), -0.5, 0.7).round(4),
        "frac_valida": frac.round(3),
        "n_pixeles": (frac * 3000).astype(int),
    })


def guardar(directorio, n_lotes: int = 60, semilla: int = 42) -> dict:
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    lotes, obs, clima = generar(n_lotes, semilla)
    rutas = {
        "lotes": directorio / "lotes_sinteticos.csv",
        "obs": directorio / "observaciones_sinteticas.csv",
        "clima": directorio / "clima_sintetico.csv",
    }
    lotes.to_csv(rutas["lotes"], index=False)
    obs.to_csv(rutas["obs"], index=False, date_format="%Y-%m-%d")
    clima.to_csv(rutas["clima"], index=False, date_format="%Y-%m-%d")
    return rutas
