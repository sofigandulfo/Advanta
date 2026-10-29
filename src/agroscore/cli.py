"""Línea de comandos: agroscore <comando> [opciones]. Ver README.md."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from agroscore import model, synthetic
from agroscore.clima import crear_proveedor_clima
from agroscore.config import RAIZ_PROYECTO, cargar_config
from agroscore.lotes import leer_lotes
from agroscore.pipeline import procesar
from agroscore.score import ParametrosScore, calcular_score, score_dataframe
from agroscore.sentinel import crear_proveedor

log = logging.getLogger("agroscore")
EJEMPLO = RAIZ_PROYECTO / "data" / "ejemplo"
SALIDA = RAIZ_PROYECTO / "outputs"
MODELO = RAIZ_PROYECTO / "models" / "modelo.joblib"


def cmd_generar_ejemplo(a):
    rutas = synthetic.guardar(a.salida, a.n_lotes, a.semilla)
    for nombre, ruta in rutas.items():
        log.info("%s -> %s", nombre, ruta)


def cmd_procesar(a):
    cfg = cargar_config(a.config)
    lotes = leer_lotes(a.lotes)
    prov_img = crear_proveedor(a.fuente_img, cfg, a.obs)
    prov_clima = crear_proveedor_clima(a.fuente_clima, a.clima)
    feats, series, obs = procesar(lotes, prov_img, prov_clima, cfg)
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    feats.to_csv(salida / "features.csv", index=False)
    series.to_csv(salida / "series.csv", index=False)
    obs.to_csv(salida / "observaciones.csv", index=False)
    errores = (feats["error"] != "").sum()
    log.info("Procesados %d lotes (%d con error) -> %s", len(feats), errores, salida)


def cmd_entrenar(a):
    feats = pd.read_csv(a.features, dtype={"lote_id": str, "campania": str})
    feats = feats[feats["error"].fillna("") == ""]
    paquete = model.entrenar(feats)
    model.guardar(paquete, a.modelo)

    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    paquete["predicciones_validacion"].to_csv(salida / "predicciones_validacion.csv", index=False)
    paquete["metricas_por_campania"].to_csv(salida / "metricas_por_campania.csv", index=False)
    resumen = {k: paquete[k] for k in ["tipo", "rmse_por_cultivo", "metricas_globales", "metricas_por_cultivo",
                                       "campanias", "cultivos", "n_lotes", "fecha_entrenamiento"]}
    (salida / "metricas.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("Validación por campaña (leave-one-campaign-out), %d lotes:", paquete["n_lotes"])
    for tipo, m in paquete["metricas_globales"].items():
        log.info("  %-8s MAE %6.0f kg/ha | RMSE %6.0f | R² %5.2f | MAPE %4.1f %%",
                 tipo, m["mae_kg_ha"], m["rmse_kg_ha"], m["r2"], m["mape_pct"])
    log.info("Modelo elegido: %s -> %s", paquete["tipo"], a.modelo)


def cmd_scorear(a):
    cfg = cargar_config(a.config)
    paquete = model.cargar(a.modelo)
    feats = pd.read_csv(a.features, dtype={"lote_id": str, "campania": str})
    feats = feats[feats["error"].fillna("") == ""].reset_index(drop=True)
    feats["rinde_modelo_final_kg_ha"] = model.predecir(paquete, feats)
    feats["rinde_estimado_kg_ha"] = feats["rinde_modelo_final_kg_ha"]
    feats["origen_estimacion"] = "modelo final"

    # Para lotes usados en el entrenamiento se usa la predicción fuera de muestra
    # (de cuando su campaña quedó excluida): así el backtest no es optimista.
    if not a.sin_validacion:
        oof = paquete["predicciones_validacion"][["lote_id", "campania", "rinde_validacion_kg_ha"]]
        feats = feats.merge(oof, on=["lote_id", "campania"], how="left")
        hay = feats["rinde_validacion_kg_ha"].notna()
        feats.loc[hay, "rinde_estimado_kg_ha"] = feats.loc[hay, "rinde_validacion_kg_ha"]
        feats.loc[hay, "origen_estimacion"] = "validación por campaña"

    scores = score_dataframe(feats, paquete["rmse_por_cultivo"], cfg)
    scores = agregar_riesgo_real(scores, cfg)
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    scores.to_csv(salida / "scores.csv", index=False)
    resumen = scores["riesgo"].value_counts().to_dict()
    log.info("Scores de %d lotes -> %s | riesgo: %s", len(scores), salida / "scores.csv", resumen)
    if scores["riesgo_real"].ne("sin dato").any():
        con_real = scores[scores["riesgo_real"] != "sin dato"]
        tabla = pd.crosstab(con_real["riesgo_real"], con_real["riesgo"], rownames=["real"], colnames=["estimado"])
        orden = [c for c in ["alto", "medio", "bajo"] if c in tabla.index.union(tabla.columns)]
        tabla = tabla.reindex(index=orden, columns=orden, fill_value=0)
        log.info("Riesgo estimado vs. riesgo con el rinde real (backtest):\n%s", tabla)
    log.info("Recordatorio: el score es una herramienta de apoyo, no una decisión de crédito.")


def agregar_riesgo_real(scores: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Cobertura y riesgo que habría tenido cada lote con su rinde real (sin margen por error).

    Permite medir si el score ordena bien: cuántos lotes de riesgo real alto quedaron como bajo, etc.
    """
    p = ParametrosScore.desde_config(cfg)
    real = [
        calcular_score(f["rinde_kg_ha"], 0.0, f["precio_usd_t"], f["costo_ref_usd_ha"], p)
        for _, f in scores.iterrows()
    ]
    return scores.assign(cobertura_real=[r["cobertura"] for r in real], riesgo_real=[r["riesgo"] for r in real])


def cmd_demo(a):
    if not (EJEMPLO / "lotes_sinteticos.csv").exists():
        cmd_generar_ejemplo(argparse.Namespace(salida=EJEMPLO, n_lotes=60, semilla=42))
    comunes = {"config": a.config, "salida": SALIDA}
    cmd_procesar(argparse.Namespace(lotes=EJEMPLO / "lotes_sinteticos.csv", fuente_img="archivo",
                                    obs=EJEMPLO / "observaciones_sinteticas.csv", fuente_clima="archivo",
                                    clima=EJEMPLO / "clima_sintetico.csv", **comunes))
    cmd_entrenar(argparse.Namespace(features=SALIDA / "features.csv", modelo=MODELO, salida=SALIDA))
    cmd_scorear(argparse.Namespace(features=SALIDA / "features.csv", modelo=MODELO, sin_validacion=False, **comunes))
    log.info("Listo. Dashboard: streamlit run app/dashboard.py")


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agroscore", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="comando", required=True)

    g = sub.add_parser("generar-ejemplo", help="Genera el dataset sintético")
    g.add_argument("--salida", default=EJEMPLO)
    g.add_argument("--n-lotes", type=int, default=60)
    g.add_argument("--semilla", type=int, default=42)
    g.set_defaults(func=cmd_generar_ejemplo)

    pr = sub.add_parser("procesar", help="Descarga/lee observaciones y clima y calcula variables por lote")
    pr.add_argument("--lotes", required=True)
    pr.add_argument("--fuente-img", choices=["archivo", "planetary", "copernicus"], default="planetary")
    pr.add_argument("--obs", help="CSV de observaciones (solo con --fuente-img archivo)")
    pr.add_argument("--fuente-clima", choices=["archivo", "nasa_power"], default="nasa_power")
    pr.add_argument("--clima", help="CSV de clima (solo con --fuente-clima archivo)")
    pr.add_argument("--salida", default=SALIDA)
    pr.add_argument("--config")
    pr.set_defaults(func=cmd_procesar)

    e = sub.add_parser("entrenar", help="Valida por campaña, elige y guarda el modelo")
    e.add_argument("--features", default=SALIDA / "features.csv")
    e.add_argument("--modelo", default=MODELO)
    e.add_argument("--salida", default=SALIDA)
    e.set_defaults(func=cmd_entrenar)

    s = sub.add_parser("scorear", help="Estima rinde y calcula el score de riesgo")
    s.add_argument("--features", default=SALIDA / "features.csv")
    s.add_argument("--modelo", default=MODELO)
    s.add_argument("--salida", default=SALIDA)
    s.add_argument("--config")
    s.add_argument("--sin-validacion", action="store_true",
                   help="Usar siempre el modelo final, aun para lotes que estuvieron en el entrenamiento")
    s.set_defaults(func=cmd_scorear)

    d = sub.add_parser("demo", help="Corre todo el circuito con el dataset sintético")
    d.add_argument("--config")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv=None):
    a = construir_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(message)s")
    np.seterr(all="ignore")
    a.func(a)


if __name__ == "__main__":
    main()
