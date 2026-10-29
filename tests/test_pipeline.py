"""Prueba de punta a punta con un dataset sintético chico (sin red)."""

import pytest

from agroscore import model, synthetic
from agroscore.clima import ProveedorClimaArchivo
from agroscore.config import cargar_config
from agroscore.lotes import leer_lotes
from agroscore.pipeline import procesar
from agroscore.score import score_dataframe
from agroscore.sentinel import ProveedorArchivo


@pytest.fixture(scope="module")
def resultado(tmp_path_factory):
    d = tmp_path_factory.mktemp("ejemplo")
    rutas = synthetic.guardar(d, n_lotes=12, semilla=1)
    cfg = cargar_config()
    lotes = leer_lotes(rutas["lotes"])
    feats, series, obs = procesar(lotes, ProveedorArchivo(rutas["obs"]), ProveedorClimaArchivo(rutas["clima"]), cfg)
    return cfg, lotes, feats, series, obs


def test_procesa_todos_los_lotes_sin_error(resultado):
    _, lotes, feats, series, _ = resultado
    assert len(feats) == len(lotes)
    assert (feats["error"] == "").all()
    assert feats["ndvi_pico"].between(0.3, 1.0).all()
    assert not series.empty


def test_observaciones_nubladas_no_se_usan(resultado):
    cfg, _, _, _, obs = resultado
    assert not obs.loc[obs["frac_valida"] < cfg["nubes"]["min_fraccion_valida"], "usada"].any()


def test_entrenar_y_scorear(resultado):
    cfg, _, feats, _, _ = resultado
    paquete = model.entrenar(feats)
    feats = feats.assign(rinde_estimado_kg_ha=model.predecir(paquete, feats))
    scores = score_dataframe(feats, paquete["rmse_por_cultivo"], cfg)
    assert scores["score"].between(0, 1000).all()
    assert set(scores["riesgo"]) <= {"alto", "medio", "bajo"}


def test_lotes_por_punto(tmp_path):
    csv = tmp_path / "lotes.csv"
    csv.write_text("lote_id,cultivo,campania,lat,lon,buffer_m\nA,Soja,2022/23,-33.9,-60.6,100\n")
    lotes = leer_lotes(csv)
    assert lotes.loc[0, "cultivo"] == "soja"
    assert lotes.loc[0, "area_ha"] == pytest.approx(4.0, rel=0.01)


def test_campania_invalida(tmp_path):
    csv = tmp_path / "lotes.csv"
    csv.write_text("lote_id,cultivo,campania,lat,lon\nA,soja,2022-23,-33.9,-60.6\n")
    with pytest.raises(ValueError, match="Campaña"):
        leer_lotes(csv)
